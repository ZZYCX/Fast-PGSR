"""CPU-only coordinator regressions. No CUDA/model packages are imported."""
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reconstruction_batch as batch
from publish_result_manifest import publish


def write(path, value=b'content'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


class FakeBatch(batch.Batch):
    """Fake expensive models, exercise real scheduling/provenance/file operations."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.calls, self.fail, self.empty_mesh = [], None, False

    def execute(self, stage):
        self.calls.append((self.scene.name, stage))
        if self.fail == (self.scene.name, stage):
            if stage == 'Export':
                write(self.model / 'target_exports/partial.ply')
            raise RuntimeError('simulated interrupted stage')
        s, m = self.scene, self.model
        if stage == 'Extract':
            write(s / 'input/frame.jpg', self.video.read_bytes() + self.cfg['FPS_TARGET'].encode())
        elif stage == 'Colmap':
            source = (s / 'input/frame.jpg').read_bytes()
            write(s / 'images/frame.jpg', source)
            for name in batch.BINS:
                write(s / 'sparse' / name, source)
        elif stage == 'Mask':
            write(s / 'object_mask/frame.png', (s / 'images/frame.jpg').read_bytes())
            write(s / 'grounded_sam2_scene/summary.json', b'{}')
        elif stage == 'Align':
            for name in batch.BINS:
                write(s / 'sparse/aligned' / name, (s / 'sparse' / name).read_bytes())
        elif stage == 'Label':
            for name in batch.BINS:
                write(s / 'sparse/0' / name, (s / 'sparse/aligned' / name).read_bytes())
            write(s / 'semantic_initialization/point_semantics.npz', (s / 'images/frame.jpg').read_bytes())
            write(s / 'semantic_initialization/input_binding.json', b'{}')
        elif stage == 'Train':
            iteration = m / 'point_cloud' / ('iteration_' + self.cfg['ITERATIONS'])
            for name in ('point_cloud.ply', 'semantic_state.npz'):
                write(iteration / name, (s / 'images/frame.jpg').read_bytes())
            write(m / 'semantic_input_binding.json', b'{}')
            write(m / 'cfg_args', b'Namespace()')
        elif stage == 'Export':
            for name in ('point_cloud.ply', 'semantic_state.npz'):
                write(m / 'target_exports/strict' / name)
        elif stage == 'Render':
            n = '0' if self.empty_mesh else '1'
            write(m / 'mesh/mesh_nocolor.ply', ('ply\nformat ascii 1.0\nelement vertex '+n+'\nelement face '+n+'\nend_header\n0 0 0\n3 0 0 0\n').encode())
            write(m / 'render_target.json', b'{}')
            write(m / 'train' / ('ours_' + self.cfg['ITERATIONS']) / 'renders/frame.jpg')
            (m / 'test').mkdir()

    def command(self, env_key, script, *args, **kwargs):
        if script == 'scripts/publish_result_manifest.py':
            publish(self.model, int(self.cfg['ITERATIONS']))
        else:
            raise AssertionError('Unexpected external command')


class BatchRegression(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = {'PROJECT_ROOT': str(Path(__file__).resolve().parents[1]),
                    'DATA_ROOT': str(self.root / 'data'), 'OUTPUT_ROOT': str(self.root / 'output'),
                    'VIDEO_DIR': str(self.root / 'videos'), 'LOG_ROOT': str(self.root / 'logs')}
        write(self.root / 'videos/plant.mp4', b'video-a')

    def tearDown(self):
        self.temp.cleanup()

    def make(self, **changes):
        return FakeBatch(batch.configuration(dict(self.env, **changes)))

    def process(self, worker, scene='plant'):
        worker.process(scene, self.root / ('videos/' + scene + '.mp4'))

    def test_completed_scene_reuses_all_verified_stages(self):
        self.process(self.make())
        worker = self.make()
        self.process(worker)
        self.assertEqual(worker.calls, [])
        self.assertEqual(worker.status['scenes']['plant']['state'], 'complete')

    def test_invalid_receipt_is_not_success_and_is_republished(self):
        worker = self.make()
        write(worker.output / 'plant' / batch.MARKER, b'{}')
        self.process(worker)
        self.assertEqual(len(worker.calls), 8)
        write(worker.output / 'plant' / batch.MARKER, b'{broken')
        again = self.make()
        self.process(again)
        self.assertEqual(again.calls, [])
        self.assertTrue(again.complete())

    def test_missing_or_corrupt_result_repairs_without_retraining(self):
        self.process(self.make())
        (self.root / 'output/plant/target_exports/strict/point_cloud.ply').unlink()
        worker = self.make()
        self.process(worker)
        self.assertIn(('plant', 'Export'), worker.calls)
        self.assertNotIn(('plant', 'Train'), worker.calls)
        # Identical repaired exports do not invalidate an unchanged valid mesh.
        write(self.root / 'output/plant/mesh/mesh_nocolor.ply', b'corrupt')
        again = self.make()
        self.process(again)
        self.assertEqual(again.calls, [('plant', 'Render')])

    def test_interrupted_export_retries_only_unfinished_stages(self):
        worker = self.make()
        worker.fail = ('plant', 'Export')
        with self.assertRaises(RuntimeError):
            self.process(worker)
        self.assertFalse((worker.output / 'plant' / batch.MARKER).exists())
        again = self.make()
        self.process(again)
        self.assertEqual(again.calls, [('plant', 'Export'), ('plant', 'Render')])
        self.assertTrue(list(again.output.glob('.batch-backups/**/partial.ply')))

    def test_partial_training_is_preserved_and_restarted(self):
        worker = self.make()
        worker.fail = ('plant', 'Train')
        with self.assertRaises(RuntimeError):
            self.process(worker)
        write(worker.output / 'plant/chkpnt.pth', b'partial training')
        again = self.make()
        self.process(again)
        self.assertEqual(again.calls[0], ('plant', 'Train'))
        self.assertTrue(list(again.output.glob('.batch-backups/**/chkpnt.pth')))

    def test_fps_video_and_missing_frames_invalidate_downstream(self):
        self.process(self.make())
        for changes in ({'FPS_TARGET': '3.5'}, {}):
            if not changes:
                write(self.root / 'videos/plant.mp4', b'changed-video')
            worker = self.make(**changes)
            self.process(worker)
            self.assertEqual([s for _, s in worker.calls], list(batch.STAGES))
        (self.root / 'data/plant/input/frame.jpg').unlink()
        worker = self.make()
        self.process(worker)
        self.assertIn(('plant', 'Extract'), worker.calls)

    def test_iteration_change_preserves_preprocessing(self):
        self.process(self.make())
        worker = self.make(ITERATIONS='40000')
        self.process(worker)
        self.assertEqual(worker.calls, [('plant', 'Train'), ('plant', 'Export'), ('plant', 'Render')])
        self.assertEqual(json.loads((worker.model / batch.MARKER).read_text())['iteration'], 40000)

    def test_render_parameter_change_does_not_retrain_or_export(self):
        self.process(self.make())
        worker = self.make(RENDER_VOXEL_SIZE='0.02')
        self.process(worker)
        self.assertEqual(worker.calls, [('plant', 'Render')])

    def test_partial_workflow_does_not_claim_complete(self):
        self.process(self.make())
        worker = self.make(RUN_RENDER='0')
        self.process(worker)
        self.assertEqual(worker.status['scenes']['plant']['state'], 'partial')
        self.assertFalse((worker.model / batch.MARKER).exists())

    def test_missing_rendered_view_is_repaired(self):
        self.process(self.make())
        (self.root / 'output/plant/train/ours_30000/renders/frame.jpg').unlink()
        worker = self.make()
        self.process(worker)
        self.assertEqual(worker.calls, [('plant', 'Render')])

    def test_empty_mesh_is_rejected(self):
        worker = self.make()
        worker.empty_mesh = True
        with self.assertRaisesRegex(ValueError, 'Empty reconstructed mesh'):
            self.process(worker)
        self.assertFalse((worker.model / batch.MARKER).exists())

    def test_strict_mode_and_numeric_preflight(self):
        for changes in ({'EXPORT_MODE': 'both'}, {'EXPORT_MODE': 'conservative'},
                        {'FPS_TARGET': '0.0'}, {'RENDER_VOXEL_SIZE': 'nan'},
                        {'SCENE': '../outside'}, {'ITERATIONS': '0'}):
            with self.assertRaises(ValueError):
                batch.configuration(dict(self.env, **changes))

    def test_video_name_cannot_alias_data_or_output_root(self):
        write(self.root / 'videos/...mp4', b'video')
        worker = self.make()
        with self.assertRaisesRegex(ValueError, 'Unsafe video scene name'):
            worker.queue()
        with self.assertRaisesRegex(ValueError, 'Unsafe scene name'):
            worker.process('..', None)

    def test_scene_serial_failure_isolation_and_exit_status(self):
        write(self.root / 'videos/second.mp4', b'video-b')
        worker = self.make()
        worker.fail = ('plant', 'Mask')
        with patch.object(batch, 'locks', lambda roots: contextlib.nullcontext()):
            self.assertEqual(worker.main(), 1)
        self.assertEqual(worker.calls[:3], [('plant', 'Extract'), ('plant', 'Colmap'), ('plant', 'Mask')])
        self.assertEqual(worker.calls[3:], [('second', s) for s in batch.STAGES])
        self.assertEqual(worker.status['scenes']['second']['state'], 'complete')

    def test_training_command_preserves_original_parameters(self):
        worker = self.make()
        worker.scene = self.root / 'data/plant'
        worker.model = self.root / 'output/plant'
        with patch.object(worker, 'validate') as validate, patch.object(worker, 'command') as command:
            batch.Batch.execute(worker, 'Train')
        self.assertEqual(validate.call_args_list[0][0], ('TRAIN_ENV', 'init'))
        self.assertEqual(validate.call_args_list[1][0], ('TRAIN_ENV', 'bind-trained'))
        self.assertEqual(command.call_args[0], ('TRAIN_ENV', 'train.py', '-s', worker.scene, '-m', worker.model,
            '--iterations', '30000', '--resolution', 1, '--data_device', 'cuda', '--ncc_scale', '0.5',
            '--densification_interval', 500, '--opacity_cull_threshold', '0.05', '--exposure_compensation', '--semantic_pipeline'))

    @unittest.skipUnless(os.name == 'posix', 'Linux process groups and flock')
    def test_concurrent_batch_is_rejected(self):
        worker = self.make()
        with batch.locks((worker.data, worker.output)):
            with self.assertRaises(RuntimeError):
                with batch.locks((worker.data, worker.output)):
                    pass

    @unittest.skipUnless(os.name == 'posix', 'Linux process groups')
    def test_timeout_kills_grandchild_and_allows_next_scene(self):
        pidfile = self.root / 'child.pid'
        code = ('import subprocess,time; from pathlib import Path; '
                'p=subprocess.Popen(["sleep","60"]); Path({!r}).write_text(str(p.pid)); time.sleep(60)').format(str(pidfile))
        with self.assertRaises(subprocess.TimeoutExpired):
            batch.run_command([sys.executable, '-c', code], self.root, os.environ.copy(),
                              self.root / 'timeout.log', time.monotonic() + 1)
        pid = int(pidfile.read_text())
        # A terminated orphan can briefly remain as a zombie until init reaps it.
        stat = Path('/proc/{}/stat'.format(pid))
        self.assertTrue(not stat.exists() or stat.read_text().rsplit(') ', 1)[1].startswith('Z '))


if __name__ == '__main__':
    unittest.main(verbosity=2)
