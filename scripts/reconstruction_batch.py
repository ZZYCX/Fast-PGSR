"""Serial, provenance-checked batch coordinator; training commands are unchanged.

Only the standard library is required in the coordinator environment. Model
commands and semantic checks run in their original Conda environments.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

STAGES = ('Extract', 'Colmap', 'Mask', 'Align', 'Label', 'Train', 'Export', 'Render')
BINS = ('cameras.bin', 'images.bin', 'points3D.bin')
MARKER = '.reconstruction_complete.json'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def unlink(path):
    # Model environments still use Python 3.7 on the lab machines.
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def snapshot(paths):
    """Hash exact file sets, including empty dirs; reject aliases and special files."""
    records = {}
    for root in paths:
        if root.is_symlink() or not root.exists():
            raise ValueError('Missing or symlink input/output: ' + str(root))
        entries = [root] + (sorted(root.rglob('*')) if root.is_dir() else [])
        for path in entries:
            if path.is_symlink():
                raise ValueError('Symlink input/output: ' + str(path))
            if path.is_file():
                if path.stat().st_size == 0:
                    raise ValueError('Empty file: ' + str(path))
                records[str(path)] = digest(path)
            elif path.is_dir():
                records[str(path)] = 'directory'
            else:
                raise ValueError('Special file: ' + str(path))
    return records


def inside(child, parent):
    return child == parent or parent in child.parents


def configuration(env):
    cfg = dict(env)
    defaults = dict(GPU_ID='0', RUBIK_COUNT='1', PLANT_INDEX='0', EXPORT_MODE='strict',
                    FORCE='0', ITERATIONS='30000', FPS_TARGET='2.5', COLMAP_USE_GPU='1',
                    EXTRACT_USE_GPU='1', TRAIN_DATA_DEVICE='cuda', TRAIN_PRELOAD_IMG='1',
                    ALIGN_TIMEOUT_SECONDS='0', SCENE_TIMEOUT_SECONDS='5400',
                    RENDER_MAX_DEPTH='32.0', RENDER_VOXEL_SIZE='0.01', TARGET_SPATIAL_RADIUS='0.8',
                    COLMAP_ENV='fast-pgsr-old', ARUCO_ENV='Fast-Sam2', MASK_ENV='Fast-Sam2',
                    TRAIN_ENV='fast-pgsr-old', SCENE='', COLMAP_MAPPER_EXECUTABLE='',
                    COLMAP_RESUME_MAPPER='0')
    defaults.update({'RUN_' + stage.upper(): '1' for stage in STAGES})
    for key, value in defaults.items():
        cfg[key] = env.get(key) or value
    for key in ('FORCE', 'COLMAP_USE_GPU', 'EXTRACT_USE_GPU', 'TRAIN_PRELOAD_IMG',
                'COLMAP_RESUME_MAPPER') + tuple('RUN_' + s.upper() for s in STAGES):
        if cfg[key] not in ('0', '1'):
            raise ValueError(key + ' must be 0 or 1')
    for key, minimum in (('ITERATIONS', 1), ('PLANT_INDEX', 0),
                         ('ALIGN_TIMEOUT_SECONDS', 0), ('SCENE_TIMEOUT_SECONDS', 1)):
        if not cfg[key].isdigit() or int(cfg[key]) < minimum:
            raise ValueError('Invalid ' + key)
    for key in ('FPS_TARGET', 'RENDER_MAX_DEPTH', 'RENDER_VOXEL_SIZE', 'TARGET_SPATIAL_RADIUS'):
        value = float(cfg[key])
        if not math.isfinite(value) or value < 0 or (value == 0 and key != 'TARGET_SPATIAL_RADIUS'):
            raise ValueError('Invalid ' + key)
    if cfg['RUBIK_COUNT'] not in ('1', '2'):
        raise ValueError('RUBIK_COUNT must be 1 or 2')
    if cfg['EXPORT_MODE'] != 'strict':
        raise ValueError('Full reconstruction requires EXPORT_MODE=strict; use the standalone exporter for other modes')
    if cfg['SCENE'] and (cfg['SCENE'] in ('.', '..') or '/' in cfg['SCENE'] or '\\' in cfg['SCENE']):
        raise ValueError('Invalid SCENE')
    if not any(cfg['RUN_' + stage.upper()] == '1' for stage in STAGES):
        raise ValueError('No stages enabled')
    cfg['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES') or cfg['GPU_ID']
    return cfg


@contextlib.contextmanager
def locks(roots):
    import fcntl  # Production entry point is Linux; locks span the entire batch.
    with contextlib.ExitStack() as stack:
        for root in sorted(set(roots)):
            path = root / '.reconstruction-batch.lock'
            if path.is_symlink():
                raise ValueError('Symlink lock: ' + str(path))
            stream = stack.enter_context(path.open('a+'))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another batch owns ' + str(root))
        yield


def run_command(command, cwd, env, log, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Scene time budget exhausted')
    log.parent.mkdir(parents=True, exist_ok=True)
    print('[command] ' + ' '.join(map(str, command)), flush=True)
    with log.open('ab') as stream:
        child = subprocess.Popen(list(map(str, command)), cwd=str(cwd), env=env,
                                 stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=remaining)
            if code:
                raise RuntimeError('Command exit {}; see {}'.format(code, log))
        finally:
            # A failed/timed-out Conda wrapper must not leave CUDA children alive.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            else:
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                # The wrapper can exit before its grandchildren; kill the group.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            child.wait()


class Batch:
    def __init__(self, cfg):
        self.cfg = cfg
        self.project = Path(cfg.get('PROJECT_ROOT') or Path(__file__).resolve().parents[1]).resolve()
        def root(key, default):
            path = Path(cfg.get(key) or self.project / default)
            return (self.project / path).resolve() if not path.is_absolute() else path.resolve()
        self.data, self.output, self.logs = root('DATA_ROOT', 'data'), root('OUTPUT_ROOT', 'output'), root('LOG_ROOT', 'logs')
        self.videos = root('VIDEO_DIR', 'videos')
        if inside(self.data, self.output) or inside(self.output, self.data):
            raise ValueError('DATA_ROOT and OUTPUT_ROOT overlap')
        if inside(self.project, self.data) or inside(self.project, self.output):
            raise ValueError('Data/output root contains project source')
        if inside(self.logs, self.data) or inside(self.logs, self.output):
            raise ValueError('LOG_ROOT must be outside data/output roots (backups live there)')
        for path in (self.data, self.output, self.logs):
            path.mkdir(parents=True, exist_ok=True)
        self.run = self.logs / ('batch-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
        self.run.mkdir()
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=cfg['CUDA_VISIBLE_DEVICES'])
        self.status = {'version': 1, 'state': 'running', 'scenes': {}, 'config': {k: cfg[k] for k in cfg if k in ('ITERATIONS', 'FPS_TARGET', 'RUBIK_COUNT', 'PLANT_INDEX', 'CUDA_VISIBLE_DEVICES')}}

    def queue(self):
        videos = {}
        if self.cfg['RUN_EXTRACT'] == '1':
            for video in sorted(self.videos.glob('*')):
                if video.suffix.lower() not in ('.mp4', '.mov'):
                    continue
                if self.cfg['SCENE'] and video.stem != self.cfg['SCENE']:
                    continue
                if video.stem in videos:
                    raise ValueError('Duplicate video scene name: ' + video.stem)
                videos[video.stem] = video
        names = set(videos)
        names.update(p.name for p in self.data.iterdir() if p.is_dir() and not p.name.startswith('.'))
        if self.cfg['SCENE']:
            names &= {self.cfg['SCENE']}
        if not names:
            raise ValueError('No matching videos/scenes')
        return [(name, videos.get(name)) for name in sorted(names)]

    def command(self, env_key, script, *args, limit=0):
        deadline = min(self.deadline, time.monotonic() + limit) if limit else self.deadline
        command = ['conda', 'run', '--no-capture-output', '-n', self.cfg[env_key], 'python',
                   str(self.project / script)] + list(map(str, args))
        run_command(command, self.project, self.env, self.log, deadline)

    def validate(self, env, stage):
        args = [stage, '--scene', self.scene, '--rubik-count', self.cfg['RUBIK_COUNT']]
        if stage in ('trained', 'bind-trained'):
            args += ['--model', self.model, '--iteration', self.cfg['ITERATIONS']]
        if stage == 'masks':
            args += ['--require-instance-guard']
        self.command(env, 'scripts/validate_semantic_inputs.py', *args)

    def model_paths(self, directory):
        return [directory / name for name in BINS]

    def spec(self, stage):
        c, s, m = self.cfg, self.scene, self.model
        iteration = m / 'point_cloud' / ('iteration_' + c['ITERATIONS'])
        common = ['RUBIK_COUNT']
        table = {
            'Extract': ([self.video] if self.video else [], [s / 'input'], ['FPS_TARGET', 'EXTRACT_USE_GPU'], 'scripts/preprocess/extract_frames.py'),
            'Colmap': ([s / 'input'], self.model_paths(s / 'sparse') + [s / 'images'], ['COLMAP_USE_GPU', 'COLMAP_MAPPER_EXECUTABLE', 'COLMAP_RESUME_MAPPER'], 'scripts/preprocess/convert.py'),
            'Mask': ([s / 'images'] + self.model_paths(s / 'sparse'), [s / 'object_mask', s / 'grounded_sam2_scene/summary.json'], common + ['PLANT_INDEX'], 'scripts/mask_generation/run_rubik_scene_auto.py'),
            'Align': (self.model_paths(s / 'sparse') + [s / 'images'], self.model_paths(s / 'sparse/aligned'), [], 'align_aruco_refine.py'),
            'Label': ([s / 'images', s / 'object_mask'] + self.model_paths(s / ('sparse/aligned' if c['RUN_ALIGN'] == '1' else 'sparse/0')), [s / 'semantic_initialization'] + self.model_paths(s / 'sparse/0'), common, 'scripts/mask_generation/label_colmap_points_viewer.py'),
            'Train': ([s / 'images', s / 'object_mask', s / 'semantic_initialization'] + self.model_paths(s / 'sparse/0'), [iteration, m / 'semantic_input_binding.json', m / 'cfg_args'], common + ['ITERATIONS', 'TRAIN_DATA_DEVICE', 'TRAIN_PRELOAD_IMG'], 'train.py'),
            'Export': ([iteration, m / 'semantic_input_binding.json'], [m / 'target_exports'], common + ['ITERATIONS', 'TARGET_SPATIAL_RADIUS', 'EXPORT_MODE'], 'scripts/export_target_gaussians.py'),
            'Render': ([m / 'target_exports/strict', m / 'cfg_args', s / 'images', s / 'object_mask'] + self.model_paths(s / 'sparse/0'), [m / 'mesh', m / 'train', m / 'test', m / 'render_target.json'], ['ITERATIONS', 'RENDER_MAX_DEPTH', 'RENDER_VOXEL_SIZE'], 'render.py'),
        }
        inputs, outputs, keys, script = table[stage]
        parameters = {key: c[key] for key in keys}
        # Opt-in recovery extraction modes also change the frame selection.
        if stage == 'Extract':
            parameters.update({key: self.env.get(key, '') for key in ('ARUCO_RETAINED_MANIFEST', 'ARUCO_NATIVE_FRAMES')})
            if parameters['ARUCO_RETAINED_MANIFEST']:
                inputs += [Path(parameters['ARUCO_RETAINED_MANIFEST'])]
            if parameters['ARUCO_RETAINED_MANIFEST']:
                manifest = Path(parameters['ARUCO_RETAINED_MANIFEST'])
                rows = json.loads(manifest.read_text(encoding='utf-8'))
                inputs += [manifest.parent / row['image'] for row in rows if row['scene'] == self.video.stem]
        dependencies = [self.project / script]
        if stage == 'Mask':
            dependencies += [self.project / 'scripts/mask_generation' / name for name in
                             ('run_grounded_sam2_scene.py', 'instance_mask_guard.py')]
        if stage in ('Train', 'Render'):
            for directory in ('arguments', 'scene', 'utils', 'gaussian_renderer'):
                dependencies += sorted((self.project / directory).rglob('*.py'))
        return {'parameters': parameters, 'inputs': snapshot(inputs),
                'scripts': snapshot(dependencies)}, outputs

    def archive(self, paths, stage):
        for index, path in enumerate(paths):
            if not path.exists() and not path.is_symlink():
                continue
            if path.is_symlink() or not (inside(path.resolve(), self.scene) or inside(path.resolve(), self.model)):
                raise ValueError('Refusing to archive alias/outside scene: ' + str(path))
            owner = self.data if inside(path.resolve(), self.scene) else self.output
            target = owner / '.batch-backups' / self.run.name / self.scene.name / stage / ('{}-{}'.format(index, path.name))
            target.parent.mkdir(parents=True, exist_ok=True)
            # os.rename is atomic on the same filesystem; refuse cross-device
            # copies rather than fill a disk or leave partially moved datasets.
            path.rename(target)
            print('[backup] {} -> {}'.format(path, target), flush=True)

    def prepare(self, stage):
        s, m = self.scene, self.model
        paths = {
            'Extract': [s / 'input'],
            'Colmap': [s / name for name in ('images', 'sparse', 'stereo', 'distorted', 'images_2', 'images_4', 'images_8', 'database.db', 'database.db-journal', 'database.db-wal', 'database.db-shm')],
            'Mask': [s / name for name in ('object_mask', 'grounded_sam2_scene', 'color_mask', 'auto_rubik_keyframe.json')],
            'Align': [s / 'sparse/aligned', s / 'aruco_alignment_report.json'],
            'Label': [s / 'semantic_initialization'] + ([s / 'sparse/0'] if self.cfg['RUN_ALIGN'] == '1' else []),
            'Train': [m], 'Export': [m / 'target_exports'],
            'Render': [m / name for name in ('mesh', 'train', 'test', 'render_target.json')],
        }[stage]
        if stage == 'Colmap' and self.cfg['COLMAP_RESUME_MAPPER'] == '1':
            paths = [p for p in paths if p.name != 'database.db']
        self.archive(paths, stage)

    def execute(self, stage):
        c, s, m = self.cfg, self.scene, self.model
        if stage == 'Extract':
            self.command('COLMAP_ENV', 'scripts/preprocess/extract_frames.py', '--video', self.video,
                         '--output_dir', s / 'input', '--fps', c['FPS_TARGET'], '--gpu_index', 0,
                         *(['--no_gpu'] if c['EXTRACT_USE_GPU'] == '0' else []))
        elif stage == 'Colmap':
            self.command('COLMAP_ENV', 'scripts/preprocess/convert.py', '--scene_path', s, '--camera', 'OPENCV', '--gpu_index', 0,
                         '--mapper_executable', c['COLMAP_MAPPER_EXECUTABLE'],
                         *(['--resume_mapper'] if c['COLMAP_RESUME_MAPPER'] == '1' else []),
                         *(['--no_gpu'] if c['COLMAP_USE_GPU'] == '0' else []))
        elif stage == 'Mask':
            self.command('MASK_ENV', 'scripts/mask_generation/run_rubik_scene_auto.py', '--scene', s,
                         '--rubik-count', c['RUBIK_COUNT'], '--plant-index', c['PLANT_INDEX'],
                         '--scan-step', 5, '--refine-top-k', 5, '--box-threshold', '0.30', '--text-threshold', '0.25',
                         '--nms-threshold', '0.72', '--direction', 'both', '--fill-hole-area', 0)
            self.validate('MASK_ENV', 'masks')
        elif stage == 'Align':
            self.command('ARUCO_ENV', 'align_aruco_refine.py', s, limit=int(c['ALIGN_TIMEOUT_SECONDS']))
        elif stage == 'Label':
            self.validate('MASK_ENV', 'masks')
            if c['RUN_ALIGN'] == '1':
                (s / 'sparse/0').mkdir(parents=True, exist_ok=True)
                for name in BINS:
                    shutil.copy2(str(s / 'sparse/aligned' / name), str(s / 'sparse/0' / name))
            self.command('MASK_ENV', 'scripts/mask_generation/label_colmap_points_viewer.py', '--scene', s,
                         '--patch-size', 5, '--target-dilation', 5, '--min-views', 3,
                         '--target-confidence', '0.70', '--background-confidence', '0.90', '--no-visualize')
            self.validate('MASK_ENV', 'bind')
        elif stage == 'Train':
            self.validate('TRAIN_ENV', 'init')
            self.command('TRAIN_ENV', 'train.py', '-s', s, '-m', m, '--iterations', c['ITERATIONS'],
                         '--resolution', 1, '--data_device', c['TRAIN_DATA_DEVICE'], '--ncc_scale', '0.5',
                         *(['--disable_preload_img'] if c['TRAIN_PRELOAD_IMG'] == '0' else []),
                         '--densification_interval', 500, '--opacity_cull_threshold', '0.05',
                         '--exposure_compensation', '--semantic_pipeline')
            self.validate('TRAIN_ENV', 'bind-trained')
        elif stage == 'Export':
            self.validate('TRAIN_ENV', 'trained')
            self.command('TRAIN_ENV', 'scripts/export_target_gaussians.py',
                         '--iteration_dir', m / 'point_cloud' / ('iteration_' + c['ITERATIONS']),
                         '--output_dir', m / 'target_exports', '--mode', 'strict', '--rubik-count', c['RUBIK_COUNT'],
                         '--unknown_target_threshold', '0.90', '--export_per_class', '--export_debug_cloud',
                         '--spatial-radius', c['TARGET_SPATIAL_RADIUS'])
        elif stage == 'Render':
            self.validate('TRAIN_ENV', 'trained')
            self.command('TRAIN_ENV', 'render.py', '-m', m, '--iteration', c['ITERATIONS'],
                         '--max_depth', c['RENDER_MAX_DEPTH'], '--voxel_size', c['RENDER_VOXEL_SIZE'],
                         '--target_ply', m / 'target_exports/strict/point_cloud.ply')

    def check_outputs(self, stage, outputs):
        records = snapshot(outputs)
        if stage == 'Extract' and not any(p.suffix.lower() in ('.jpg', '.jpeg', '.png') for p in (self.scene / 'input').iterdir()):
            raise ValueError('No extracted frames')
        if stage == 'Export':
            snapshot([self.model / 'target_exports/strict/point_cloud.ply', self.model / 'target_exports/strict/semantic_state.npz'])
        if stage == 'Render':
            renders = self.model / 'train' / ('ours_' + self.cfg['ITERATIONS']) / 'renders'
            if not renders.is_dir() or not any(p.is_file() for p in renders.iterdir()):
                raise ValueError('Missing rendered training views')
            # A nonempty PLY header is not a usable mesh. This parser needs no
            # Open3D/CUDA and works for ASCII and binary PLY headers.
            mesh = self.model / 'mesh/mesh_nocolor.ply'
            with mesh.open('rb') as stream:
                counts = {}
                for _ in range(1000):
                    line = stream.readline().decode('ascii').strip()
                    if line == 'end_header':
                        break
                    if line.startswith('element '):
                        _, name, count = line.split()
                        counts[name] = int(count)
                else:
                    raise ValueError('Invalid mesh PLY header')
            if counts.get('vertex', 0) <= 0 or counts.get('face', 0) <= 0:
                raise ValueError('Empty reconstructed mesh')
        return records

    def stage(self, stage):
        signature, outputs = self.spec(stage)
        receipt = self.scene / '.batch-state' / (stage + '.json')
        cached = False
        if self.cfg['FORCE'] == '0':
            try:
                old = json.loads(receipt.read_text(encoding='utf-8'))
                cached = old['signature'] == signature and old['outputs'] == self.check_outputs(stage, outputs)
            except (OSError, ValueError, KeyError, TypeError):
                pass
        if cached:
            print('[reuse] {} {}'.format(self.scene.name, stage), flush=True)
            return 'reused'
        # Remove completion eligibility before moving or changing any product.
        unlink(self.model / MARKER)
        unlink(receipt)
        self.prepare(stage)
        self.execute(stage)
        records = self.check_outputs(stage, outputs)
        after, _ = self.spec(stage)
        if after != signature:
            raise RuntimeError('Inputs changed while running ' + stage)
        save(receipt, {'version': 1, 'signature': signature, 'outputs': records})
        return 'completed'

    def complete(self):
        try:
            receipt = json.loads((self.model / MARKER).read_text(encoding='utf-8'))
            if receipt['version'] != 1 or receipt['scene'] != self.scene.name or receipt['iteration'] != int(self.cfg['ITERATIONS']):
                return False
            from publish_result_manifest import snapshot as result_snapshot
            files, directories = result_snapshot(self.model)
            records = [{'path': relative, 'size': size, 'sha256': digest(path)} for path, relative, size, _ in files]
            return records == receipt['files'] and directories == receipt['directories']
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def process(self, name, video):
        self.scene, self.model, self.video = self.data / name, self.output / name, video
        if self.scene.is_symlink() or self.model.is_symlink():
            raise ValueError('Symlink scene/output refused')
        self.scene.mkdir(exist_ok=True)
        self.deadline = time.monotonic() + int(self.cfg['SCENE_TIMEOUT_SECONDS'])
        record = self.status['scenes'][name] = {'state': 'running', 'stages': {}}
        save(self.run / 'status.json', self.status)
        for stage in STAGES:
            if self.cfg['RUN_' + stage.upper()] != '1' or (stage == 'Extract' and video is None):
                continue
            record['stage'] = stage
            save(self.run / 'status.json', self.status)
            self.log = self.run / name / (stage + '.log')
            print('[{}] {}'.format(stage, name), flush=True)
            record['stages'][stage] = self.stage(stage)
        if all(self.cfg['RUN_' + s] == '1' for s in ('TRAIN', 'EXPORT', 'RENDER')):
            if not self.complete():
                self.log = self.run / name / 'ResultManifest.log'
                self.command('TRAIN_ENV', 'scripts/publish_result_manifest.py', '--scene-output', self.model,
                             '--iteration', self.cfg['ITERATIONS'])
                if not self.complete():
                    raise ValueError('Published completion manifest failed verification')
            record['state'] = 'complete'
        else:
            # Partial workflows never leave a stale full-reconstruction receipt.
            unlink(self.model / MARKER)
            record['state'] = 'partial'
        save(self.run / 'status.json', self.status)

    def main(self):
        failures = 0
        print('Project: {}\nVideos: {}\nData: {}\nOutput: {}\nGPU visible: {}\nIterations: {}\nFPS: {}'.format(
            self.project, self.videos, self.data, self.output, self.cfg['CUDA_VISIBLE_DEVICES'],
            self.cfg['ITERATIONS'], self.cfg['FPS_TARGET']), flush=True)
        with locks((self.data, self.output)):
            for name, video in self.queue():
                try:
                    self.process(name, video)
                except Exception as error:
                    failures += 1
                    record = self.status['scenes'].setdefault(name, {})
                    record.update(state='failed', error='{}: {}'.format(type(error).__name__, error))
                    print('[failed] {}: {}'.format(name, error), file=sys.stderr, flush=True)
                save(self.run / 'status.json', self.status)
            self.status.update(state='failed' if failures else 'finished', failures=failures)
            save(self.run / 'status.json', self.status)
        print('Finished; failed scenes={}; logs={}'.format(failures, self.run), flush=True)
        return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check queue and Conda environments without running reconstruction')
    args = parser.parse_args()
    def stop(signum, frame):
        raise KeyboardInterrupt('Signal ' + str(signum))
    signal.signal(signal.SIGTERM, stop)
    batch = None
    try:
        cfg = configuration(os.environ)
        if not shutil.which('conda'):
            raise ValueError('conda not found on PATH')
        batch = Batch(cfg)
        if args.check:
            batch.deadline = time.monotonic() + 120
            batch.log = batch.run / 'preflight.log'
            seen = set()
            for key in ('COLMAP_ENV', 'MASK_ENV', 'ARUCO_ENV', 'TRAIN_ENV'):
                if cfg[key] in seen:
                    continue
                seen.add(cfg[key])
                run_command(['conda', 'run', '--no-capture-output', '-n', cfg[key], 'python', '-c',
                             'import sys, numpy, PIL; print(sys.executable)'], batch.project,
                            batch.env, batch.log, batch.deadline)
            names = [name for name, _ in batch.queue()]
            batch.status.update(state='preflight_passed', queued_scenes=names)
            save(batch.run / 'status.json', batch.status)
            print('Preflight passed; queued scenes={}; logs={}'.format(len(names), batch.run))
            return 0
        return batch.main()
    except KeyboardInterrupt:
        if batch:
            batch.status['state'] = 'interrupted'
            save(batch.run / 'status.json', batch.status)
        return 130
    except Exception as error:
        print('{}: {}'.format(type(error).__name__, error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
