"""Offline security/integrity tests; run with unittest discover."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from receive_results import MARKER, safe_relative, validate_manifest, local_path

spec = importlib.util.spec_from_file_location('publisher', ROOT / 'scripts/publish_result_manifest.py')
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


class ResultManifestTests(unittest.TestCase):
    def make_scene(self, base):
        scene = base / 'plant'
        for name in ['point_cloud/iteration_30000/point_cloud.ply',
                     'target_exports/strict/point_cloud.ply', 'mesh/mesh_nocolor.ply']:
            path = scene / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'reconstruction-data\x00\x01')
        (scene / 'empty').mkdir()
        return scene

    def test_full_manifest_and_empty_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = self.make_scene(Path(directory))
            result = publisher.publish(scene, 30000)
            self.assertEqual(3, len(result['files']))
            self.assertIn('empty', result['directories'])
            validate_manifest((scene / MARKER).read_bytes(), 'plant')

    def test_missing_product_invalidates_old_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = self.make_scene(Path(directory))
            publisher.publish(scene, 30000)
            (scene / 'mesh/mesh_nocolor.ply').unlink()
            with self.assertRaises(ValueError):
                publisher.publish(scene, 30000)
            self.assertFalse((scene / MARKER).exists())

    def test_reject_unsafe_windows_paths(self):
        for name in ['../outside', '/absolute', 'a/../b', 'a\\b', 'C:/file',
                     'file:stream', 'NUL.txt', 'a/CON', 'a.', 'a//b', 'a\x00b']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_relative(name)

    def test_manifest_rejects_case_collision_and_file_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = self.make_scene(Path(directory))
            manifest = publisher.publish(scene, 30000)
            record = dict(manifest['files'][0])
            record['path'] = record['path'].upper()
            manifest['files'].append(record)
            with self.assertRaises(ValueError):
                validate_manifest(json.dumps(manifest), 'plant')
            manifest['files'].pop()
            manifest['directories'].append(manifest['files'][0]['path'] + '/child')
            with self.assertRaises(ValueError):
                validate_manifest(json.dumps(manifest), 'plant')

    def test_atomic_manifest_detects_changed_output(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = self.make_scene(Path(directory))
            original = publisher.sha256
            def mutate(path):
                value = original(path)
                path.write_bytes(path.read_bytes() + b'changed')
                return value
            publisher.sha256 = mutate
            try:
                with self.assertRaises(RuntimeError):
                    publisher.publish(scene, 30000)
                self.assertFalse((scene / MARKER).exists())
            finally:
                publisher.sha256 = original


if __name__ == '__main__':
    unittest.main()
