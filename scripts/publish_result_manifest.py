#!/usr/bin/env python3
"""Publish an atomic, content-addressed receipt after all reconstruction stages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time
import uuid

MARKER = '.reconstruction_complete.json'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def snapshot(root):
    files, directories = [], []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root).as_posix()
        if relative in (MARKER, MARKER + '.tmp'):
            continue
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            directories.append(relative)
        elif stat.S_ISREG(info.st_mode):
            files.append((path, relative, info.st_size, info.st_mtime_ns))
        else:
            raise ValueError('Refusing symlink or special output: ' + str(path))
    return files, directories


def publish(root, iteration):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Output must be a real scene directory')
    marker = root / MARKER
    try:
        marker.unlink()
    except FileNotFoundError:
        pass
    # A historical empty output tree is not a completed reconstruction.
    required = [f'point_cloud/iteration_{iteration}/point_cloud.ply',
                'target_exports/strict/point_cloud.ply', 'mesh/mesh_nocolor.ply']
    for relative in required:
        path = root / relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError('Missing or empty reconstruction product: ' + str(path))
    before, directories = snapshot(root)
    records = [{'path': relative, 'size': size, 'sha256': sha256(path)}
               for path, relative, size, _ in before]
    after, after_directories = snapshot(root)
    if before != after or directories != after_directories:
        raise RuntimeError('Output changed while hashing; completion not published')
    manifest = {'version': 1, 'generation': uuid.uuid4().hex,
                'scene': root.name, 'iteration': iteration,
                'completed_at_unix': time.time(), 'files': records,
                'directories': directories}
    temporary = root / (MARKER + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, marker)
    print(f'RESULT_READY {root.name}: {len(records)} files, '
          f'{sum(record["size"] for record in records)} bytes')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-output', required=True)
    parser.add_argument('--iteration', type=int, required=True)
    arguments = parser.parse_args()
    publish(arguments.scene_output, arguments.iteration)
    from batch_boundary import checkpoint
    checkpoint(Path(__file__).resolve().parent.parent, Path(arguments.scene_output).name)
