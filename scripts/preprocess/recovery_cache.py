"""Bind a CPU-validated alignment to its exact inputs and recovery implementation."""
import hashlib
import json
from pathlib import Path
import shutil

MODEL = ('cameras.bin', 'images.bin', 'points3D.bin')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def bind(source, project, compatibility):
    source, project = Path(source), Path(project)
    files = [source/'sparse'/n for n in MODEL]
    files += sorted(p for p in (source/'images').iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    files += sorted((source/'object_mask').glob('*.png'))
    for name in ['auto_rubik_keyframe.json', 'grounded_sam2_scene/summary.json']:
        if (source/name).is_file(): files.append(source/name)
    records = []
    for path in files:
        stat = path.stat()
        records.append({'path': str(path), 'size': stat.st_size,
                        'mtime_ns': stat.st_mtime_ns, 'sha256': sha(path)})
    code = [project/'align_aruco_refine.py', project/'scripts/preprocess/aruco_recovery.py']
    runtime = [Path(compatibility)/'cv2/__init__.py'] + list((Path(compatibility)/'cv2').glob('*.so'))
    return {'files': records, 'code': {str(p): sha(p) for p in code},
            'runtime': {str(p): sha(p) for p in runtime}}


def binding_current(binding):
    try:
        for record in binding['files']:
            stat = Path(record['path']).stat()
            if (stat.st_size, stat.st_mtime_ns) != (record['size'], record['mtime_ns']):
                return False
        for group in ['code', 'runtime']:
            if any(sha(path) != value for path, value in binding[group].items()):
                return False
        return True
    except (OSError, KeyError, ValueError):
        return False


def load(cache_root, scene):
    try:
        record = json.loads((Path(cache_root)/'index'/(scene+'.json')).read_text(encoding='utf-8'))
        if record['scene'] != scene or not binding_current(record['inputs']):
            return None
        if record['state'] == 'passed':
            if set(record['aligned_sha256']) != set(MODEL): return None
            for name, expected in record['aligned_sha256'].items():
                if sha(Path(record['aligned'])/name) != expected: return None
        return record
    except (OSError, ValueError, KeyError):
        return None


def apply(record, trial):
    if record['state'] != 'passed' or not binding_current(record['inputs']):
        raise ValueError('Alignment cache is not verified/current')
    target = Path(trial)/'sparse/aligned'
    target.mkdir()
    for name in MODEL:
        source = Path(record['aligned'])/name
        if sha(source) != record['aligned_sha256'][name]:
            raise ValueError('Alignment cache artifact changed')
        shutil.copy2(str(source), str(target/name))
    # Exactly match sha256sum's original-input, absolute-path stamp.
    (target/'input.sha256').write_text(''.join(
        sha(Path(trial)/'sparse'/name)+'  '+str(Path(trial)/'sparse'/name)+'\n'
        for name in MODEL), encoding='utf-8')
