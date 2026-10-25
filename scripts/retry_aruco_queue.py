#!/usr/bin/env python3
"""Run a saved retry plan only when existing Fast-PGSR jobs have exited."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(str(temporary), str(path))


def active_jobs(project):
    jobs = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            arguments = (process / 'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            if (os.readlink(str(process / 'cwd')) == str(project)
                    and any(os.path.basename(argument) in
                            ('run_batch_gpu1.sh', 'train.py', 'render.py') for argument in arguments)):
                jobs.append(int(process.name))
        except (OSError, ValueError):
            continue
    return jobs


def result_valid(project, scene):
    output = project / 'output' / scene
    try:
        manifest = json.loads((output / '.reconstruction_complete.json').read_text(encoding='utf-8'))
        if manifest.get('scene') != scene or not manifest.get('files'):
            return False
        for record in manifest['files']:
            relative = Path(record['path'])
            if relative.is_absolute() or '..' in relative.parts:
                return False
            path = output / relative
            if path.is_symlink() or not path.is_file() or path.stat().st_size != record['size']:
                return False
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != record['sha256']:
                return False
        return True
    except (OSError, ValueError, KeyError):
        return False


def run(plan_file, check=False):
    plan = json.loads(plan_file.read_text(encoding='utf-8'))
    project, root = Path(plan['project_root']).resolve(), plan_file.parent.resolve()
    if check:
        jobs = active_jobs(project)
        print(json.dumps({'state': 'waiting_main_batch' if jobs else 'ready', 'active_pids': jobs}))
        return 0
    with (root / 'queue.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'pid': os.getpid(), 'state': 'waiting_main_batch',
                 'jobs': [dict(job, state='pending') for job in plan['jobs']]}
        for job in state['jobs']:
            # Recheck between plants too: a newly started main batch has priority.
            while True:
                active = active_jobs(project)
                if not active:
                    break
                state.update(state='waiting_main_batch', active_pids=active,
                             updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
                write_json(root / 'queue-status.json', state)
                time.sleep(30)
            scene = job['scene']
            if result_valid(project, scene):
                job.update(state='already_complete')
                write_json(root / 'queue-status.json', state)
                continue
            environment = os.environ.copy()
            environment['PATH'] = plan['conda_base'] + '/bin:' + environment['PATH']
            environment.update({'ARUCO_RECOVERY': '1', 'RENDER_VOXEL_SIZE': '0.02',
                                'SCENE': scene, 'FORCE': '0', 'RUN_ALIGN': '1',
                                'RUN_LABEL': '1', 'RUN_TRAIN': '1', 'RUN_EXPORT': '1', 'RUN_RENDER': '1'})
            if job['mode'] == 'resume':
                environment.update({'RUN_EXTRACT': '0', 'RUN_COLMAP': '0', 'RUN_MASK': '0'})
            elif job['mode'] == 'dense':
                environment.update({'RUN_EXTRACT': '1', 'RUN_COLMAP': '1', 'RUN_MASK': '1',
                                    'DATA_ROOT': plan['dense_data_root'], 'FPS_TARGET': '10'})
            else:
                raise ValueError('Unknown retry mode')
            # Avoid accidental inheritance of alternate output/data roots.
            environment['OUTPUT_ROOT'] = str(project / 'output')
            environment['PROJECT_ROOT'] = str(project)
            environment['VIDEO_DIR'] = str(project / 'videos')
            environment['LOG_ROOT'] = str(project / 'logs')
            if job['mode'] == 'resume':
                environment['DATA_ROOT'] = str(project / 'data')
            logfile = root / ('retry-' + scene + '.log')
            job.update(state='running', log=str(logfile))
            state.update(state='running', active_pids=[], updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
            write_json(root / 'queue-status.json', state)
            print('RETRY_START ' + scene + ' mode=' + job['mode'], flush=True)
            with logfile.open('ab', buffering=0) as output:
                process = subprocess.Popen(['bash', 'run_batch_gpu1.sh'], cwd=str(project),
                                           env=environment, stdout=output, stderr=subprocess.STDOUT)
                code = process.wait()
            valid = result_valid(project, scene)
            job.update(state='complete' if code == 0 and valid else 'failed',
                       exit_code=code, verified_result=valid)
            state['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            write_json(root / 'queue-status.json', state)
            print('RETRY_END ' + scene + ' state=' + job['state'], flush=True)
        state.update(state='finished', updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        write_json(root / 'queue-status.json', state)
        return 1 if any(job['state'] == 'failed' for job in state['jobs']) else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--check', action='store_true')
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.plan.resolve(), arguments.check))
