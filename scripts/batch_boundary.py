"""Transfer an existing batch only at its verified result-publication boundary."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def identity(pid):
    try:
        fields = Path('/proc/{}/stat'.format(pid)).read_text().rsplit(') ', 1)[1].split()
        return {'state': fields[0], 'parent': int(fields[1]), 'start': fields[19]}
    except (OSError, ValueError):
        return None


def same(record):
    info = identity(record['pid'])
    return info is not None and info['state'] != 'Z' and info['start'] == record['start']


def save(path, value):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(str(temp), str(path))


def checkpoint(project, scene):
    request = Path(project) / '.batch-cutover.json'
    if not request.exists():
        return
    plan = json.loads(request.read_text())
    target = plan['foreground']
    pid = os.getpid()
    ancestors = set()
    while pid > 1:
        ancestors.add(pid)
        info = identity(pid)
        if info is None:
            break
        pid = info['parent']
    if target['pid'] not in ancestors or not same(target):
        return
    # The publisher has already committed its complete SHA256 manifest. Hold
    # only its waiting parent batch; the completed publisher exits normally.
    os.kill(target['pid'], signal.SIGSTOP)
    try:
        save(Path(plan['boundary']), {'scene': scene, 'foreground': target,
                                    'published_at': time.time()})
    except Exception:
        os.kill(target['pid'], signal.SIGCONT)
        raise


def descendants(pid):
    parents = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            info = identity(int(entry.name))
            if info and info['state'] != 'Z':
                parents[int(entry.name)] = info['parent']
    children, pending = set(), [pid]
    while pending:
        parent = pending.pop()
        for child, value in parents.items():
            if value == parent and child not in children:
                children.add(child)
                pending.append(child)
    return children


def supervise(plan_path):
    plan = json.loads(plan_path.read_text())
    status = plan_path.parent / 'cutover-status.json'
    foreground = plan['foreground']
    boundary = Path(plan['boundary'])
    save(status, {'state': 'armed_waiting_current_plant', 'foreground': foreground})
    if plan.get('drain_existing_child') and same(foreground):
        # Suspend only the scheduling parent, never its already-running scene
        # subprocess or process group. The full scene pipeline drains normally,
        # including error exits; the replacement retries unfinished results.
        os.kill(foreground['pid'], signal.SIGSTOP)
        save(boundary, {'scene': plan['active_scene'], 'foreground': foreground,
                        'mode': 'existing_scene_pipeline_drains', 'held_at': time.time()})
    while same(foreground) and not boundary.exists():
        time.sleep(2)
    if boundary.exists():
        completed = json.loads(boundary.read_text())
        if completed['foreground'] != foreground:
            raise RuntimeError('Boundary belongs to another batch')
        save(status, dict(completed, state='draining_completed_publisher'))
        while same(foreground) and descendants(foreground['pid']):
            time.sleep(1)
        if same(foreground) and identity(foreground['pid'])['state'] not in ('T', 't'):
            raise RuntimeError('Foreground was not stopped at publication boundary')
    for command in plan.get('validation_commands', []):
        save(status, {'state': 'isolated_validation_at_completed_boundary', 'command': command})
        subprocess.run(command, cwd=plan['project_root'], check=True)
    # Retire only waiting coordinators whose exact process identities and saved
    # states were recorded. No training subprocess may be attached to them.
    for retired in plan['retire']:
        if not same(retired):
            continue
        os.kill(retired['pid'], signal.SIGSTOP)
        state = json.loads(Path(retired['status']).read_text())
        if state.get('state') != 'waiting_main_batch' or descendants(retired['pid']):
            os.kill(retired['pid'], signal.SIGCONT)
            raise RuntimeError('Previous recovery coordinator is active')
        os.kill(retired['pid'], signal.SIGTERM)
        os.kill(retired['pid'], signal.SIGCONT)
        state.update(state='superseded_by_all_unfinished', replacement=str(plan_path))
        save(Path(retired['status']), state)
    if same(foreground):
        os.kill(foreground['pid'], signal.SIGTERM)
        os.kill(foreground['pid'], signal.SIGCONT)
    while same(foreground) or any(same(p) for p in plan['retire']):
        time.sleep(1)
    save(status, {'state': 'latest_batch_coordinator_started',
                  'boundary': json.loads(boundary.read_text()) if boundary.exists() else None,
                  'started_at': time.time()})
    code = subprocess.call([sys.executable, '-u', str(plan_path.parent / 'recovery.py'),
                            '--plan', str(plan_path.parent / 'recovery-plan.json')],
                           cwd=plan['project_root'])
    save(status, {'state': 'coordinator_exited', 'exit_code': code, 'ended_at': time.time()})
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    try:
        code = supervise(args.plan.resolve())
    except Exception as error:
        plan = json.loads(args.plan.read_text())
        save(args.plan.parent / 'cutover-status.json',
             {'state': 'error', 'error': type(error).__name__ + ': ' + str(error)})
        # A failed handoff must leave the original batch able to continue.
        request = Path(plan['project_root']) / '.batch-cutover.json'
        if request.exists() and json.loads(request.read_text()).get('foreground') == plan['foreground']:
            request.unlink()
        if same(plan['foreground']) and plan.get('resume_original_on_error', True):
            os.kill(plan['foreground']['pid'], signal.SIGCONT)
        raise
    sys.exit(code)
