"""CPU-only alignment recovery while the foreground batch retains the GPU.

Publishes input-bound caches, never edits original scenes or completion markers.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(str(temp), str(path))


def run(plan_path):
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    project = Path(plan['project_root']); root = Path(plan['alignment_cache_root'])
    root.mkdir(parents=True, exist_ok=True); (root/'index').mkdir(exist_ok=True)
    sys.path.insert(0, str(project))
    from scripts.preprocess.recovery_cache import bind, binding_current, load, sha, MODEL
    os.nice(19)
    state = {'pid': os.getpid(), 'state': 'running', 'jobs': []}
    completed = set()
    old_path = plan_path.parent/'align-status.json'
    old_passes = {r['scene'] for r in json.loads(old_path.read_text()) if r['state']=='passed'} if old_path.exists() else set()
    with (root/'preflight.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            scenes = [p.stem for p in (Path(plan['batch_log_dir'])/'Align_Error').glob('*.log')]
            scenes.sort(key=lambda name:(name not in old_passes, name))
            pending = [s for s in scenes if s not in completed]
            if not pending:
                main_pid = plan.get('foreground_batch_pid')
                if not main_pid or not Path('/proc', str(main_pid)).exists():
                    break
                state.update(state='watching_new_align_failures', updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
                save(root/'status.json', state); time.sleep(30); continue
            scene = pending[0]; completed.add(scene)
            if (project/'output'/scene/'.reconstruction_complete.json').is_file():
                continue
            cached = load(root, scene)
            if cached:
                state['jobs'].append({'scene':scene, 'state':cached['state'], 'reused_existing_cache':True})
                save(root/'status.json', state); continue
            source = project/'data'/scene
            trial = root/'trials'/(scene+'-'+uuid.uuid4().hex[:8])/scene
            row = {'scene':scene, 'state':'running', 'trial':str(trial)}
            state['jobs'].append(row);state.update(state='running', active_scene=scene)
            save(root/'status.json', state)
            try:
                inputs = bind(source, project, plan['aruco_compat_path'])
                (trial/'sparse').mkdir(parents=True)
                for name in MODEL:shutil.copy2(str(source/'sparse'/name), str(trial/'sparse'/name))
                for name in ['images','object_mask','grounded_sam2_scene']:
                    os.symlink(str(source/name), str(trial/name))
                if (source/'auto_rubik_keyframe.json').is_file():
                    shutil.copy2(str(source/'auto_rubik_keyframe.json'), str(trial/'auto_rubik_keyframe.json'))
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           PYTHONPATH=str(project), ARUCO_RECOVERY='1', ARUCO_COMPAT_PATH=plan['aruco_compat_path'])
                log = trial.parent/'align.log'
                bootstrap = "import cv2,runpy,sys;cv2.setNumThreads(1);sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
                with log.open('wb') as output:
                    code = subprocess.run([sys.executable, '-u', '-c', bootstrap,
                                           str(project/'align_aruco_refine.py'), str(trial)],
                                          env=env, stdout=output, stderr=subprocess.STDOUT, timeout=1800).returncode
                unchanged = binding_current(inputs)
                aligned = trial/'sparse/aligned'
                passed = code==0 and unchanged and all((aligned/n).is_file() and (aligned/n).stat().st_size for n in MODEL)
                text = log.read_text(errors='replace')
                reason = 'insufficient_registered_views' if '至少需要 2 个注册视角' in text else (
                    'geometry_rejected' if '没有满足3D正方形约束的可用 Marker' in text else 'processing_error')
                record = {'scene':scene, 'state':'passed' if passed else 'failed', 'inputs':inputs,
                          'reason':None if passed else reason, 'aligned':str(aligned),
                          'aligned_sha256':{n:sha(aligned/n) for n in MODEL} if passed else {},
                          'source_unchanged':unchanged,'exit_code':code,'log':str(log),
                          'finished_at':time.strftime('%Y-%m-%d %H:%M:%S')}
                if unchanged:save(root/'index'/(scene+'.json'), record)
                row.update(state=record['state'],reason=record['reason'],source_unchanged=unchanged,log=str(log))
            except Exception as error:
                row.update(state='processing_error',error=repr(error))
            state['updated_at']=time.strftime('%Y-%m-%d %H:%M:%S')
            save(root/'status.json',state); print(json.dumps(row),flush=True)
        state.update(state='finished',active_scene=None,updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        save(root/'status.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    run(parser.parse_args().plan.resolve())
