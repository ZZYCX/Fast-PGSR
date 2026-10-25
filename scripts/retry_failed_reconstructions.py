"""Repair failed scenes in isolation after the main batch and older retry finish.

Never lowers geometry, mask, training, export, or completion checks. Input scenes
are read-only; only verified reconstruction outputs go to the normal output root.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import sqlite3


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(str(temp), str(path))


def blockers(project):
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            args = (p / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            names = {os.path.basename(a) for a in args}
            relevant = {'run_batch_gpu1.sh', 'train.py', 'render.py', 'retry_aruco_queue.py'}
            if names & relevant and os.readlink(str(p / 'cwd')) == str(project):
                found.append(int(p.name))
        except (OSError, ValueError):
            pass
    return found


def discover(batch):
    jobs = {}
    for stage in ('Extract', 'Colmap', 'Mask', 'Align', 'Label', 'Train', 'Export', 'Render'):
        for log in sorted((batch / (stage + '_Error')).glob('*.log')):
            jobs.setdefault(log.stem, {'scene': log.stem, 'failed_stage': stage})
    return sorted(jobs.values(), key=lambda j: (j['failed_stage'] != 'Align', j['scene']))


def discover_plan(plan):
    jobs = discover(Path(plan['batch_log_dir']))
    if plan.get('all_unfinished'):
        known = {j['scene'] for j in jobs}
        project = Path(plan['project_root'])
        names = {p.stem for p in (project / 'videos').iterdir()
                 if p.is_file() and p.suffix.lower() in ('.mp4', '.mov')}
        names.update(p.name for p in (project / 'data').iterdir() if p.is_dir())
        jobs.extend({'scene': n, 'failed_stage': 'New'} for n in sorted(names - known))
    priority = {name: index for index, name in enumerate(plan.get('priority_scenes', []))}
    return sorted(jobs, key=lambda j: priority.get(j['scene'], len(priority)))


def install(project, root, plan):
    if plan.get('sam2_bounded_frames'):
        proof = json.loads((root / 'sam-cache-validation.json').read_text())
        if proof.get('state') != 'passed' or not proof.get('tensor_layout_preserved'):
            raise RuntimeError('SAM2 bounded-cache tensor and layout validation required')
    # Stage everything first, and refuse to overwrite code changed since audit.
    backup = root / 'code-backup'
    for relative, verified in plan.get('staged_code', {}).items():
        if digest(root / 'staged' / relative) != verified:
            raise RuntimeError('Staged code differs from verified code: ' + relative)
    for filename, verified in plan.get('compatibility_files', {}).items():
        if digest(filename) != verified:
            raise RuntimeError('Compatibility runtime changed: ' + filename)
    for relative, expected in plan['expected_code'].items():
        target = project / relative
        staged = root / 'staged' / relative
        current = digest(target) if target.exists() else None
        if current == digest(staged):
            continue
        if current != expected:
            raise RuntimeError('Code changed since audit: ' + relative)
        saved = backup / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not saved.exists():
            shutil.copy2(str(target), str(saved))
    for relative in plan['expected_code']:
        target = project / relative
        temp = target.with_name(target.name + '.failure-repair.tmp')
        shutil.copy2(str(root / 'staged' / relative), str(temp))
        os.replace(str(temp), str(target))


def prepare(source, target, with_masks):
    # Copies the untouched undistorted model, never a previously aligned sparse/0.
    if target.exists():
        raise FileExistsError('Retry directory already exists: ' + str(target))
    (target / 'sparse').mkdir(parents=True)
    for name in ('cameras.bin', 'images.bin', 'points3D.bin'):
        shutil.copy2(str(source / 'sparse' / name), str(target / 'sparse' / name))
    os.symlink(str(source / 'images'), str(target / 'images'))
    if (source / 'auto_rubik_keyframe.json').is_file():
        shutil.copy2(str(source / 'auto_rubik_keyframe.json'), str(target / 'auto_rubik_keyframe.json'))
    if with_masks:
        for name in ('object_mask', 'grounded_sam2_scene'):
            os.symlink(str(source / name), str(target / name))


def fresh_data_root(root, prefix, scene):
    for index in range(1000):
        candidate = root / (prefix if not index else prefix+'-'+str(index))
        if not (candidate / scene).exists():
            return candidate
    raise RuntimeError('Too many prior recovery attempts: ' + scene)


def run_logged(command, project, env, log, timeout_seconds=None):
    if timeout_seconds is not None:
        if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
            raise ValueError('Invalid scene timeout')
        command = ['timeout', '--kill-after=10s', str(timeout_seconds) + 's'] + command
    with log.open('ab', buffering=0) as output:
        return subprocess.run(command, cwd=str(project), env=env,
                              stdin=subprocess.DEVNULL,
                              stdout=output, stderr=subprocess.STDOUT).returncode


def mask_identity(source, trial, report):
    # A new seed may not silently switch to another plant/cube in the same plot.
    import numpy as np
    from PIL import Image
    seed = report['selected_frame']
    final = np.asarray(Image.open(trial / 'object_mask' / (Path(seed).stem + '.png')))
    values = {}
    for name, label in [('single_green_leafy_plant', 1), ('rubik_1', 2)]:
        old = np.asarray(Image.open(source / 'grounded_sam2_scene/initialization' / (name + '.png'))) > 0
        new = final == label
        union = int((old | new).sum())
        values[name] = float((old & new).sum() / max(1, union))
    return all(v >= .5 for v in values.values()), values


def mask_identity_in_env(source, trial, env):
    command = [env['CONDA_BASE'] + '/bin/conda', 'run', '--no-capture-output',
               '-n', 'Fast-Sam2', 'python', str(Path(__file__).resolve()),
               '--mask-identity', str(source), str(trial)]
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    for line in result.stdout.splitlines():
        if result.returncode == 0 and line.startswith('MASK_IDENTITY_RESULT='):
            return json.loads(line.split('=', 1)[1])
    raise RuntimeError('Mask identity check failed in Fast-Sam2: ' + result.stdout[-2000:])


def regenerate_masks(project, root, source, job, env):
    if not (source / 'auto_rubik_keyframe.json').is_file():
        return None
    report = json.loads((source / 'auto_rubik_keyframe.json').read_text(encoding='utf-8'))
    ranking = sorted([r for r in report['ranking'] if r.get('final_score') is not None],
                     key=lambda r: r['final_score'], reverse=True)
    candidates = [None] + [r for r in ranking if r['image_name'] != report['selected_frame']]
    for index, candidate in enumerate(candidates):
        trial = fresh_data_root(root, 'mask-attempt-%s' % index, job['scene']) / job['scene']
        prepare(source, trial, False)
        selection = dict(report)
        selection['scene'] = str(trial)
        if candidate:
            selection.update(selected_frame=candidate['image_name'],
                             selected_objects=[candidate['plant']] + candidate['selected_rubiks'])
        frame = selection['selected_frame']
        selection['selected_frame_sha256'] = digest(trial / 'images' / frame)
        path = trial / 'auto_rubik_keyframe.json'
        save(path, selection)
        log = root / ('mask-%s-attempt-%s.log' % (job['scene'], index))
        code = run_logged([env['CONDA_BASE'] + '/bin/conda', 'run', '--no-capture-output',
                           '-n', 'Fast-Sam2', 'python',
                           str(project / 'scripts/mask_generation/run_grounded_sam2_scene.py'),
                           '--scene', str(trial), '--init-frame', frame,
                           '--selection-json', str(path), '--rubik-count', '1',
                           '--direction', 'both', '--fill-hole-area', '0'], project, env, log)
        attempt = {'frame': frame, 'exit_code': code, 'log': str(log), 'data': str(trial)}
        job.setdefault('mask_attempts', []).append(attempt)
        if code == 0:
            same, overlaps = mask_identity_in_env(source, trial, env)
            attempt['original_seed_iou'] = overlaps
            if same:
                return trial
    return None


def main(plan_file, check=False):
    root = plan_file.parent.resolve()
    plan = json.loads(plan_file.read_text(encoding='utf-8'))
    project = Path(plan['project_root']).resolve()
    if check:
        print(json.dumps({'blockers': blockers(project), 'jobs': discover(Path(plan['batch_log_dir']))}))
        return 0
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(project / 'scripts'))
    from retry_aruco_queue import result_valid
    state_path = root / 'recovery-status.json'
    state = {'pid': os.getpid(), 'state': 'waiting_main_batch',
             'jobs': [dict(j, state='pending') for j in discover_plan(plan)]}
    with (project / '.failed-reconstruction-recovery.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        def wait_idle():
            while True:
                active = blockers(project)
                if not active:
                    # Require a second idle observation to avoid a between-scene gap.
                    time.sleep(5)
                    active = blockers(project)
                if not active:
                    return
                state.update(state='waiting_main_batch', active_pids=active,
                             updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
                save(state_path, state)
                time.sleep(30)

        wait_idle()
        install(project, root, plan)
        state['jobs'] = [dict(j, state='pending') for j in discover_plan(plan)]
        for job in state['jobs']:
            scene = job['scene']
            if result_valid(project, scene):
                job['state'] = 'already_complete'
                save(state_path, state)
                continue
            policy_path = root / 'recovery-policy.json'
            policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
            if scene in policy.get('skip_scenes', {}):
                job.update(state='skipped', reason=policy['skip_scenes'][scene])
                save(state_path, state)
                continue
            # An incomplete previous training output needs individual diagnosis.
            if (project / 'output' / scene).exists() and not plan.get('all_unfinished'):
                job.update(state='needs_review', reason='Existing incomplete output preserved')
                save(state_path, state)
                continue
            wait_idle()
            while shutil.disk_usage(str(project)).free < 15 * 1024**3:
                state.update(state='waiting_storage', required_free_gib=15)
                save(state_path, state)
                time.sleep(30)
            job['state'] = 'running'
            state.update(state='running', active_scene=scene,
                         updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
            save(state_path, state)
            env = os.environ.copy()
            env.update(CONDA_BASE=plan['conda_base'],
                       PATH=plan['conda_base'] + '/bin:' + env['PATH'],
                       ARUCO_RECOVERY='1', MASK_PROMPT_RECOVERY='1', CUDA_VISIBLE_DEVICES='0',
                       RENDER_VOXEL_SIZE='0.02', RUBIK_COUNT='1', EXPORT_MODE='strict',
                       ITERATIONS='30000', FORCE='0', SCENE=scene,
                       PROJECT_ROOT=str(project), VIDEO_DIR=str(project / 'videos'),
                       OUTPUT_ROOT=str(project / 'output'), LOG_ROOT=str(project / 'logs'),
                       RUN_EXTRACT='0', RUN_COLMAP='0', RUN_MASK='0', RUN_ALIGN='1',
                       RUN_LABEL='1', RUN_TRAIN='1', RUN_EXPORT='1', RUN_RENDER='1')
            if plan.get('aruco_compat_path'):
                env['ARUCO_COMPAT_PATH'] = plan['aruco_compat_path']
            if plan.get('resource_safe'):
                env['TRAIN_PRELOAD_IMG'] = '0'
            if plan.get('sam2_bounded_frames'):
                env['SAM2_BOUNDED_FRAMES'] = '1'
            if plan.get('colmap_mapper_executable'):
                env['COLMAP_MAPPER_EXECUTABLE'] = plan['colmap_mapper_executable']
            if plan.get('simple_flow'):
                env.update(ARUCO_RECOVERY='1' if plan.get('aruco_detection_compat') else '0',
                           MASK_PROMPT_RECOVERY='0',
                           ARUCO_NATIVE_FRAMES='0', FPS_TARGET=str(plan.get('fps_target', 2.5)),
                           TRAIN_PRELOAD_IMG='1', SAM2_BOUNDED_FRAMES='0',
                           ALIGN_TIMEOUT_SECONDS=str(plan.get('align_timeout_seconds', 0)))
            source = project / 'data' / scene
            scene_started = time.monotonic()
            try:
                # Keep partial trained products intact. Rebuild in a separate
                # output tree and promote only after every original check passes.
                isolated = (project / 'output' / scene).exists()
                result_project = project
                if isolated:
                    result_project = fresh_data_root(root, 'rebuilt-results', scene) / scene
                    env['OUTPUT_ROOT'] = str(result_project / 'output')
                output = Path(env['OUTPUT_ROOT']) / scene
                prepared = plan.get('prepared_inputs', {}).get(scene)
                native_attempted = scene in plan.get('native_retry_scenes', [])
                if plan.get('simple_flow') and not (plan.get('reuse_preprocessing') and prepared):
                    native_attempted = False
                    trial = fresh_data_root(root, 'standard-data', scene) / scene
                    env.update(RUN_EXTRACT='1', RUN_COLMAP='1', RUN_MASK='1')
                elif native_attempted:
                    while shutil.disk_usage(str(project)).free < 35 * 1024**3:
                        state.update(state='waiting_storage', required_free_gib=35)
                        save(state_path, state)
                        time.sleep(30)
                    state.update(state='running', active_scene=scene)
                    trial = fresh_data_root(root, 'native-data', scene) / scene
                    env.update(ARUCO_NATIVE_FRAMES='1', RUN_EXTRACT='1', RUN_COLMAP='1', RUN_MASK='1')
                    job['native_frame_retry'] = True
                elif prepared:
                    saved_source = Path(prepared['path'])
                    trial = fresh_data_root(root, 'prepared-data', scene) / scene
                    if prepared['stage'] == 'Colmap':
                        trial.mkdir(parents=True)
                        os.symlink(str(saved_source / 'input'), str(trial / 'input'))
                        source_db = sqlite3.connect('file:' + str(saved_source / 'database.db') + '?mode=ro', uri=True)
                        target_db = sqlite3.connect(str(trial / 'database.db'))
                        try:
                            source_db.backup(target_db, pages=1024)
                        finally:
                            target_db.close()
                            source_db.close()
                        env.update(RUN_COLMAP='1', RUN_MASK='1', COLMAP_RESUME_MAPPER='1')
                    else:
                        prepare(saved_source, trial, True)
                        if prepared['stage'] == 'Train':
                            shutil.copytree(str(saved_source / 'sparse/0'), str(trial / 'sparse/0'))
                            shutil.copytree(str(saved_source / 'semantic_initialization'),
                                            str(trial / 'semantic_initialization'))
                            env.update(RUN_ALIGN='0', RUN_LABEL='0')
                        elif prepared['stage'] != 'Align':
                            raise ValueError('Unsupported prepared stage')
                        if plan.get('reuse_preprocessing') and prepared['stage'] == 'Align':
                            env['RUN_MASK'] = '1'  # Verify existing instance guards; do not regenerate.
                    job['verified_prepared_input'] = prepared
                elif job['failed_stage'] not in ('Mask', 'Align') or isolated:
                    trial = fresh_data_root(root, 'new-data', scene) / scene
                    env.update(RUN_EXTRACT='1', RUN_COLMAP='1', RUN_MASK='1')
                elif job['failed_stage'] == 'Mask':
                    trial = regenerate_masks(project, root, source, job, env)
                elif job['failed_stage'] == 'Align':
                    trial = fresh_data_root(root, 'resume-data', scene) / scene
                    prepare(source, trial, True)
                    if plan.get('alignment_cache_root'):
                        from scripts.preprocess.recovery_cache import load, apply
                        cached = load(plan['alignment_cache_root'], scene)
                        if cached and cached['state'] == 'passed':
                            apply(cached, trial)
                            job['verified_cpu_alignment_reused'] = cached['aligned']
                        elif cached and cached.get('reason') in ('insufficient_registered_views', 'geometry_rejected'):
                            job['preflight_requires_more_frames'] = cached['reason']
                            trial = None
                else:
                    raise RuntimeError('New failure stage needs diagnosis: ' + job['failed_stage'])
                log = root / ('retry-' + scene + '.log')
                code = None
                if trial is not None:
                    env['DATA_ROOT'] = str(trial.parent)
                    job['data'] = str(trial)
                    code = run_logged(['bash', 'run_batch_gpu1.sh'], project, env, log,
                                      timeout_seconds=plan.get('scene_timeout_seconds'))
                # Exactly one 3.5-FPS resampling attempt, sharing the same scene
                # deadline. Never retrain after a Train/Export/Render failure.
                if (plan.get('simple_flow') and plan.get('fps_retry', False)
                        and code not in (None, 0, 124, 137) and not output.exists()
                        and float(env.get('FPS_TARGET', 2.5)) < 3.5):
                    remaining = int(plan['scene_timeout_seconds'] - (time.monotonic() - scene_started))
                    if remaining > 60:
                        trial = fresh_data_root(root, 'fps35-data', scene) / scene
                        env.update(DATA_ROOT=str(trial.parent), FPS_TARGET='3.5',
                                   RUN_EXTRACT='1', RUN_COLMAP='1', RUN_MASK='1',
                                   RUN_ALIGN='1', RUN_LABEL='1')
                        job.update(fps_retry=3.5, data=str(trial))
                        code = run_logged(['bash', 'run_batch_gpu1.sh'], project, env, log,
                                          timeout_seconds=remaining)
                # More source frames may resolve blur/occlusion and sparse support.
                # Only retry preprocessing when training has never created output.
                if (code != 0 and not output.exists() and not native_attempted
                        and not plan.get('simple_flow')):
                    env.pop('COLMAP_RESUME_MAPPER', None)
                    dense_root = fresh_data_root(root, 'dense-data', scene)
                    env.update(DATA_ROOT=str(dense_root), FPS_TARGET='10',
                               RUN_EXTRACT='1', RUN_COLMAP='1', RUN_MASK='1', RUN_ALIGN='1', RUN_LABEL='1')
                    retained = plan.get('retained_manifests', {}).get(scene)
                    if retained:
                        env['ARUCO_RETAINED_MANIFEST'] = retained
                    job['dense_retry'] = True
                    job['data'] = str(dense_root / scene)
                    code = run_logged(['bash', 'run_batch_gpu1.sh'], project, env, log)
                # A rare source frame can also fall between 10 FPS samples.
                # Try every native source frame losslessly before declaring that
                # the source cannot satisfy the unchanged geometry/mask checks.
                if (code != 0 and not output.exists() and plan.get('native_frame_fallback')
                        and not native_attempted and not plan.get('simple_flow')):
                    while shutil.disk_usage(str(project)).free < 35 * 1024**3:
                        state.update(state='waiting_storage', required_free_gib=35)
                        save(state_path, state)
                        time.sleep(30)
                    native_root = fresh_data_root(root, 'native-data', scene)
                    state.update(state='running', active_scene=scene)
                    env.update(DATA_ROOT=str(native_root), ARUCO_NATIVE_FRAMES='1',
                               RUN_EXTRACT='1', RUN_COLMAP='1', RUN_MASK='1')
                    job.update(native_frame_retry=True, data=str(native_root / scene))
                    code = run_logged(['bash', 'run_batch_gpu1.sh'], project, env, log)
                valid = code == 0 and result_valid(result_project, scene)
                if valid and isolated:
                    backup = root / 'preserved-incomplete-output' / scene
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    if backup.exists():
                        raise FileExistsError('Preserved output already exists: ' + str(backup))
                    original = project / 'output' / scene
                    os.rename(str(original), str(backup))
                    try:
                        os.rename(str(output), str(original))
                    except Exception:
                        os.rename(str(backup), str(original))
                        raise
                    job['preserved_output'] = str(backup)
                job.update(state='complete' if valid else 'needs_review',
                           exit_code=code, verified_result=valid, log=str(log))
                if not valid and plan.get('simple_flow'):
                    job.update(state='skipped', reason='Standard pipeline failed; no recovery retries')
            except Exception as error:
                job.update(state='skipped' if plan.get('simple_flow') else 'needs_review',
                           reason=type(error).__name__ + ': ' + str(error))
            state['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            save(state_path, state)
        pending = any(j['state'] == 'needs_review' for j in state['jobs'])
        state.update(state='finished_with_pending' if pending else 'finished', active_scene=None,
                     updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        save(state_path, state)
        return int(any(j['state'] == 'needs_review' for j in state['jobs']))


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--mask-identity':
        source, trial = Path(sys.argv[2]), Path(sys.argv[3])
        report = json.loads((source / 'auto_rubik_keyframe.json').read_text(encoding='utf-8'))
        print('MASK_IDENTITY_RESULT=' + json.dumps(mask_identity(source, trial, report)))
        raise SystemExit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        raise SystemExit(main(args.plan.resolve(), args.check))
    except Exception as error:
        status_path = args.plan.resolve().parent / 'recovery-status.json'
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        previous.update(state='error', error=type(error).__name__ + ': ' + str(error),
                        updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        save(status_path, previous)
        raise
