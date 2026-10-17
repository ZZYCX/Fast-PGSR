"""CPU regression tests; run with Python + numpy, Pillow, cv2, plyfile.
Model adapters are fakes; production selection/scoring/propagation orchestration runs.
"""
import argparse
import ast
from contextlib import nullcontext, redirect_stdout
import csv
import gc
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from utils.semantic_io import _xyz_checksum, _semantic_checksum
import export_target_gaussians as exporter
import validate_semantic_inputs as validator
from plyfile import PlyData, PlyElement


def functions(path, extra):
    """Load actual function bodies without importing the CUDA/model packages."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    defs = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    defs += [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    namespace = dict(np=np, cv2=cv2, Path=Path, json=json, hashlib=hashlib, argparse=argparse,
                     math=math, re=re, sys=sys, gc=gc, csv=csv, io=io, shutil=shutil,
                     subprocess=subprocess, nullcontext=nullcontext, redirect_stdout=redirect_stdout)
    namespace.update(extra)
    exec(compile(ast.fix_missing_locations(ast.Module(body=defs, type_ignores=[])), str(path), 'exec'), namespace)
    return namespace


class Progress:
    def __init__(self, items=None, **kw): self.items = items
    def __iter__(self): return iter(self.items)
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def update(self, *args): pass
    @staticmethod
    def write(*args): pass


class Tensor:
    def __init__(self, data): self.data = np.asarray(data)
    def __getitem__(self, key): return Tensor(self.data[key])
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.data


class Predictor:
    def set_image(self, image): self.shape = image.shape[:2]
    def predict(self, box, **kw):
        masks = np.zeros((len(box), *self.shape), dtype=bool)
        for i, bounds in enumerate(box):
            x1,y1,x2,y2 = np.asarray(bounds).astype(int)
            masks[i,y1:y2,x1:x2] = True
        return masks, np.full(len(box), .9), None


class Video:
    def __init__(self): self.registered = {}; self.fill_hole_area = 0
    def init_state(self, **kw): return {}
    def reset_state(self, state): pass
    def add_new_mask(self, state, idx, obj, mask): self.registered[obj] = mask
    def propagate_in_video(self, state, start_frame_idx, reverse):
        indices = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, 3)
        for i in indices:
            logits = np.stack([np.where(m, 2., -2.) for m in self.registered.values()])[:,None]
            yield i, list(self.registered), Tensor(logits)


class Regression(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.scene = Path(self.tmp.name) / 'soybean with spaces'
        (self.scene/'images').mkdir(parents=True)
        for i in range(3):
            Image.new('RGB',(40,30),(0,160,0)).save(self.scene/'images'/('frame_%06d.jpg'%i))
        self.video = Video()
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None),
            inference_mode=nullcontext, autocast=lambda *a,**kw: nullcontext(), bfloat16=None,
            tensor=lambda data,**kw: np.asarray(data),float32=np.float32)
        common = dict(torch=fake_torch,tqdm=Progress,PROJECT_ROOT=ROOT,
                      GSAM2_ROOT=self.scene,DEFAULT_PROMPT='test',PLANT_KEY='plant',RUBIK_KEY='rubik',
                      OBJECT_NAMES={1:'plant',2:'rubik_1',3:'rubik_2'},
                      PALETTE={0:(0,0,0),1:(0,200,0),2:(255,0,0),3:(0,0,255)},SCRIPT_VERSION='test',
                      SUFFIXES={'.jpg'},BASE_SCRIPT=ROOT/'scripts/mask_generation/run_grounded_sam2_scene.py')
        self.base = functions(ROOT/'scripts/mask_generation/run_grounded_sam2_scene.py', common)
        self.auto = functions(ROOT/'scripts/mask_generation/run_rubik_scene_auto.py', dict(common,LOCAL_DEFAULT_PROMPT='test'))
        self.detections = [self.det('plant',[1,1,12,25]), self.det('rubik',[15,4,23,12]), self.det('rubik',[26,4,34,12])]
        for i,d in enumerate(self.detections): d['order'] = i
        self.base.update(verify_bert=lambda *a: None,load_grounding_model_offline=lambda *a: object(),
                         build_sam2=lambda *a,**kw: object(),SAM2ImagePredictor=lambda *a: Predictor(),
                         build_sam2_video_predictor=lambda **kw: self.video,
                         load_image=lambda path:(cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB),None),
                         run_grounding=lambda *a: self.detections,
                         nms_preserve_order=lambda detections,threshold: detections)
        self.auto['box_iou'] = lambda a,b: np.zeros((len(a),len(b)))
        self.auto['load_base'] = lambda: types.SimpleNamespace(**self.base)
    def tearDown(self): self.tmp.cleanup()
    @staticmethod
    def det(key, box): return dict(label_key=key,label=key,score=.9,box=np.array(box,dtype=np.float32),order=0)

    def test_cli_defaults_and_missing_cube(self):
        for ns in (self.auto,self.base):
            argv=['--scene',str(self.scene)] + (['--init-frame','frame_000001.jpg'] if ns is self.base else [])
            self.assertEqual(ns['parser']().parse_args(argv).rubik_count,1)
        with self.assertRaises(RuntimeError): self.base['choose_targets'](self.detections[:1],.7)
        with self.assertRaises(ValueError): self.base['active_ids'](0)
        with self.assertRaises(ValueError): self.base['choose_targets'](self.detections,.7,1,8)

    def test_scoring_and_target_limits(self):
        for count in (1,2):
            plant,selected,_,_,_=self.base['choose_targets'](self.detections,.7,count)
            self.assertEqual(len(selected),count)
            self.assertEqual(selected[0]['order'],1)
            score=self.auto['fast_score'](plant,self.detections[1:],selected,np.zeros((30,40,3),np.uint8))
            masks=np.zeros((count+1,30,40),bool);masks[:,2:12,2:12]=True
            _,details=self.auto['refine_score'](score,np.zeros((30,40,3),np.uint8),masks,[.9]*(count+1))
            self.assertEqual(details['rubik_2_area_ratio'] is None,count==1)
            self.assertEqual(details['rubik_mask_iou'] is None,count==1)

    def run_mask_flow(self,count,full_res):
        commands=[]
        def child(command,check=False):
            commands.append(command)
            # Only dependency/model loading is mocked; actual base.main runs.
            with patch.object(sys,'argv',command[1:]), patch('scripts.mask_generation.instance_mask_guard.refine_training_masks',return_value={'version':'sfm_sam_instance_v1','implementation_sha256':validator.digest(ROOT/'scripts/mask_generation/instance_mask_guard.py'),'frames':[]}):
                self.base['main']()
            return types.SimpleNamespace(returncode=0)
        args=['auto','--scene',str(self.scene),'--rubik-count',str(count)]
        if full_res: args += ['--full-res']
        with patch.object(sys,'argv',args),patch.object(subprocess,'run',child),redirect_stdout(io.StringIO()):
            self.assertEqual(self.auto['main'](),0)
        self.assertEqual(set(self.video.registered),set(range(1,count+2)))
        self.assertIn('--selection-json',commands[0])
        self.assertEqual(commands[0][commands[0].index('--rubik-count')+1],str(count))
        summary=json.loads((self.scene/'grounded_sam2_scene/summary.json').read_text())
        self.assertEqual(summary['rubik_count'],count)
        self.assertEqual(set(summary['object_ids']),set(map(str,range(1,count+2))))
        for path in (self.scene/'object_mask').glob('*.png'):
            values=set(np.unique(np.asarray(Image.open(path))))
            self.assertEqual(values,set(range(count+2)))
        report=(self.scene/'grounded_sam2_scene/tracking_report.csv').read_text(encoding='utf-8-sig')
        self.assertEqual('rubik_2_missing' in report,False)
        validator.validate_masks(self.scene,count)
        # A frame can lose all objects without complaining about inactive cube2.
        _,flags=self.base['tracking_quality'](np.zeros((5,5)),None,self.base['active_ids'](count))
        self.assertEqual('rubik_2_missing' in flags,count==2)
        # Propagation boxes are exactly those scored, not a second detection result.
        auto_report=json.loads((self.scene/'auto_rubik_keyframe.json').read_text())
        self.assertEqual(len(auto_report['selected_objects']),count+1)

    def test_single_mask_full_flow(self): self.run_mask_flow(1,False)
    def test_single_full_resolution_flow(self): self.run_mask_flow(1,True)
    def test_two_cube_regression_flow(self): self.run_mask_flow(2,False)

    def test_selection_report_rejects_stale_inputs(self):
        self.run_mask_flow(1,False)
        report=self.scene/'auto_rubik_keyframe.json'
        saved=json.loads(report.read_text());frame=saved['selected_frame']
        with self.assertRaises(ValueError): self.base['read_selection'](report,self.scene,frame,2,(40,30))
        Image.new('RGB',(40,30),(200,0,0)).save(self.scene/'images'/frame)
        with self.assertRaises(ValueError): self.base['read_selection'](report,self.scene,frame,1,(40,30))
        with self.assertRaises(ValueError): validator.validate_masks(self.scene,1)

    def test_initialization_binding_and_changed_coordinates(self):
        self.run_mask_flow(1,False)
        sparse=self.scene/'sparse/0'; sparse.mkdir(parents=True)
        for f in ('cameras.bin','images.bin','points3D.bin'): (sparse/f).write_bytes(b'fake colmap')
        folder=self.scene/'semantic_initialization';folder.mkdir()
        np.savez(folder/'point_semantics.npz',object_id=np.array([0,1,2,-1]),
                 point3D_ids=np.arange(4),object_probs=np.eye(4)[[0,1,2,0]])
        signature=validator.initialization_signature(self.scene,1)
        validator.binding_path(self.scene).write_text(json.dumps(signature))
        self.assertEqual(validator.check_binding(self.scene,1),signature)
        (sparse/'points3D.bin').write_bytes(b'changed coordinates')
        with self.assertRaises(ValueError): validator.check_binding(self.scene,1)

    def test_strict_export_preserves_every_attribute(self):
        for count in (1,2):
            folder=Path(self.tmp.name)/('iteration'+str(count));folder.mkdir()
            # Full Gaussian layout, including SH rest; all 62 original fields.
            fields=['x','y','z','nx','ny','nz']+['f_dc_'+str(i) for i in range(3)]+['f_rest_'+str(i) for i in range(45)]+['opacity']+['scale_'+str(i) for i in range(3)]+['rot_'+str(i) for i in range(4)]
            vertices=np.zeros(5,dtype=[(f,'f4') for f in fields])
            for i,f in enumerate(fields): vertices[f]=np.arange(5)+i*.01
            PlyData([PlyElement.describe(vertices,'vertex')]).write(str(folder/'point_cloud.ply'))
            sem=dict(object_id=np.array([0,1,2,-1,3 if count==2 else 1],np.int64),
                     object_prob=np.eye(4,dtype=np.float32)[[0,1,2,0,3 if count==2 else 1]],
                     semantic_confidence=np.ones(5,np.float32),source_point3D_id=np.arange(5,dtype=np.int64),
                     semantic_evidence_sum=np.zeros((5,4),np.float32),semantic_obs_count=np.ones(5,np.int64),
                     semantic_stable_count=np.zeros(5,np.int64),semantic_candidate_id=np.full(5,-1,np.int64),
                     semantic_candidate_stable_count=np.zeros(5,np.int64))
            xyz=np.stack([vertices[f] for f in ('x','y','z')],1)
            checksum=_semantic_checksum(*[sem[k] for k in exporter.SEMANTIC_KEYS])
            np.savez(folder/'semantic_state.npz',**sem,semantic_state_version=2,semantics_attached=True,
                     gaussian_count=5,xyz_order_checksum=_xyz_checksum(xyz),semantic_order_checksum=checksum)
            before=hashlib.sha256((folder/'point_cloud.ply').read_bytes()).hexdigest()
            output=folder/'export'
            with redirect_stdout(io.StringIO()): exporter.export_target_gaussians(str(folder),str(output),mode='strict',export_per_class=True,rubik_count=count)
            subset=PlyData.read(str(output/'strict/point_cloud.ply'))['vertex'].data
            wanted=vertices[[1,2,4]]
            self.assertEqual(subset.dtype,wanted.dtype)
            np.testing.assert_array_equal(subset,wanted)
            self.assertEqual(before,hashlib.sha256((folder/'point_cloud.ply').read_bytes()).hexdigest())
            exporter.validate_input(str(output/'strict/point_cloud.ply'),str(output/'strict/semantic_state.npz'))
            if count==1: self.assertFalse((output/'rubik_2_only').exists())
            else:
                with self.assertRaises(RuntimeError): exporter.export_target_gaussians(str(folder),str(folder/'bad'),mode='strict',rubik_count=1)

    def test_shared_checksums_match_original_training_algorithm(self):
        # Independent calculation matching the original persisted byte protocol.
        arrays=[np.arange(12,dtype=np.float32).reshape(4,3),np.array([1,2],np.int64)]
        self.assertEqual(_xyz_checksum(arrays[0]),hashlib.sha256(arrays[0].tobytes()).hexdigest())
        parts=[np.arange(4,dtype=np.int64) for _ in range(9)]
        self.assertEqual(_semantic_checksum(*parts),hashlib.sha256(b''.join(a.tobytes() for a in parts)).hexdigest())

    def test_colmap_argv_and_output_protection(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('converter',ROOT/'scripts/preprocess/convert.py')
        converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
        scene=Path(self.tmp.name)/'colmap scene with spaces';(scene/'input').mkdir(parents=True)
        (scene/'input/frame.jpg').write_bytes(b'frame')
        args=argparse.Namespace(overwrite=False,no_gpu=False,gpu_index=0,colmap_executable='colmap',camera='OPENCV',resize=False)
        calls=[]
        def run(argv,check):
            calls.append(argv)
            if argv[1]=='mapper':
                (scene/'sparse/0').mkdir()
                for f in ('cameras.bin','images.bin','points3D.bin'): (scene/'sparse/0'/f).write_bytes(b'model')
            if argv[1]=='image_undistorter': (scene/'images').mkdir()
        with patch.object(subprocess,'run',run): converter.prepare_scene(scene,args)
        self.assertEqual(calls[0][calls[0].index('--image_path')+1],str(scene.resolve()/'input'))
        self.assertEqual(calls[0][calls[0].index('--SiftExtraction.gpu_index')+1],'0')
        with self.assertRaises(FileExistsError): converter.prepare_scene(scene,args)


class BatchRegression(unittest.TestCase):
    def run_batch(self, fail_mask=False, existing_output=False, label_fixture=False, multi_scene=False, videos=False):
        bash = os.environ.get('BASH_TEST_EXECUTABLE') or shutil.which('bash')
        if not bash:
            self.skipTest('Bash not available')
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);bin_dir=folder/'bin';bin_dir.mkdir()
            data=folder/'data';(data/'soybean').mkdir(parents=True)
            if multi_scene:
                (data/'soybean2').mkdir(parents=True)
            video_dir=folder/'videos';video_dir.mkdir()
            if videos:
                (video_dir/'soybean.mp4').touch();(video_dir/'soybean2.mp4').touch()
            output=folder/'output'
            if existing_output:
                (output/'soybean').mkdir(parents=True)
                (output/'soybean/keep.txt').write_text('original result')
            if label_fixture:
                (data/'soybean/sparse/0').mkdir(parents=True)
                for f in ('cameras.bin','images.bin','points3D.bin'):
                    (data/'soybean/sparse'/f).write_text('registered model')
            calls=folder/'calls.log'
            conda=bin_dir/'conda'
            conda.write_text("#!/bin/bash\nprintf 'GPU=%s ' \"$CUDA_VISIBLE_DEVICES\" >> \"$CALLS\"\nprintf '%s ' \"$@\" >> \"$CALLS\"\nprintf '\\n' >> \"$CALLS\"\nif [[ \"$*\" == *run_rubik_scene_auto.py* && \"${FAIL_MASK:-0}\" == 1 ]]; then exit 9; fi\nexit 0\n",encoding='utf-8')
            if label_fixture:
                conda.write_text(conda.read_text().replace('exit 0',
                    'if [[ "$*" == *align_aruco_refine.py* ]]; then mkdir -p "$DATA_ROOT/soybean/sparse/aligned"; cp "$DATA_ROOT/soybean/sparse/"{cameras,images,points3D}.bin "$DATA_ROOT/soybean/sparse/aligned/"; fi\nexit 0'))
            if multi_scene:
                conda.write_text(conda.read_text().replace('exit 0',
                    'if [[ "$*" == *convert.py* ]]; then while (($#)); do if [[ "$1" == --scene_path ]]; then shift; scene="$1"; fi; shift; done; mkdir -p "$scene/sparse"; for f in cameras.bin images.bin points3D.bin; do echo model > "$scene/sparse/$f"; done; fi\nexit 0'))
            if videos:
                conda.write_text(conda.read_text().replace('exit 0',
                    'if [[ "$*" == *extract_frames.py* ]]; then while (($#)); do if [[ "$1" == --output_dir ]]; then shift; frames="$1"; fi; shift; done; touch "$frames/frame_000001.jpg"; fi\nexit 0'))
                ffmpeg=bin_dir/'ffmpeg'
                ffmpeg.write_text('#!/bin/bash\nprintf "Extract %s\\n" "$*" >> "$CALLS"\nfor arg in "$@"; do if [[ "$arg" == *frame_%06d.jpg ]]; then touch "${arg/frame_%06d.jpg/frame_000001.jpg}"; fi; done\n',encoding='utf-8')
                ffmpeg.chmod(0o755)
            conda.chmod(0o755)
            env=os.environ.copy()
            for key in ('CUDA_VISIBLE_DEVICES','GPU_ID','RUBIK_COUNT','FORCE','SCENE'):
                env.pop(key,None)
            env.update(PROJECT_ROOT=ROOT.as_posix(),DATA_ROOT=data.as_posix(),OUTPUT_ROOT=output.as_posix(),
                       LOG_ROOT=(folder/'logs').as_posix(),CALLS=calls.as_posix(),FAKE_BIN=bin_dir.as_posix(),
                       ENTRY=(ROOT/'run_batch_gpu1.sh').as_posix(),RUN_EXTRACT='0',RUN_COLMAP='0',
                       RUN_MASK='1',RUN_ALIGN='0',RUN_LABEL='0',RUN_TRAIN='1',RUN_RENDER='0',RUN_EXPORT='1',
                       FAIL_MASK='1' if fail_mask else '0')
            if multi_scene:
                env.update(RUN_COLMAP='1')
            if videos:
                env.update(RUN_EXTRACT='1',VIDEO_DIR=video_dir.as_posix())
            if label_fixture:
                env.update(RUN_MASK='0',RUN_ALIGN='1',RUN_LABEL='1',RUN_TRAIN='0',RUN_EXPORT='0')
            # Git Bash converts the Windows temporary path; Linux uses it directly.
            command='export PATH="$(cygpath -u "$FAKE_BIN"):$PATH"; exec bash "$ENTRY"' if os.name=='nt' else 'export PATH="$FAKE_BIN:$PATH"; exec bash "$ENTRY"'
            result=subprocess.run([bash,'-c',command],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            log=calls.read_text(encoding='utf-8') if calls.exists() else ''
            if existing_output:
                self.assertEqual((output/'soybean/keep.txt').read_text(),'original result')
            if label_fixture:
                for f in ('cameras.bin','images.bin','points3D.bin'):
                    self.assertEqual((data/'soybean/sparse/0'/f).read_text(),'registered model')
            return result,log

    def test_scene_finishes_before_next_video_is_extracted(self):
        result,log=self.run_batch(multi_scene=True,videos=True)
        self.assertEqual(result.returncode,0,result.stdout)
        lines=log.splitlines()
        second_extract=next(i for i,line in enumerate(lines) if '/extract_frames.py ' in line and 'soybean2.mp4' in line)
        first_export=next(i for i,line in enumerate(lines) if '/export_target_gaussians.py ' in line and '/soybean/' in line)
        second_colmap=next(i for i,line in enumerate(lines) if '/convert.py ' in line and '/soybean2 ' in line)
        self.assertLess(first_export,second_extract)
        self.assertLess(second_extract,second_colmap)

    def test_existing_scenes_are_processed_serially(self):
        result,log=self.run_batch(multi_scene=True)
        self.assertEqual(result.returncode,0,result.stdout)
        lines=log.splitlines()
        first_export=next(i for i,line in enumerate(lines) if '/export_target_gaussians.py ' in line and '/soybean/' in line)
        second_colmap=next(i for i,line in enumerate(lines) if '/convert.py ' in line and '/soybean2 ' in line)
        self.assertLess(first_export,second_colmap)

    def test_empty_sparse_zero_receives_aligned_model(self):
        result,log=self.run_batch(label_fixture=True)
        self.assertEqual(result.returncode,0,result.stdout)
        self.assertIn('label_colmap_points_viewer.py',log)

    def test_upstream_failure_blocks_training_and_export(self):
        result,log=self.run_batch(fail_mask=True)
        self.assertNotEqual(result.returncode,0,result.stdout)
        self.assertIn('--rubik-count 1',log)
        self.assertNotIn('/train.py ',log)
        self.assertNotIn('/export_target_gaussians.py ',log)

    def test_default_gpu_and_strict_parameters(self):
        result,log=self.run_batch()
        self.assertEqual(result.returncode,0,result.stdout)
        self.assertIn('GPU=0 ',log)
        self.assertIn('--semantic_pipeline',log)
        self.assertIn('--mode strict --rubik-count 1',log)

    def test_existing_results_are_not_deleted(self):
        result,log=self.run_batch(existing_output=True)
        self.assertEqual(result.returncode,0,result.stdout)
        self.assertNotIn('/train.py ',log)


if __name__ == '__main__': unittest.main(verbosity=2)
