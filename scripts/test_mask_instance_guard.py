"""CPU checks for source-mask instance constraints (Python 3.10 environment)."""
import sys,tempfile,json,hashlib,io
from contextlib import redirect_stdout
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from scripts.mask_generation.instance_mask_guard import dense_components,sample_prompts,refine_training_masks
from scripts.validate_semantic_inputs import validate_masks,LEGACY_STRICT_GUARD_SHA256

rng=np.random.RandomState(7)
a=rng.normal(0,.02,(30,3));b=rng.normal(0,.02,(15,3))+[3,0,0]
xyz=np.vstack((a,b,[[20,20,20]]))
labels=dense_components(xyz,.2)
assert len(set(labels[:30]))==1 and len(set(labels[30:45]))==1
assert labels[0]!=labels[30] and labels[-1]==-1
scaled=dense_components(xyz*5+17,1.)
assert np.array_equal(labels,scaled)
xy=rng.uniform(0,100,(100,2))
assert np.array_equal(sample_prompts(xy,6),sample_prompts(xy,6))
assert len(sample_prompts(xy,6))==6
with tempfile.TemporaryDirectory() as d:
 p=Path(d)
 for folder in ['images','object_mask','grounded_sam2_scene']:(p/folder).mkdir()
 image=p/'images/a.jpg';cv2.imwrite(str(image),np.zeros((20,20,3),np.uint8))
 mask=p/'object_mask/a.png';m=np.zeros((20,20),np.uint8);m[2:10,2:10]=1;m[12:16,12:16]=2
 cv2.imwrite(str(mask),m)
 sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
 summary={'rubik_count':1,'object_ids':{'1':'plant','2':'rubik_1'},'image_sha256':{'a.jpg':sha(image)}}
 report=p/'grounded_sam2_scene/summary.json';report.write_text(json.dumps(summary))
 try:validate_masks(p,1,True)
 except ValueError as e:assert '实例约束' in str(e)
 else:raise AssertionError('legacy masks accepted')
 summary.update(instance_guard={'version':'sfm_sam_instance_v1','implementation_sha256':sha(Path(__file__).parent/'mask_generation/instance_mask_guard.py')},mask_sha256={'a':sha(mask)})
 report.write_text(json.dumps(summary));validate_masks(p,1,True)
 m[0,0]=1;cv2.imwrite(str(mask),m)
 try:validate_masks(p,1,True)
 except ValueError as e:assert '已改变' in str(e)
 else:raise AssertionError('modified mask accepted')
 summary['mask_sha256']['a']=sha(mask)
 summary['instance_guard']['implementation_sha256']=LEGACY_STRICT_GUARD_SHA256
 report.write_text(json.dumps(summary));validate_masks(p,1,True)
 summary['instance_guard']['implementation_sha256']='0'*64
 report.write_text(json.dumps(summary))
 try:validate_masks(p,1,True)
 except ValueError as e:assert '实现已更新' in str(e)
 else:raise AssertionError('unknown guard implementation accepted')

class GuardPredictor:
 def __init__(self):self.calls=0
 def set_image(self,image):self.shape=image.shape[:2]
 def predict(self,**kwargs):
  self.calls+=1
  mask=np.ones((1,*self.shape),bool)
  return mask,np.ones(1),np.ones((1,256,256),np.float32)

# Boundaries 0/1/5 must keep complete propagated plant supervision and preserve
# same-frame cube constraints; 6 must still take the existing refinement path.
with tempfile.TemporaryDirectory() as d:
 p=Path(d);image=p/'frame.jpg'
 cv2.imwrite(str(image),np.zeros((40,40,3),np.uint8))
 previous=np.zeros((40,40),np.uint8)
 previous[4:26,4:26]=1
 previous[31:35,31:35]=2
 previous[1:3,31:33]=2
 xy=np.array([[5,5],[10,5],[20,5],[5,15],[10,20],[20,20]],np.int32)
 for n in (0,1,5,6):
  frames={0:previous.copy()};predictor=GuardPredictor()
  support=({0:(np.arange(n),xy[:n],np.array([30,30,36,36]))},
           np.ones(6,bool),np.zeros(6,bool),{})
  with patch('scripts.mask_generation.instance_mask_guard.build_support',return_value=support),redirect_stdout(io.StringIO()):
   result=refine_training_masks(p,[image],frames,0,predictor)
  assert frames[0].shape==previous.shape and frames[0].dtype==previous.dtype
  assert set(np.unique(frames[0]))<={0,1,2}
  assert np.all(frames[0][31:35,31:35]==2) and np.all(frames[0][1:3,31:33]!=2)
  record=result['frames'][0]
  assert record['observed_target_support']==n
  if n<6:
   assert predictor.calls==0
   assert np.array_equal(frames[0]==1,previous==1)
   assert np.all(frames[0][1:3,31:33]==0)
   assert record['mode']=='sam2_propagation_low_sfm_support'
   assert record['positive_hit'] is None and record['negative_hit'] is None
   assert result['low_support_frames']==['frame.jpg']
  else:
   assert predictor.calls==1 and result['low_support_frames']==[]
   assert record['mode']=='sfm_sam_refined'
   assert record['positive_hit']==1 and record['negative_hit']==0
 frames={0:previous.copy(),1:previous.copy()};predictor=GuardPredictor()
 support=({0:(np.empty(0,dtype=int),xy[:0],np.array([30,30,36,36])),
           1:(np.arange(6),xy,np.array([30,30,36,36]))},
          np.ones(6,bool),np.zeros(6,bool),{})
 with patch('scripts.mask_generation.instance_mask_guard.build_support',return_value=support),redirect_stdout(io.StringIO()):
  result=refine_training_masks(p,[image,image],frames,0,predictor)
 assert len(result['frames'])==2 and predictor.calls==1
 assert np.array_equal(frames[0]==1,previous==1)
print('MASK_INSTANCE_GUARD_TESTS_PASSED')
