"""CPU checks for source-mask instance constraints (Python 3.10 environment)."""
import sys,tempfile,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from scripts.mask_generation.instance_mask_guard import dense_components,sample_prompts
from scripts.validate_semantic_inputs import validate_masks

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
print('MASK_INSTANCE_GUARD_TESTS_PASSED')
