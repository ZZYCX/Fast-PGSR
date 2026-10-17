"""Independent geometric fixtures for the single-plant export filter (CPU)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.export_target_gaussians import spatial_target_selection

def grid(center, n):
    x=np.arange(n, dtype=float)*0.02
    return np.column_stack((x,x%0.06,x%0.04))+center

# The cube is deliberately separate from the plant; a valid detached leaf must
# survive, but a remote dense neighbour and isolated near/far points must not.
main=grid(np.array([0.,0.,2.]),40)
leaf=grid(np.array([1.1,0.,2.]),8)
neighbour=grid(np.array([5.,0.,2.]),12)
cube=grid(np.array([0.,0.,0.]),10)
xyz=np.vstack((main,leaf,neighbour,cube,[[.4,.7,2.],[30,30,30]]))
oid=np.array([1]*60+[2]*10+[1,1])
keep,report=spatial_target_selection(xyz,oid,.2)
expected=np.array([True]*48+[False]*12+[True]*10+[False,False])
assert np.array_equal(keep,expected),np.flatnonzero(keep!=expected)
assert report['classes']['2']['kept']==10
assert report['classes']['1']['removed']==14

# The same scene at another scale produces the same selection when the physical
# radius is scaled with it. No largest-component heuristic may guess a tie.
scaled,_=spatial_target_selection(xyz*3,oid,.6)
assert np.array_equal(scaled,expected)
try:
    spatial_target_selection(np.vstack((main,main+[10,0,0])),np.ones(80),.2)
except RuntimeError as exc:
    assert '主实例' in str(exc)
else:
    raise AssertionError('ambiguous instances were silently accepted')
for bad in [0,-1,float('nan'),float('inf')]:
    try:spatial_target_selection(xyz,oid,bad)
    except RuntimeError:pass
    else:raise AssertionError(bad)
print('SPATIAL_TARGET_TESTS_PASSED')
