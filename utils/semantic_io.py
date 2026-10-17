"""CPU-only semantic sidecar checksums, shared with training and export."""
import hashlib
import numpy as np

SEMANTIC_STATE_VERSION = 2

def _xyz_checksum(xyz_np):
    """SHA-256 over the row-ordered float32 XYZ (binds a PLY to its saved order)."""
    xyz = np.ascontiguousarray(np.asarray(xyz_np, dtype=np.float32))
    return hashlib.sha256(xyz.tobytes()).hexdigest()


def _semantic_checksum(obj_id, prob, conf, src, ev, oc, sc, cand_id=None, cand_stable=None):
    """SHA-256 over the row-ordered semantic arrays (detects any row permutation of
    semantic_state.npz itself). The two candidate arrays are optional so legacy v1
    sidecars (which never had them) validate against their own checksum."""
    parts = [np.ascontiguousarray(a) for a in (obj_id, prob, conf, src, ev, oc, sc)]
    if cand_id is not None:
        parts.append(np.ascontiguousarray(cand_id))
    if cand_stable is not None:
        parts.append(np.ascontiguousarray(cand_stable))
    return hashlib.sha256(b"".join(p.tobytes() for p in parts)).hexdigest()

