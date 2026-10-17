# Lean LeafFit plant segmentation script
# Core Apex->Root logic: Clean Graph + Heat geodesic Gr + turn penalty.
# No intersection-based secondary path refinement.

import argparse
from collections import deque
import colorsys
import heapq
import re
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
import fpsample
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

@dataclass
class GaussianData:
    xyz: np.ndarray
    rot: np.ndarray
    scale: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray
    nxnynz: np.ndarray
    filter_3Ds: np.ndarray

    def __len__(self):
        return len(self.xyz)
gm = types.ModuleType('gaussian_utils')
gm.GaussianData = GaussianData
sys.modules['gaussian_utils'] = gm
CROSS_SECTION_GRAPH_HOPS = 12
CROSS_SECTION_MAX_NODES = 1024

def _clean_graph_cross_section_candidates(center_idx, da_field, root_field, clean_adj, tolerance_percentage=0.02, max_hops=CROSS_SECTION_GRAPH_HOPS, max_nodes=CROSS_SECTION_MAX_NODES):
    """
    Return the LOCAL connected Da+RootField cross-section containing center_idx.

    Hard conditions:
      1) candidate must lie in the Da iso-band of center_idx;
      2) candidate must lie in the RootField iso-band of center_idx;
      3) candidate must be reachable from center_idx through Clean-Graph edges
         while staying inside the same iso-band;
      4) traversal is locally capped to avoid a section leaking through a long
         graph chain into another organ.

    Tolerances are enlarged only when too few points survive.
    """
    center_idx = int(center_idx)
    da_field = np.asarray(da_field, dtype=np.float64)
    root_field = np.asarray(root_field, dtype=np.float64)
    n = len(da_field)
    if clean_adj is None or center_idx < 0 or center_idx >= n or (len(clean_adj) != n):
        return np.empty((0,), dtype=np.int64)
    da0 = float(da_field[center_idx])
    r0 = float(root_field[center_idx])
    da_range = float(np.nanmax(da_field) - np.nanmin(da_field))
    r_range = float(np.nanmax(root_field) - np.nanmin(root_field))
    da_floor = max(da_range * 0.0001, 1e-08)
    r_floor = max(r_range * 0.0001, 1e-08)
    best = np.asarray([center_idx], dtype=np.int64)
    for scale in (1.0, 1.5, 2.0, 3.0):
        da_tol = max(abs(da0) * float(tolerance_percentage) * scale, da_floor * scale)
        r_tol = max(abs(r0) * float(tolerance_percentage) * scale, r_floor * scale)

        def in_band(idx):
            idx = int(idx)
            return abs(float(da_field[idx]) - da0) <= da_tol and abs(float(root_field[idx]) - r0) <= r_tol
        visited = {center_idx}
        q = deque([(center_idx, 0)])
        component = [center_idx]
        while q and len(component) < int(max_nodes):
            u, depth = q.popleft()
            if depth >= int(max_hops):
                continue
            nbr = clean_adj[int(u)]
            if nbr is None:
                continue
            for v in np.asarray(nbr, dtype=np.int64):
                v = int(v)
                if v in visited:
                    continue
                visited.add(v)
                if not in_band(v):
                    continue
                component.append(v)
                q.append((v, depth + 1))
                if len(component) >= int(max_nodes):
                    break
        idx = np.asarray(sorted(set(map(int, component))), dtype=np.int64)
        if len(idx) > len(best):
            best = idx
        if len(idx) >= 5:
            return idx
    return best

def _cross_section_center_geometry(xyz, center_idx, candidate_idx):
    """
    Estimate a two-sided transverse section around center_idx.

    The connected cross-section candidates should form a narrow band. PCA on
    that band gives its dominant transverse direction e. Candidates are
    projected onto e relative to center_idx.

                         endpoint_a
                             |
        a ----------- center/pi ----------- b
                             |
                         endpoint_b

    A good centerline point has comparable distances to both sides:

        balance ~= 1
        imbalance ~= 0

    An edge-biased point:

        a ---------------- pi -- b

    has:

        balance -> 0
        imbalance -> 1

    Returns a dictionary with endpoints, side lengths, total span and balance.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    center_idx = int(center_idx)
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    invalid = {'valid': False, 'endpoint_a': -1, 'endpoint_b': -1, 'side_a': 0.0, 'side_b': 0.0, 'width': np.nan, 'balance': 0.0, 'imbalance': 1.0}
    if len(candidate_idx) < 3:
        return invalid
    if center_idx not in set(map(int, candidate_idx)):
        candidate_idx = np.concatenate([np.asarray([center_idx], dtype=np.int64), candidate_idx])
    rel = np.asarray(xyz[candidate_idx], dtype=np.float64) - np.asarray(xyz[center_idx], dtype=np.float64)[None, :]
    nonzero_norm = np.linalg.norm(rel, axis=1)
    usable = nonzero_norm > 1e-10
    if np.count_nonzero(usable) < 2:
        return invalid
    cov = rel[usable].T @ rel[usable]
    cov /= max(int(np.count_nonzero(usable)), 1)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return invalid
    axis = eigvecs[:, int(np.argmax(eigvals))]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        return invalid
    axis = axis / axis_norm
    proj = rel @ axis
    neg = np.where(proj < -1e-09)[0]
    pos = np.where(proj > 1e-09)[0]
    if len(neg) == 0 or len(pos) == 0:
        return invalid
    local_a = int(neg[np.argmin(proj[neg])])
    local_b = int(pos[np.argmax(proj[pos])])
    endpoint_a = int(candidate_idx[local_a])
    endpoint_b = int(candidate_idx[local_b])
    side_a = float(-proj[local_a])
    side_b = float(proj[local_b])
    width = float(side_a + side_b)
    if not np.isfinite(width) or width <= 1e-10 or side_a <= 1e-10 or (side_b <= 1e-10):
        return invalid
    balance = float(min(side_a, side_b) / max(side_a, side_b))
    imbalance = float(abs(side_a - side_b) / max(side_a + side_b, 1e-10))
    return {'valid': True, 'endpoint_a': endpoint_a, 'endpoint_b': endpoint_b, 'side_a': side_a, 'side_b': side_b, 'width': width, 'balance': balance, 'imbalance': imbalance}

def args_parse():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--leaf-fit-root', default='third_party/leaf_fit')
    p.add_argument('--pick-root', action='store_true', help='Select Root interactively with Open3D. Shift+左键选点，Shift+右键撤销，按 Q 结束。Default (without this flag): use the input Gaussian whose center has the minimum Z coordinate.')
    p.add_argument('--root-point-size', type=float, default=4.0)
    p.add_argument('--fps-points', type=int, default=8192)
    p.add_argument('--tip-k', type=int, default=None)
    p.add_argument('--path-k', type=int, default=None)
    p.add_argument('--full-cross-section-span', action='store_true', help='Legacy optional mode: use the full maximum pairwise span inside the LOCAL Clean-Graph cross-section. By default, the section is two-sided around the current path point (a---pi---b), and its width is the sum of the two projected side lengths.')
    p.add_argument('--root-radius', type=float, default=0.1)
    p.add_argument('--geometry-scale', type=float, default=1.0)
    p.add_argument('--boundary-tol', '--junction-tol', dest='boundary_tol', type=float, default=0.02, help='Relative slack for the selected Apex->Boundary semantic extraction. Boundary is the nearer one of Base/Junction on the Apex->Root path. Used by both Da and Dr constraints. Default: 0.02.')
    p.add_argument('--disable-apex-cluster', action='store_true', help='Disable LeafFit apex grouping. Each detected Apex is treated as one independent single-tip leaf candidate.')
    return p.parse_args()

def read_ply(path):
    ply = PlyData.read(str(path))
    v = ply['vertex'].data
    names = v.dtype.names or ()
    for k in ('x', 'y', 'z'):
        if k not in names:
            raise RuntimeError('PLY missing property: %s' % k)
    xyz = np.column_stack([v['x'], v['y'], v['z']]).astype(np.float32)
    return (ply, xyz)

def make_gaussians(xyz):
    n = len(xyz)
    return GaussianData(xyz=xyz.astype(np.float32, copy=False), rot=np.tile(np.array([[1, 0, 0, 0]], np.float32), (n, 1)), scale=np.ones((n, 3), np.float32), opacity=np.ones((n, 1), np.float32), sh=np.zeros((n, 3), np.float32), nxnynz=np.zeros((n, 3), np.float32), filter_3Ds=np.zeros((n, 1), np.float32))

def rgb_from_ply(ply):
    v = ply['vertex'].data
    names = v.dtype.names or ()
    if not all((k in names for k in ('f_dc_0', 'f_dc_1', 'f_dc_2'))):
        return None
    sh = np.column_stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']])
    return np.clip(0.28209479177387814 * sh + 0.5, 0.0, 1.0)

def write_subset_original(ply, indices, path):
    """
    Write a subset of the ORIGINAL Gaussian records without recoloring or
    changing opacity/scale/rotation/SH.

    Returns the resolved output Path and verifies that the file was actually
    created. This makes leaves.ply/stem.ply failures impossible to miss.
    """
    indices = np.asarray(indices, np.int64)
    v = ply['vertex'].data[indices].copy()
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    out_ply = PlyData([PlyElement.describe(v, 'vertex')], text=ply.text, byte_order=ply.byte_order, comments=list(ply.comments), obj_info=list(ply.obj_info))
    out_ply.write(str(path))
    if not path.is_file():
        raise RuntimeError('PLY write failed: %s' % path)
    if path.stat().st_size <= 0:
        raise RuntimeError('PLY file is empty after write: %s' % path)
    return path

def path_palette(n):
    """
    High-contrast deterministic colors for Apex->Root paths.
    Keypoint colors (red/magenta/blue/yellow) are applied afterward.
    """
    out = []
    h = 0.0
    golden = 0.618033988749895
    for i in range(int(n)):
        h = (h + golden) % 1.0
        out.append(colorsys.hsv_to_rgb(h, 0.9, 1.0))
    return np.asarray(out, dtype=np.float64)

def flatten(values, n):
    out = []

    def f(x):
        if x is None:
            return
        if np.isscalar(x):
            try:
                out.append(int(x))
            except Exception:
                pass
            return
        try:
            for y in x:
                f(y)
        except TypeError:
            pass
    f(values)
    if not out:
        return np.empty(0, np.int64)
    a = np.unique(np.asarray(out, np.int64))
    return a[(a >= 0) & (a < n)]

def pick_root(xyz, ply, point_size):
    try:
        import open3d as o3d
    except ImportError:
        raise RuntimeError('Open3D is required for --pick-root.')
    while True:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, np.float64))
        rgb = rgb_from_ply(ply)
        if rgb is not None:
            pcd.colors = o3d.utility.Vector3dVector(rgb)
        print('\nShift+左键选 Root，Shift+右键撤销，Q 结束。')
        vis = o3d.visualization.VisualizerWithEditing()
        if not vis.create_window('LeafFit - Pick Root', 1280, 900):
            raise RuntimeError('Open3D window creation failed.')
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        if opt is not None:
            opt.point_size = float(point_size)
        vis.run()
        picked = list(vis.get_picked_points())
        vis.destroy_window()
        if not picked:
            continue
        idx = int(picked[-1])
        print('Root: %d, xyz=%s' % (idx, xyz[idx]))
        return idx

def load_leaf_fit(core, tip_k, path_k, disable_apex_cluster=False, full_cross_section_span=False):
    sys.path.insert(0, str(core))
    pp = core / 'petiole_detection.py'
    s = pp.read_text(encoding='utf-8')
    pat = '(?P<i>[ \\t]*)min_idx_after_protection\\s*=\\s*np\\.argmin\\(selected_dists\\[protection_period:\\]\\)\\s*\\+\\s*protection_period'

    def repl(m):
        i = m.group('i')
        return i + 'search_start = min(protection_period, len(selected_dists) - 1)\n' + i + 'tail_dists = selected_dists[search_start:]\n' + i + 'min_idx_after_protection = np.argmin(tail_dists) + search_start'
    s = re.sub(pat, repl, s, count=1)
    pm = types.ModuleType('petiole_detection')
    pm.__file__ = str(pp)
    pm.__package__ = ''
    sys.modules['petiole_detection'] = pm
    exec(compile(s, str(pp), 'exec'), pm.__dict__)
    ap = core / 'auto_segment.py'
    s = ap.read_text(encoding='utf-8')
    if 'from __future__ import annotations' not in s:
        s = 'from __future__ import annotations\n' + s
    s, a = re.subn('k\\s*=\\s*len\\(sparse_indices\\)\\s*//\\s*64', 'k=(LEAFFIT_TIP_K if LEAFFIT_TIP_K is not None else len(sparse_indices)//64)', s, count=1)
    s, b = re.subn('k\\s*=\\s*len\\(sparse_indices\\)\\s*//\\s*32', 'k=(LEAFFIT_PATH_K if LEAFFIT_PATH_K is not None else len(sparse_indices)//32)', s, count=1)
    if a != 1 or b != 1:
        raise RuntimeError('Could not patch LeafFit tip/path KNN expressions.')
    am = types.ModuleType('auto_segment')
    am.__file__ = str(ap)
    am.__package__ = ''
    am.LEAFFIT_TIP_K = tip_k
    am.LEAFFIT_PATH_K = path_k
    am.LEAFFIT_BASE_CROSS_SECTIONS = {}
    sys.modules['auto_segment'] = am
    exec(compile(s, str(ap), 'exec'), am.__dict__)
    _original_find_path_from_tip_to_root = am.find_path_from_tip_to_root
    am.LEAFFIT_CLEAN_ADJ = None
    am.LEAFFIT_CLEAN_EDGE_SET = None
    am.LEAFFIT_ROOT_SOURCE_SET = None
    am.LEAFFIT_ROOT_GRAPH_DISTANCE = None
    am.LEAFFIT_GLOBAL_PATH_STATS = []

    def _robust_field_scale(values):
        """
        Robust positive scale used only to make the different path penalties
        dimensionless. It does NOT change the underlying Dr/Da/Temperature.
        """
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return 1.0
        q10, q90 = np.percentile(values, [10.0, 90.0])
        scale = float(q90 - q10)
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.max(values) - np.min(values))
        return max(scale, 1e-08)

    def _reconstruct_path(prev, start_idx, end_idx):
        path = []
        cur = int(end_idx)
        while cur >= 0:
            path.append(cur)
            if cur == int(start_idx):
                break
            cur = int(prev[cur])
        if not path or path[-1] != int(start_idx):
            return [int(start_idx)]
        path.reverse()
        return path

    def _reconstruct_state_path(parent_state, final_state, start_state):
        """
        Reconstruct node path from directed Dijkstra states:
            state = (previous_node, current_node)
        """
        states = []
        state = final_state
        while True:
            states.append(state)
            if state == start_state:
                break
            state = parent_state.get(state)
            if state is None:
                return [int(start_state[1])]
        states.reverse()
        path = [int(states[0][1])]
        for state in states[1:]:
            node = int(state[1])
            if node != path[-1]:
                path.append(node)
        return path

    def _path_turn_statistics(path, xyz):
        """
        Return mean/max turning angle in degrees and the number of sharp
        turns >= 60 degrees. 0 degree means continuing straight ahead.
        """
        if len(path) < 3:
            return (0.0, 0.0, 0)
        angles = []
        for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
            v1 = xyz[int(b)] - xyz[int(a)]
            v2 = xyz[int(c)] - xyz[int(b)]
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 <= 1e-12 or n2 <= 1e-12:
                continue
            cos_theta = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            angles.append(float(np.degrees(np.arccos(cos_theta))))
        if not angles:
            return (0.0, 0.0, 0)
        arr = np.asarray(angles, dtype=np.float64)
        return (float(np.mean(arr)), float(np.max(arr)), int(np.count_nonzero(arr >= 60.0)))

    def _find_path_from_tip_to_root_clean(gaussians, temperature_field, tip_idx, heat_source_idx, distance_param, is_path_marks, k=128):
        """
        Global Apex->Root search without any center/boundary preference.

        Hard constraint:
          - every transition must be an edge of the filtered Clean Graph.

        Soft costs:
          1) Root direction from Heat-Method geodesic Root distance Gr;
          2) turn continuity;
          3) weak original LeafFit preference:
                 Temperature - alpha * Da.

        No boundary-distance, center, or cross-section-symmetry term is used
        for Apex->Root path selection.

        Gr is the multi-source Heat geodesic distance to the Root region.
        It remains SOFT: lateral or slightly geodesically Root-away motion is
        allowed, but receives a larger cost than a geodesically Root-directed
        step.
        """
        clean_adj = getattr(am, 'LEAFFIT_CLEAN_ADJ', None)
        Gr = getattr(am, 'LEAFFIT_ROOT_GRAPH_DISTANCE', None)
        if clean_adj is None or Gr is None:
            return _original_find_path_from_tip_to_root(gaussians, temperature_field, tip_idx, heat_source_idx, distance_param, is_path_marks, k=k)
        if distance_param is None:
            raise ValueError('Invalid distance parameter')
        method = distance_param.get('method')
        dense_solver = distance_param.get('dense_solver')
        if dense_solver is None:
            raise ValueError('distance_param has no dense_solver')
        if method == 'euclidean':
            da_weight = 0.25
        elif method == 'geodesic':
            da_weight = 0.5
        else:
            raise ValueError('Invalid distance parameter')
        xyz = np.asarray(gaussians.xyz, dtype=np.float64)
        raw_temperature = np.asarray(temperature_field, dtype=np.float64)
        Gr = np.asarray(Gr, dtype=np.float64)
        tip_idx = int(tip_idx)
        target = int(heat_source_idx)
        Da = np.asarray(dense_solver.compute_distance(tip_idx), dtype=np.float64)
        leaf_fit_score = raw_temperature - da_weight * Da
        score_scale = _robust_field_scale(leaf_fit_score)
        root_source_set = getattr(am, 'LEAFFIT_ROOT_SOURCE_SET', None)
        if not root_source_set:
            root_source_set = {target}
        else:
            root_source_set = set(map(int, root_source_set))
        if not np.isfinite(Gr[tip_idx]):
            am.LEAFFIT_GLOBAL_PATH_STATS.append({'tip': tip_idx, 'status': 'clean_graph_disconnected', 'path_len': 1, 'visited': 1, 'target': tip_idx, 'cost': 0.0, 'mean_turn_deg': 0.0, 'max_turn_deg': 0.0, 'sharp_turns_60': 0, 'gr_start': np.inf, 'gr_end': np.inf})
            return [tip_idx]
        if tip_idx in root_source_set or Gr[tip_idx] <= 1e-12:
            am.LEAFFIT_GLOBAL_PATH_STATS.append({'tip': tip_idx, 'status': 'already_in_root', 'path_len': 1, 'visited': 1, 'target': tip_idx, 'cost': 0.0, 'mean_turn_deg': 0.0, 'max_turn_deg': 0.0, 'sharp_turns_60': 0, 'gr_start': float(Gr[tip_idx]), 'gr_end': float(Gr[tip_idx])})
            return [tip_idx]
        start_state = (-1, tip_idx)
        best = {start_state: 0.0}
        parent_state = {start_state: None}
        settled = set()
        heap = [(0.0, -1, tip_idx)]
        final_state = None
        visited_state_count = 0
        TURN_WEIGHT = 2.5
        ROOT_DIRECTION_WEIGHT = 3.0
        LEAFFIT_AUX_WEIGHT = 0.2
        MARKED_PATH_FACTOR = 0.97
        while heap:
            current_cost, prev, u = heapq.heappop(heap)
            prev = int(prev)
            u = int(u)
            state = (prev, u)
            if state in settled:
                continue
            known = best.get(state, np.inf)
            if current_cost > known + 1e-12:
                continue
            settled.add(state)
            visited_state_count += 1
            if u in root_source_set or Gr[u] <= 1e-10:
                final_state = state
                break
            nbr = clean_adj[u]
            if nbr is None or len(nbr) == 0:
                continue
            incoming = None
            incoming_norm = 0.0
            if prev >= 0:
                incoming = xyz[u] - xyz[prev]
                incoming_norm = float(np.linalg.norm(incoming))
            for v in np.asarray(nbr, dtype=np.int64):
                v = int(v)
                if prev >= 0 and v == prev:
                    continue
                if not np.isfinite(Gr[v]):
                    continue
                outgoing = xyz[v] - xyz[u]
                edge_len = float(np.linalg.norm(outgoing))
                if not np.isfinite(edge_len) or edge_len <= 1e-12:
                    continue
                gr_drop = float(Gr[u] - Gr[v])
                gr_slope = float(np.clip(gr_drop / max(edge_len, 1e-12), -1.0, 1.0))
                root_direction_penalty = (1.0 - gr_slope) ** 2
                turn_penalty = 0.0
                if incoming is not None and incoming_norm > 1e-12:
                    cos_theta = float(np.clip(np.dot(incoming, outgoing) / (incoming_norm * edge_len), -1.0, 1.0))
                    turn_penalty = (1.0 - cos_theta) ** 2
                leaf_fit_score_drop = max(0.0, float(leaf_fit_score[u] - leaf_fit_score[v])) / score_scale
                edge_cost = edge_len * (1.0 + ROOT_DIRECTION_WEIGHT * root_direction_penalty + TURN_WEIGHT * turn_penalty + LEAFFIT_AUX_WEIGHT * leaf_fit_score_drop)
                if is_path_marks[v] >= 1:
                    edge_cost *= MARKED_PATH_FACTOR
                next_state = (u, v)
                new_cost = current_cost + edge_cost
                if new_cost + 1e-12 < best.get(next_state, np.inf):
                    best[next_state] = new_cost
                    parent_state[next_state] = state
                    heapq.heappush(heap, (new_cost, u, v))
        if final_state is not None:
            path = _reconstruct_state_path(parent_state, final_state, start_state)
            status = 'reached_root'
            final_target = int(final_state[1])
            final_cost = float(best[final_state])
        else:
            candidate_states = [state for state in best if state != start_state and np.isfinite(best[state])]
            if candidate_states:
                final_state = min(candidate_states, key=lambda s: (float(Gr[int(s[1])]), float(best[s])))
                path = _reconstruct_state_path(parent_state, final_state, start_state)
                final_target = int(final_state[1])
                final_cost = float(best[final_state])
            else:
                path = [tip_idx]
                final_target = tip_idx
                final_cost = 0.0
            status = 'root_guided_search_failed'
        for node in path[1:]:
            is_path_marks[int(node)] += 1
        mean_turn_deg, max_turn_deg, sharp_turns_60 = _path_turn_statistics(path, xyz)
        am.LEAFFIT_GLOBAL_PATH_STATS.append({'tip': tip_idx, 'status': status, 'path_len': int(len(path)), 'visited': int(visited_state_count), 'target': int(final_target), 'cost': float(final_cost), 'mean_turn_deg': float(mean_turn_deg), 'max_turn_deg': float(max_turn_deg), 'sharp_turns_60': int(sharp_turns_60), 'gr_start': float(Gr[tip_idx]), 'gr_end': float(Gr[final_target])})
        return path
    am.find_path_from_tip_to_root = _find_path_from_tip_to_root_clean
    if not hasattr(am, 'find_base_idx_by_geodesic_tip_graph'):
        raise RuntimeError('LeafFit auto_segment.py has no find_base_idx_by_geodesic_tip_graph().')
    _original_base_detector = am.find_base_idx_by_geodesic_tip_graph
    BASE_EQUAL_DA_SAMPLES = 48
    BASE_PRE_WINDOW = 4
    BASE_POST_WINDOW = 5
    BASE_FLAT_SLOPE_RATIO = 0.3
    BASE_STABLE_CV = 0.15
    BASE_DESCENDING_FRACTION = 0.67

    def _exact_farthest_pair_width(coords, block_size=256, return_pair=False):
        """
        Exact cross-section diameter:
            max ||xi - xj||_2

        When return_pair=True, also return the two LOCAL indices (i, j)
        whose straight Euclidean segment is the actual width used by the
        Base detector. These endpoints are later visualized in root_path.ply.
        """
        coords = np.asarray(coords, dtype=np.float64)
        m = len(coords)
        if m < 2:
            if return_pair:
                return (np.nan, -1, -1)
            return np.nan
        max_d2 = -1.0
        best_i = -1
        best_j = -1
        for i0 in range(0, m, block_size):
            a = coords[i0:i0 + block_size]
            diff = a[:, None, :] - coords[None, :, :]
            d2 = np.einsum('ijk,ijk->ij', diff, diff)
            flat = int(np.argmax(d2))
            local_d2 = float(d2.reshape(-1)[flat])
            if local_d2 > max_d2:
                ii, jj = np.unravel_index(flat, d2.shape)
                max_d2 = local_d2
                best_i = int(i0 + ii)
                best_j = int(jj)
        width = float(np.sqrt(max(max_d2, 0.0)))
        if return_pair:
            return (width, best_i, best_j)
        return width

    def _path_point_to_edge_width(xyz, path_idx, candidate_idx, return_endpoint=False):
        """
        Default / previous cross-section logic.

        Keep the current path point p_i fixed and find the candidate g* with
        the largest Euclidean distance from p_i:

            W_i = max_{g in S_i} ||x(p_i) - x(g)||_2

        This behaves like a path-centered local half-width / radius indicator.
        """
        candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
        if len(candidate_idx) == 0:
            if return_endpoint:
                return (np.nan, -1)
            return np.nan
        center = np.asarray(xyz[int(path_idx)], dtype=np.float64)
        coords = np.asarray(xyz[candidate_idx], dtype=np.float64)
        dist = np.linalg.norm(coords - center[None, :], axis=1)
        local = int(np.argmax(dist))
        width = float(dist[local])
        endpoint = int(candidate_idx[local])
        if return_endpoint:
            return (width, endpoint)
        return width

    def _equal_da_path_samples(path, da_field, max_samples):
        """
        Approximately equally spaced samples in Da along Apex->Root path.
        Each equal-Da target is mapped back to the nearest discrete path node.
        """
        path = [int(x) for x in path]
        if len(path) <= 2:
            return path
        values = np.asarray([float(da_field[idx]) for idx in path], dtype=np.float64)
        finite = np.isfinite(values)
        if not np.all(finite):
            path = [p for p, ok in zip(path, finite) if ok]
            values = values[finite]
        if len(path) <= 2:
            return path
        mono = np.maximum.accumulate(values)
        if float(mono[-1] - mono[0]) <= 1e-10:
            return path
        count = max(3, min(int(max_samples), len(path)))
        targets = np.linspace(mono[0], mono[-1], count)
        positions = []
        for target in targets:
            j = int(np.searchsorted(mono, target, side='left'))
            if j <= 0:
                pos = 0
            elif j >= len(mono):
                pos = len(mono) - 1
            else:
                left = j - 1
                right = j
                if abs(mono[left] - target) <= abs(mono[right] - target):
                    pos = left
                else:
                    pos = right
            if not positions or pos != positions[-1]:
                positions.append(pos)
        return [path[pos] for pos in positions]

    def _cross_section_candidates(path_idx, da_field, root_field, tolerance_percentage):
        """
        Base cross-section = LOCAL Clean-Graph connected iso-band.

        Compared with the previous global Da+RootField mask, a candidate must
        now be connected to path_idx through the filtered Clean Graph while
        remaining inside the same Da + RootField band.
        """
        clean_adj = getattr(am, 'LEAFFIT_CLEAN_ADJ', None)
        if clean_adj is None:
            return np.empty((0,), dtype=np.int64)
        return _clean_graph_cross_section_candidates(path_idx, da_field, root_field, clean_adj, tolerance_percentage=tolerance_percentage)

    def _median_smooth_widths(widths):
        """
        Very light 3-sample median smoothing. This suppresses a single bad
        cross-section without flattening the leaf->petiole transition.
        """
        w = np.asarray(widths, dtype=np.float64)
        out = w.copy()
        for i in range(len(w)):
            lo = max(0, i - 1)
            hi = min(len(w), i + 2)
            local = w[lo:hi]
            local = local[np.isfinite(local)]
            if len(local):
                out[i] = float(np.median(local))
        return out

    def _linear_slope(values):
        """Least-squares slope with sample index as the independent variable."""
        y = np.asarray(values, dtype=np.float64)
        if len(y) < 2 or not np.all(np.isfinite(y)):
            return np.nan
        x = np.arange(len(y), dtype=np.float64)
        x = x - np.mean(x)
        y0 = y - np.mean(y)
        denom = float(np.dot(x, x))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(x, y0) / denom)

    def _normalized_slope(values):
        """
        Scale-independent local slope:
            raw slope / local median width
        so large and small leaves are judged by trend rather than size.
        """
        y = np.asarray(values, dtype=np.float64)
        med = float(np.median(np.abs(y)))
        if med <= 1e-10:
            return np.nan
        return _linear_slope(y) / med

    def _find_base_by_width_trend(cluster_info_item, gaussians, dense_solver, temperature_field, min_distance_threshold=0.05, tolerance_percentage=0.02, last_virtual_path_distance_factor=2.5, protection_period_ratio=0.25, debug_vis=False):
        tips = cluster_info_item.get('tips', [])
        path = [int(x) for x in cluster_info_item.get('potential_base_idx', [])]
        final_tip = int(cluster_info_item.get('final_tip', tips[0] if tips else path[0] if path else 0))
        cluster_info_item['_base_cross_sections'] = []
        cluster_info_item['_base_cross_section_tip'] = final_tip
        am.LEAFFIT_BASE_CROSS_SECTIONS[final_tip] = []
        if not path:
            return final_tip
        lca = cluster_info_item.get('lca', None)
        if lca is not None and int(lca) in path:
            path = path[:path.index(int(lca)) + 1]
        min_needed = BASE_PRE_WINDOW + BASE_POST_WINDOW + 1
        if len(path) < min_needed:
            return _original_base_detector(cluster_info_item, gaussians, dense_solver, temperature_field, min_distance_threshold=min_distance_threshold, tolerance_percentage=tolerance_percentage, last_virtual_path_distance_factor=last_virtual_path_distance_factor, protection_period_ratio=protection_period_ratio, debug_vis=debug_vis)
        da_field = dense_solver.compute_distance(final_tip)
        root_field = np.asarray(temperature_field, dtype=np.float64)
        sampled_path = _equal_da_path_samples(path, da_field, BASE_EQUAL_DA_SAMPLES)
        if len(sampled_path) < min_needed:
            return _original_base_detector(cluster_info_item, gaussians, dense_solver, temperature_field, min_distance_threshold=min_distance_threshold, tolerance_percentage=tolerance_percentage, last_virtual_path_distance_factor=last_virtual_path_distance_factor, protection_period_ratio=protection_period_ratio, debug_vis=debug_vis)
        widths = []
        candidate_counts = []
        cross_sections = []
        for sample_i, path_idx in enumerate(sampled_path):
            candidate_idx = _cross_section_candidates(path_idx, da_field, root_field, tolerance_percentage)
            candidate_counts.append(len(candidate_idx))
            section = {'sample_i': int(sample_i), 'path_idx': int(path_idx), 'candidate_count': int(len(candidate_idx)), 'endpoint_a': -1, 'endpoint_b': -1, 'width': np.nan, 'balance': 0.0}
            if len(candidate_idx) < 2:
                widths.append(np.nan)
                cross_sections.append(section)
                continue
            center_geom = _cross_section_center_geometry(gaussians.xyz, path_idx, candidate_idx)
            if not center_geom['valid']:
                widths.append(np.nan)
                cross_sections.append(section)
                continue
            if full_cross_section_span:
                coords = gaussians.xyz[candidate_idx]
                width, local_a, local_b = _exact_farthest_pair_width(coords, return_pair=True)
                endpoint_a = int(candidate_idx[local_a])
                endpoint_b = int(candidate_idx[local_b])
            else:
                endpoint_a = int(center_geom['endpoint_a'])
                endpoint_b = int(center_geom['endpoint_b'])
                width = float(center_geom['width'])
            section['endpoint_a'] = endpoint_a
            section['endpoint_b'] = endpoint_b
            section['width'] = float(width)
            section['balance'] = float(center_geom['balance'])
            widths.append(width)
            cross_sections.append(section)
        cluster_info_item['_base_cross_sections'] = cross_sections
        cluster_info_item['_base_sampled_path'] = [int(x) for x in sampled_path]
        am.LEAFFIT_BASE_CROSS_SECTIONS[final_tip] = cross_sections
        widths = np.asarray(widths, dtype=np.float64)
        smooth = _median_smooth_widths(widths)
        finite_idx = np.where(np.isfinite(smooth))[0]
        if len(finite_idx) < min_needed:
            print('    [Base-trend] tip=%d: too few valid widths; fallback to original Base detector.' % final_tip)
            return _original_base_detector(cluster_info_item, gaussians, dense_solver, temperature_field, min_distance_threshold=min_distance_threshold, tolerance_percentage=tolerance_percentage, last_virtual_path_distance_factor=last_virtual_path_distance_factor, protection_period_ratio=protection_period_ratio, debug_vis=debug_vis)
        max_pos = int(finite_idx[np.argmax(smooth[finite_idx])])
        n = len(sampled_path)
        protection = int(np.ceil(n * float(protection_period_ratio)))
        search_start = max(max_pos + 1, protection, BASE_PRE_WINDOW)
        search_end = n - BASE_POST_WINDOW + 1
        chosen = None
        chosen_stats = None
        for i in range(search_start, search_end):
            before = smooth[i - BASE_PRE_WINDOW:i]
            after = smooth[i:i + BASE_POST_WINDOW]
            if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
                continue
            before_slope = _normalized_slope(before)
            diffs_before = np.diff(before)
            descending_fraction = float(np.mean(diffs_before < 0.0))
            if not np.isfinite(before_slope) or before_slope >= 0.0 or descending_fraction < BASE_DESCENDING_FRACTION:
                continue
            after_slope = _normalized_slope(after)
            after_mean = float(np.mean(after))
            if not np.isfinite(after_slope) or after_mean <= 1e-10:
                continue
            after_cv = float(np.std(after) / after_mean)
            flatter_than_before = abs(after_slope) <= BASE_FLAT_SLOPE_RATIO * abs(before_slope)
            stable_after = after_cv <= BASE_STABLE_CV
            not_rebounding = float(np.median(after)) <= float(np.median(before))
            if flatter_than_before and stable_after and not_rebounding:
                chosen = i
                chosen_stats = (before_slope, after_slope, descending_fraction, after_cv)
                break
        if chosen is not None:
            base_idx = int(sampled_path[chosen])
            before_slope, after_slope, desc_frac, after_cv = chosen_stats
            print('    [Base-trend] tip=%d base=%d sample=%d/%d Wmax@%d slope_before=%.5f slope_after=%.5f descending=%.2f stable_cv=%.3f candidates=%d' % (final_tip, base_idx, chosen, n, max_pos, before_slope, after_slope, desc_frac, after_cv, candidate_counts[chosen]))
            return base_idx
        print('    [Base-trend] tip=%d: no descending->stable transition; fallback to original geodesic_tip_graph Base detector.' % final_tip)
        return _original_base_detector(cluster_info_item, gaussians, dense_solver, temperature_field, min_distance_threshold=min_distance_threshold, tolerance_percentage=tolerance_percentage, last_virtual_path_distance_factor=last_virtual_path_distance_factor, protection_period_ratio=protection_period_ratio, debug_vis=debug_vis)
    am.find_base_idx_by_geodesic_tip_graph = _find_base_by_width_trend
    _original_group_apexes = am.group_apexes_by_inequality

    def _group_apexes_per_tip(found_tips, pathes, root_cahced_distance=None, overlap_cut=0.8, triangle_cut=0.62, slack_eps=1e-09, max_iters=50, dense_solver=None):
        if not disable_apex_cluster:
            return _original_group_apexes(found_tips, pathes, root_cahced_distance=root_cahced_distance, overlap_cut=overlap_cut, triangle_cut=triangle_cut, slack_eps=slack_eps, max_iters=max_iters, dense_solver=dense_solver)
        cluster_info = []
        for tip, path in zip(found_tips, pathes):
            tip = int(tip)
            path = [int(x) for x in path]
            lca = int(path[-1]) if path else tip
            if root_cahced_distance is not None and len(path):
                tip_to_lca = float(root_cahced_distance[tip] - root_cahced_distance[lca])
            else:
                tip_to_lca = 0.0
            cluster_info.append({'tips': [tip], 'tips_dist_to_lca': {tip: tip_to_lca}, 'final_tip': tip, 'lca': lca, 'potential_base_idx': path, 'pathes': [path], 'path': path, 'type': 'single_tip'})
        return cluster_info
    am.group_apexes_by_inequality = _group_apexes_per_tip
    return am

def _collect_apex_root_paths(clusters, n):
    """
    Collect every Apex->Root path from LeafFit cluster_info.

    Prefer cluster["pathes"] because it preserves all per-Apex paths when
    clustering is enabled. In the recommended no-grouping mode there is
    exactly one path per cluster.
    """
    paths = []
    seen = set()
    for c in clusters:
        raw = c.get('pathes', None)
        if raw is None or len(raw) == 0:
            raw = [c.get('path', [])]
        else:
            try:
                first = raw[0]
                if np.isscalar(first):
                    raw = [raw]
            except Exception:
                raw = [c.get('path', [])]
        for p in raw:
            try:
                p = [int(x) for x in p if 0 <= int(x) < n]
            except TypeError:
                continue
            compact = []
            for x in p:
                if not compact or x != compact[-1]:
                    compact.append(x)
            if len(compact) < 2:
                continue
            key = tuple(compact)
            if key not in seen:
                seen.add(key)
                paths.append(compact)
    return paths

def _root_path_render_scale(ply, xyz):
    """
    Automatically choose a visible line-Gaussian sigma and sampling spacing.
    Uses the original 3DGS Gaussian scales when available, otherwise the plant
    bounding-box diagonal.
    """
    v = ply['vertex'].data
    names = v.dtype.names or ()
    diag = float(np.linalg.norm(np.max(xyz, axis=0).astype(np.float64) - np.min(xyz, axis=0).astype(np.float64)))
    diag = max(diag, 1e-06)
    if all((k in names for k in ('scale_0', 'scale_1', 'scale_2'))):
        log_s = np.column_stack([v['scale_0'], v['scale_1'], v['scale_2']]).astype(np.float64)
        s = np.exp(np.clip(log_s, -20.0, 20.0))
        typical_sigma = float(np.median(s))
        if not np.isfinite(typical_sigma) or typical_sigma <= 0:
            typical_sigma = diag * 0.001
    else:
        typical_sigma = diag * 0.001
    line_sigma = max(0.45 * typical_sigma, diag * 0.0006)
    spacing = max(1.15 * line_sigma, diag * 0.0004)
    return (float(line_sigma), float(spacing))

def _make_point_marker_rows(template_vertex, positions, colors, marker_sigma):
    """
    Synthetic Gaussian markers placed exactly on path nodes so the discrete
    Apex->Root path samples are clearly visible in Gaussian viewers.
    """
    positions = np.asarray(positions, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    m = len(positions)
    if m == 0:
        return np.empty((0,), dtype=template_vertex.dtype)
    rows = np.zeros(m, dtype=template_vertex.dtype)
    names = rows.dtype.names or ()
    rows['x'] = positions[:, 0]
    rows['y'] = positions[:, 1]
    rows['z'] = positions[:, 2]
    if all((k in names for k in ('nx', 'ny', 'nz'))):
        rows['nx'] = 0
        rows['ny'] = 0
        rows['nz'] = 0
    if not all((k in names for k in ('f_dc_0', 'f_dc_1', 'f_dc_2'))):
        raise RuntimeError('path-node markers require f_dc_0..2.')
    fdc = (np.clip(colors, 0.0, 1.0) - 0.5) / 0.28209479177387814
    rows['f_dc_0'] = fdc[:, 0]
    rows['f_dc_1'] = fdc[:, 1]
    rows['f_dc_2'] = fdc[:, 2]
    for name in names:
        if name.startswith('f_rest_'):
            rows[name] = 0
    if 'opacity' in names:
        rows['opacity'] = 6.0
    if all((k in names for k in ('scale_0', 'scale_1', 'scale_2'))):
        log_sigma = float(np.log(max(marker_sigma, 1e-10)))
        rows['scale_0'] = log_sigma
        rows['scale_1'] = log_sigma
        rows['scale_2'] = log_sigma
    if all((k in names for k in ('rot_0', 'rot_1', 'rot_2', 'rot_3'))):
        rows['rot_0'] = 1.0
        rows['rot_1'] = 0.0
        rows['rot_2'] = 0.0
        rows['rot_3'] = 0.0
    if 'filter_3D' in names:
        rows['filter_3D'] = 0.0
    return rows

def _make_path_gaussian_rows(template_vertex, positions, colors, line_sigma):
    """
    Create synthetic 3DGS Gaussian rows at line-sample positions.

    These are real vertex/Gaussian records rather than a PLY 'edge' element,
    because normal Gaussian-Splatting viewers usually ignore PLY edges.
    """
    positions = np.asarray(positions, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    m = len(positions)
    if m == 0:
        return np.empty((0,), dtype=template_vertex.dtype)
    rows = np.zeros(m, dtype=template_vertex.dtype)
    names = rows.dtype.names or ()
    rows['x'] = positions[:, 0]
    rows['y'] = positions[:, 1]
    rows['z'] = positions[:, 2]
    if all((k in names for k in ('nx', 'ny', 'nz'))):
        rows['nx'] = 0
        rows['ny'] = 0
        rows['nz'] = 0
    if not all((k in names for k in ('f_dc_0', 'f_dc_1', 'f_dc_2'))):
        raise RuntimeError('root_path.ply requires f_dc_0..2 so path colors can be displayed.')
    fdc = (np.clip(colors, 0.0, 1.0) - 0.5) / 0.28209479177387814
    rows['f_dc_0'] = fdc[:, 0]
    rows['f_dc_1'] = fdc[:, 1]
    rows['f_dc_2'] = fdc[:, 2]
    for name in names:
        if name.startswith('f_rest_'):
            rows[name] = 0
    if 'opacity' in names:
        rows['opacity'] = 5.0
    if all((k in names for k in ('scale_0', 'scale_1', 'scale_2'))):
        log_sigma = float(np.log(max(line_sigma, 1e-10)))
        rows['scale_0'] = log_sigma
        rows['scale_1'] = log_sigma
        rows['scale_2'] = log_sigma
    if all((k in names for k in ('rot_0', 'rot_1', 'rot_2', 'rot_3'))):
        rows['rot_0'] = 1.0
        rows['rot_1'] = 0.0
        rows['rot_2'] = 0.0
        rows['rot_3'] = 0.0
    if 'filter_3D' in names:
        rows['filter_3D'] = 0.0
    return rows

def _visual_cross_section_candidates(path_idx, da_field, root_field, clean_adj, tolerance_percentage=0.02):
    """
    Visual cross-section uses the SAME local Clean-Graph connected iso-band
    idea as the Base detector and path-centerline cost.
    """
    return _clean_graph_cross_section_candidates(path_idx, da_field, root_field, clean_adj, tolerance_percentage=tolerance_percentage)

def _farthest_pair_global_indices(xyz, candidate_idx, block_size=256):
    """
    Among one cross-section candidate set, find the TWO points with the
    largest 3D Euclidean distance.

    Returns:
        endpoint_a_global, endpoint_b_global, width
    """
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    if len(candidate_idx) < 2:
        return (-1, -1, np.nan)
    coords = np.asarray(xyz[candidate_idx], dtype=np.float64)
    m = len(coords)
    max_d2 = -1.0
    best_i = -1
    best_j = -1
    for i0 in range(0, m, int(block_size)):
        a = coords[i0:i0 + int(block_size)]
        diff = a[:, None, :] - coords[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        flat = int(np.argmax(d2))
        local_d2 = float(d2.reshape(-1)[flat])
        if local_d2 > max_d2:
            ii, jj = np.unravel_index(flat, d2.shape)
            max_d2 = local_d2
            best_i = int(i0 + ii)
            best_j = int(jj)
    if best_i < 0 or best_j < 0:
        return (-1, -1, np.nan)
    return (int(candidate_idx[best_i]), int(candidate_idx[best_j]), float(np.sqrt(max(max_d2, 0.0))))

def _compute_path_cross_section_lines(xyz, paths, solver, Dr, node_users, clean_adj, tolerance_percentage=0.02, full_cross_section_span=False):
    """
    Compute ONE transverse width line for EVERY node of EVERY Apex->Root path.

    This does NOT depend on whether the trend Base detector succeeded or
    fell back. Therefore even a short path still gets its cross-section lines.

    For path point p_i:
      1) compute Da from this path's Apex;
      2) use Da + RootField AND Clean-Graph connectivity to obtain the LOCAL
         cross-section containing p_i;
      3) DEFAULT: estimate two opposite section endpoints around p_i:
             a ------- p_i ------- b
         so the visualization directly shows whether the path is centered;
      4) optional --full-cross-section-span keeps the legacy farthest-pair mode,
         but still inside this local Clean-Graph candidate set.

    Shared path nodes are marked so their transverse line can be drawn white.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    Dr = np.asarray(Dr, dtype=np.float64)
    root_field = float(np.max(Dr)) - Dr
    all_sections = []
    for path_id, p in enumerate(paths):
        if len(p) < 1:
            continue
        apex_idx = int(p[0])
        da_field = np.asarray(solver.compute_distance(apex_idx), dtype=np.float64)
        valid_count = 0
        for order_i, path_idx in enumerate(p):
            path_idx = int(path_idx)
            candidate_idx = _visual_cross_section_candidates(path_idx, da_field, root_field, clean_adj, tolerance_percentage=tolerance_percentage)
            if len(candidate_idx) < 2:
                continue
            center_geom = _cross_section_center_geometry(xyz, path_idx, candidate_idx)
            if not center_geom['valid']:
                continue
            if full_cross_section_span:
                a, b, width = _farthest_pair_global_indices(xyz, candidate_idx)
            else:
                a = int(center_geom['endpoint_a'])
                b = int(center_geom['endpoint_b'])
                width = float(center_geom['width'])
            if a < 0 or b < 0 or a == b or (not np.isfinite(width)):
                continue
            shared = len(node_users.get(path_idx, set())) >= 2
            all_sections.append({'path_id': int(path_id), 'path_order': int(order_i), 'path_idx': path_idx, 'endpoint_a': int(a), 'endpoint_b': int(b), 'width': float(width), 'balance': float(center_geom['balance']), 'candidate_count': int(len(candidate_idx)), 'shared': bool(shared)})
            valid_count += 1
        print('      path_%03d apex=%d: cross-section lines=%d/%d path nodes' % (path_id + 1, apex_idx, valid_count, len(p)))
    return all_sections

def save_root_paths(ply, xyz, root_idx, tips, bases, junctions, clusters, solver, Dr, clean_adj, path, full_cross_section_span=False):
    """
    Save debug/root_path.ply.

    Visualization:
      dark/faint plant = spatial reference
      magenta = Apex
      blue    = Base
      yellow  = Junction
      red     = Root

      unique Apex->Root path = one deterministic color per Apex path
      shared/merged path     = WHITE

    Straight connections are represented by densely sampled synthetic small
    Gaussians between consecutive path nodes, so the result stays compatible
    with Gaussian-Splatting PLY viewers.
    """
    n = len(xyz)
    paths = _collect_apex_root_paths(clusters, n)
    if not paths:
        print('      root_path.ply skipped: no Apex->Root paths found')
        return
    v = ply['vertex'].data.copy()
    names = v.dtype.names or ()
    if not all((k in names for k in ('f_dc_0', 'f_dc_1', 'f_dc_2'))):
        raise RuntimeError('PLY has no f_dc_0..2.')
    node_users = {}
    edge_users = {}
    for path_id, p in enumerate(paths):
        for node in p:
            node_users.setdefault(int(node), set()).add(path_id)
        for a, b in zip(p[:-1], p[1:]):
            a, b = (int(a), int(b))
            if a == b:
                continue
            edge = (a, b) if a < b else (b, a)
            edge_users.setdefault(edge, set()).add(path_id)
    palette = path_palette(len(paths))
    white = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    cross_sections = _compute_path_cross_section_lines(xyz, paths, solver, Dr, node_users, clean_adj, tolerance_percentage=0.02, full_cross_section_span=full_cross_section_span)
    if not cross_sections:
        print('      WARNING: no valid cross-section line could be constructed.')
    colors = np.full((n, 3), 0.1, dtype=np.float64)
    for node, users in node_users.items():
        if len(users) >= 2:
            colors[node] = white
        else:
            path_id = next(iter(users))
            colors[node] = palette[path_id]
    aidx = flatten(tips, n)
    bidx = flatten(bases, n)
    jidx = flatten(junctions, n)
    colors[jidx] = [1.0, 0.85, 0.0]
    colors[bidx] = [0.05, 0.25, 1.0]
    colors[aidx] = [1.0, 0.0, 1.0]
    colors[int(root_idx)] = [1.0, 0.05, 0.05]
    fdc = (np.clip(colors, 0.0, 1.0) - 0.5) / 0.28209479177387814
    v['f_dc_0'] = fdc[:, 0]
    v['f_dc_1'] = fdc[:, 1]
    v['f_dc_2'] = fdc[:, 2]
    for name in names:
        if name.startswith('f_rest_'):
            v[name] = 0
    path_nodes = np.asarray(sorted(node_users.keys()), dtype=np.int64)
    visible = np.zeros(n, dtype=bool)
    visible[path_nodes] = True
    visible[aidx] = True
    visible[bidx] = True
    visible[jidx] = True
    visible[int(root_idx)] = True
    if 'opacity' in names:
        v['opacity'][~visible] = -5.0
        v['opacity'][visible] = 5.0
    if all((k in names for k in ('scale_0', 'scale_1', 'scale_2'))):
        marker_idx = np.unique(np.concatenate([path_nodes, aidx, bidx, jidx, np.asarray([int(root_idx)], dtype=np.int64)]))
        if len(marker_idx):
            logs = np.column_stack([v['scale_0'][marker_idx], v['scale_1'][marker_idx], v['scale_2'][marker_idx]]).astype(np.float64)
            median_log = np.median(logs, axis=1)
            marker_log = np.minimum(median_log - 0.9, -2.3)
            v['scale_0'][marker_idx] = marker_log
            v['scale_1'][marker_idx] = marker_log
            v['scale_2'][marker_idx] = marker_log
    line_sigma, spacing = _root_path_render_scale(ply, xyz)
    marker_positions = []
    marker_colors = []
    marker_sigma = max(line_sigma * 1.85, 1e-10)
    for node, users in node_users.items():
        node = int(node)
        if len(users) >= 2:
            color = white
        else:
            color = palette[next(iter(users))]
        marker_positions.append(xyz[node].astype(np.float64))
        marker_colors.append(color)
    marker_positions = np.asarray(marker_positions, dtype=np.float64) if marker_positions else np.empty((0, 3), dtype=np.float64)
    marker_colors = np.asarray(marker_colors, dtype=np.float64) if marker_colors else np.empty((0, 3), dtype=np.float64)
    line_positions = []
    line_colors = []
    shared_edge_count = 0
    for edge, users in edge_users.items():
        a, b = edge
        pa = xyz[a].astype(np.float64)
        pb = xyz[b].astype(np.float64)
        delta = pb - pa
        length = float(np.linalg.norm(delta))
        if length <= 1e-12:
            continue
        if len(users) >= 2:
            color = white
            shared_edge_count += 1
        else:
            color = palette[next(iter(users))]
        steps = max(1, int(np.ceil(length / spacing)))
        steps = min(steps, 256)
        t = (np.arange(steps, dtype=np.float64) + 0.5) / float(steps)
        pos = pa[None, :] + t[:, None] * delta[None, :]
        line_positions.append(pos)
        line_colors.append(np.tile(color[None, :], (steps, 1)))
    if line_positions:
        line_positions = np.concatenate(line_positions, axis=0)
        line_colors = np.concatenate(line_colors, axis=0)
        line_rows = _make_path_gaussian_rows(v, line_positions, line_colors, line_sigma)
    else:
        line_rows = np.empty((0,), dtype=v.dtype)
    section_positions = []
    section_colors = []
    section_sigma = max(line_sigma * 1.2, 1e-10)
    section_spacing = max(spacing * 0.65, 1e-10)
    for section in cross_sections:
        path_id = int(section['path_id'])
        a = int(section['endpoint_a'])
        b = int(section['endpoint_b'])
        if not (0 <= path_id < len(palette) and 0 <= a < n and (0 <= b < n) and (a != b)):
            continue
        pa = xyz[a].astype(np.float64)
        pb = xyz[b].astype(np.float64)
        delta = pb - pa
        length = float(np.linalg.norm(delta))
        if length <= 1e-12:
            continue
        color = white if bool(section.get('shared', False)) else palette[path_id]
        steps = max(2, int(np.ceil(length / section_spacing)))
        steps = min(steps, 256)
        t = (np.arange(steps, dtype=np.float64) + 0.5) / float(steps)
        pos = pa[None, :] + t[:, None] * delta[None, :]
        section_positions.append(pos)
        section_colors.append(np.tile(color[None, :], (steps, 1)))
    if section_positions:
        section_positions = np.concatenate(section_positions, axis=0)
        section_colors = np.concatenate(section_colors, axis=0)
        section_rows = _make_path_gaussian_rows(v, section_positions, section_colors, section_sigma)
    else:
        section_rows = np.empty((0,), dtype=v.dtype)
    marker_rows = _make_point_marker_rows(v, marker_positions, marker_colors, marker_sigma)
    out_vertex = np.concatenate([v, marker_rows, line_rows, section_rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(out_vertex, 'vertex')], text=False).write(str(path))
    print('      root_path.ply =', path)
    print('      Apex->Root paths =', len(paths))
    print('      unique path colors =', len(paths))
    print('      shared edges (white) =', shared_edge_count)
    print('      explicit path nodes =', len(marker_rows))
    print('      synthetic path-line Gaussians =', len(line_rows))
    print('      path-node cross-section lines =', len(cross_sections))
    print('      synthetic cross-section Gaussians =', len(section_rows))
    print('      path line sigma=%.6f, spacing=%.6f' % (line_sigma, spacing))
    print('      path-node marker sigma=%.6f' % (marker_sigma,))
    print('      cross-section sigma=%.6f, spacing=%.6f' % (section_sigma, section_spacing))
    print('      path legend: each Apex path=different color, merged/shared path=white')
    print('      cross-section mode =', 'FULL candidate-set maximum span' if full_cross_section_span else 'path point -> farthest edge candidate')
    print('      cross-section legend: one line per valid path node; same rendered path color')

def first_junctions_per_cluster(clusters, n):
    """
    Return the first path-intersection node (or None) for every cluster.

    Here "intersection" means the FIRST Apex->Root path Gaussian node that is
    also used by at least one other Apex->Root path. In other words, this is
    the exact shared Gaussian where two (or more) Apex paths first merge.

    The historical variable/function name "Junction" is retained because the
    rest of the Base/leaf code already uses that terminology.
    """
    paths = []
    for c in clusters:
        p = [int(x) for x in c.get('path', []) if 0 <= int(x) < n]
        paths.append(p)
    node_users = {}
    for path_id, p in enumerate(paths):
        for node in p:
            node_users.setdefault(int(node), set()).add(path_id)
    junctions = []
    for path_id, p in enumerate(paths):
        junction = None
        for node in p[1:]:
            if len(node_users.get(int(node), set())) >= 2:
                junction = int(node)
                break
        junctions.append(junction)
    return junctions

def _path_position(path, node_idx):
    """
    Return the discrete Apex->Root path position of node_idx.
    None means the node is not on this path.
    """
    if node_idx is None:
        return None
    try:
        return path.index(int(node_idx))
    except ValueError:
        return None

def extract_apex_to_nearest_boundary_regions(xyz, clusters, solver, Dr, tolerance_percentage=0.02):
    """
    For each Apex, use whichever appears first on its Apex->Root path:
    Base or Junction. Then keep the Apex-side Gaussian region using Da + Dr.
    """
    xyz = np.asarray(xyz)
    Dr = np.asarray(Dr, dtype=np.float64)
    n = len(xyz)
    junctions = first_junctions_per_cluster(clusters, n)
    dr_range = float(np.nanmax(Dr) - np.nanmin(Dr))
    dr_floor = max(dr_range * 0.0001, 1e-08)
    instances = []
    for leaf_id, (cluster, junction) in enumerate(zip(clusters, junctions), start=1):
        path = [int(x) for x in cluster.get('path', []) if 0 <= int(x) < n]
        if not path:
            instances.append(np.empty(0, dtype=np.int64))
            print('      leaf_%03d: empty path' % leaf_id)
            continue
        tip = int(cluster.get('selected_tip', path[0]))
        if not 0 <= tip < n:
            tip = int(path[0])
        base = cluster.get('base_idx', None)
        base = int(base) if base is not None and 0 <= int(base) < n else None
        junction = int(junction) if junction is not None and 0 <= int(junction) < n else None
        choices = []
        base_pos = _path_position(path, base)
        junction_pos = _path_position(path, junction)
        if base_pos is not None:
            choices.append((base_pos, 'base', base))
        if junction_pos is not None:
            choices.append((junction_pos, 'junction', junction))
        if choices:
            boundary_pos, boundary_type, boundary = min(choices, key=lambda x: (x[0], 0 if x[1] == 'base' else 1))
        else:
            boundary_pos = len(path) - 1
            boundary_type = 'path_end'
            boundary = int(path[-1])
        Da = np.asarray(solver.compute_distance(tip), dtype=np.float64)
        da_b = float(Da[boundary])
        dr_b = float(Dr[boundary])
        da_range = float(np.nanmax(Da) - np.nanmin(Da))
        da_tol = max(abs(da_b) * float(tolerance_percentage), da_range * 0.0001, 1e-08)
        dr_tol = max(abs(dr_b) * float(tolerance_percentage), dr_floor)
        keep = np.flatnonzero((Da <= da_b + da_tol) & (Dr >= dr_b - dr_tol)).astype(np.int64)
        if tip not in keep:
            keep = np.unique(np.concatenate([keep, np.asarray([tip], dtype=np.int64)]))
        instances.append(keep)
        print('      leaf_%03d: choose %s=%d (path pos=%d), points=%d' % (leaf_id, boundary_type, boundary, boundary_pos, len(keep)))
    return instances
CONTACT_MAX_K = 64
CONTACT_SCALE_K = 8
CONTACT_NORMAL_K = 16
CONTACT_COMMON_K = 16
CONTACT_DISTANCE_ALPHA = 2.8
CONTACT_TANGENT_THRESHOLD = 0.55
CONTACT_PLANARITY_MIN = 0.1
CONTACT_MIN_COMMON = 2
CONTACT_MIN_JACCARD = 0.08

# ------------------------------------------------------------------
# Conservative cross-organ edge veto.
#
# It is applied only AFTER the ordinary Clean-Graph filters.  We do not
# globally tighten KNN / distance / tangent thresholds, because that can
# destroy true leaf->petiole connectivity.
#
# A candidate edge is suspicious when several independent cues agree:
#   A) edge is long relative to local Gaussian spacing;
#   B) endpoint local geometry differs strongly (planarity mismatch);
#   C) the interior of the segment has poor third-point support;
#   D) common neighbors do not geometrically support the segment.
#
# Connectivity safeguard:
# rejected edges are selectively restored when needed to preserve the
# connected-component structure that existed before this veto.
# ------------------------------------------------------------------
CROSS_EDGE_RATIO_MIN = 1.70
CROSS_PLANARITY_DIFF_MIN = 0.20
CROSS_BRIDGE_RATIO_MAX = 0.90
CROSS_COMMON_SEGMENT_RATIO_MAX = 1.00
CROSS_MIN_SUSPICIOUS = 2
CROSS_BRIDGE_QUERY_K = 8

def _effective_contact_k(path_k, n):
    if n <= 1:
        return 1
    if path_k is None:
        k = CONTACT_MAX_K
    else:
        k = int(path_k)
    k = max(4, min(k, CONTACT_MAX_K, n - 1))
    return k

def _quat_from_x_axis(direction):
    """Quaternion (w,x,y,z) rotating local +X onto direction."""
    u = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(u))
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    u = u / norm
    dot = float(np.clip(u[0], -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if dot < -1.0 + 1e-10:
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    cross = np.cross(np.array([1.0, 0.0, 0.0], dtype=np.float64), u)
    q = np.array([1.0 + dot, cross[0], cross[1], cross[2]], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return q

def _make_colored_edge_rows(template_vertex, xyz, edges, colors, thickness, opacity_value):
    """
    Render each edge as one elongated anisotropic Gaussian.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    colors = np.asarray(colors, dtype=np.float64)
    if len(edges) == 0:
        return np.empty((0,), dtype=template_vertex.dtype)
    if len(colors) != len(edges):
        raise ValueError('colors and edges must have the same length')
    rows = np.zeros(len(edges), dtype=template_vertex.dtype)
    names = rows.dtype.names or ()
    a = xyz[edges[:, 0]]
    b = xyz[edges[:, 1]]
    mid = 0.5 * (a + b)
    vec = b - a
    length = np.linalg.norm(vec, axis=1)
    rows['x'] = mid[:, 0]
    rows['y'] = mid[:, 1]
    rows['z'] = mid[:, 2]
    if all((k in names for k in ('nx', 'ny', 'nz'))):
        rows['nx'] = 0
        rows['ny'] = 0
        rows['nz'] = 0
    if all((k in names for k in ('f_dc_0', 'f_dc_1', 'f_dc_2'))):
        fdc = (np.clip(colors, 0.0, 1.0) - 0.5) / 0.28209479177387814
        rows['f_dc_0'] = fdc[:, 0]
        rows['f_dc_1'] = fdc[:, 1]
        rows['f_dc_2'] = fdc[:, 2]
    for name in names:
        if name.startswith('f_rest_'):
            rows[name] = 0
    if 'opacity' in names:
        rows['opacity'] = float(opacity_value)
    if all((k in names for k in ('scale_0', 'scale_1', 'scale_2'))):
        long_sigma = np.maximum(length / 2.6, thickness)
        transverse = np.full(len(edges), max(float(thickness), 1e-10))
        rows['scale_0'] = np.log(np.maximum(long_sigma, 1e-10))
        rows['scale_1'] = np.log(transverse)
        rows['scale_2'] = np.log(transverse)
    if all((k in names for k in ('rot_0', 'rot_1', 'rot_2', 'rot_3'))):
        quats = np.asarray([_quat_from_x_axis(v) for v in vec], dtype=np.float64)
        rows['rot_0'] = quats[:, 0]
        rows['rot_1'] = quats[:, 1]
        rows['rot_2'] = quats[:, 2]
        rows['rot_3'] = quats[:, 3]
    if 'filter_3D' in names:
        rows['filter_3D'] = 0.0
    return rows

def _pointcloud_heat_graph_edges(xyz):
    """
    Build the point-cloud local triangulation used by potpourri3d and convert
    all local triangles into one unique undirected edge set.
    """
    import potpourri3d as pp3d
    xyz = np.asarray(xyz, dtype=np.float64)
    local_triangulation = pp3d.PointCloudLocalTriangulation(xyz)
    local_tris = np.asarray(local_triangulation.get_local_triangulation(), dtype=np.int64)
    edges = set()
    valid_triangles = 0
    for tri in local_tris.reshape(-1, 3):
        a, b, c = map(int, tri)
        if a < 0 or b < 0 or c < 0:
            continue
        if a == b or b == c or a == c:
            continue
        valid_triangles += 1
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = (v, u)
            edges.add((u, v))
    if edges:
        edge_array = np.asarray(sorted(edges), dtype=np.int64)
    else:
        edge_array = np.empty((0, 2), dtype=np.int64)
    return (edge_array, valid_triangles, local_tris.shape)

def _filter_graph_edges(xyz, edges, path_k):
    xyz = np.asarray(xyz, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    n = len(xyz)
    if n < 2 or len(edges) == 0:
        return (np.empty((0, 2), dtype=np.int64), {'graph_k': 0, 'tri_edges': int(len(edges)), 'mutual': 0, 'distance': 0, 'tangent': 0, 'continuity': 0, 'veto_flagged': 0, 'veto_restored': 0, 'veto_removed': 0, 'clean': 0, 'spacing_median': 0.0})
    graph_k = _effective_contact_k(path_k, n)
    tree = cKDTree(xyz)
    d_all, i_all = tree.query(xyz, k=min(graph_k + 1, n))
    if np.ndim(i_all) == 1:
        i_all = i_all[:, None]
        d_all = d_all[:, None]
    nbrs, nbr_dists = ([], [])
    for i in range(n):
        ids = np.asarray(i_all[i], dtype=np.int64).reshape(-1)
        ds = np.asarray(d_all[i], dtype=np.float64).reshape(-1)
        keep = ids != i
        nbrs.append(ids[keep][:graph_k])
        nbr_dists.append(ds[keep][:graph_k])
    nbr_sets = [set(map(int, ids)) for ids in nbrs]
    local_spacing = np.full(n, np.nan, dtype=np.float64)
    for i, ds in enumerate(nbr_dists):
        take = ds[:min(CONTACT_SCALE_K, len(ds))]
        take = take[np.isfinite(take) & (take > 1e-12)]
        if len(take):
            local_spacing[i] = float(np.median(take))
    finite = local_spacing[np.isfinite(local_spacing)]
    global_spacing = float(np.median(finite)) if len(finite) else 1.0
    local_spacing[~np.isfinite(local_spacing)] = global_spacing
    local_spacing = np.maximum(local_spacing, 1e-10)
    normal_k = min(CONTACT_NORMAL_K, graph_k)
    normals = np.zeros((n, 3), dtype=np.float64)
    planarity = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ids = nbrs[i][:normal_k]
        if len(ids) < 3:
            continue
        pts = np.vstack([xyz[i][None, :], xyz[ids]])
        centered = pts - np.mean(pts, axis=0, keepdims=True)
        cov = centered.T @ centered / max(len(centered) - 1, 1)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        order = np.argsort(eigvals)
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]
        normals[i] = eigvecs[:, 0]
        planarity[i] = float(max(eigvals[1] - eigvals[0], 0.0) / max(float(eigvals[2]), 1e-12))
    mutual_edges = [(int(u), int(v)) for u, v in edges if int(v) in nbr_sets[int(u)] and int(u) in nbr_sets[int(v)]]
    distance_edges = []
    for u, v in mutual_edges:
        d = float(np.linalg.norm(xyz[v] - xyz[u]))
        limit = CONTACT_DISTANCE_ALPHA * float(np.sqrt(local_spacing[u] * local_spacing[v]))
        if d <= limit:
            distance_edges.append((u, v))
    tangent_edges = []
    for u, v in distance_edges:
        vec = xyz[v] - xyz[u]
        length = float(np.linalg.norm(vec))
        if length <= 1e-12:
            continue
        direction = vec / length
        good_u = planarity[u] < CONTACT_PLANARITY_MIN or abs(float(np.dot(direction, normals[u]))) <= CONTACT_TANGENT_THRESHOLD
        good_v = planarity[v] < CONTACT_PLANARITY_MIN or abs(float(np.dot(direction, normals[v]))) <= CONTACT_TANGENT_THRESHOLD
        if good_u and good_v:
            tangent_edges.append((u, v))
    common_k = min(CONTACT_COMMON_K, graph_k)
    small_sets = [set(map(int, ids[:common_k])) for ids in nbrs]

    continuity_edges = []
    for u, v in tangent_edges:
        nu, nv = (small_sets[u], small_sets[v])
        common = len(nu & nv)
        union = len(nu | nv)
        jaccard = float(common / union) if union else 0.0

        if (
            common >= CONTACT_MIN_COMMON
            and jaccard >= CONTACT_MIN_JACCARD
        ):
            continuity_edges.append((u, v))

    # --------------------------------------------------------------
    # Stage 5: conservative cross-organ edge veto.
    #
    # The strongest cue is "bridge support": sample 25%, 50%, 75%
    # along u-v and ask whether OTHER Gaussians (excluding endpoints)
    # occur near the segment.  A false leaf-edge -> branch shortcut
    # often crosses empty space and therefore has weak support.
    # --------------------------------------------------------------
    veto_flagged = 0
    veto_restored = 0
    veto_removed = 0

    if continuity_edges:
        ce = np.asarray(
            continuity_edges,
            dtype=np.int64,
        )

        u_idx = ce[:, 0]
        v_idx = ce[:, 1]

        a = xyz[u_idx]
        b = xyz[v_idx]
        vec = b - a

        edge_len = np.linalg.norm(
            vec,
            axis=1,
        )

        local_scale = np.sqrt(
            local_spacing[u_idx]
            * local_spacing[v_idx]
        )
        local_scale = np.maximum(
            local_scale,
            1e-10,
        )

        # Cue A: long relative to local point spacing.
        edge_ratio = (
            edge_len
            / local_scale
        )
        flag_long = (
            edge_ratio
            >= CROSS_EDGE_RATIO_MIN
        )

        # Cue B: local geometry mismatch.
        planarity_diff = np.abs(
            planarity[u_idx]
            - planarity[v_idx]
        )
        flag_geometry = (
            planarity_diff
            >= CROSS_PLANARITY_DIFF_MIN
        )

        # Cue C: segment-interior support.
        #
        # Query several nearest points at each sample location and explicitly
        # ignore u/v, otherwise the endpoints themselves would falsely look
        # like "support".
        ts = np.asarray(
            [0.25, 0.50, 0.75],
            dtype=np.float64,
        )
        samples = (
            a[:, None, :]
            + ts[None, :, None]
            * vec[:, None, :]
        )

        sample_flat = samples.reshape(
            -1,
            3,
        )

        qk = min(
            max(3, CROSS_BRIDGE_QUERY_K),
            n,
        )
        support_d, support_i = tree.query(
            sample_flat,
            k=qk,
        )

        support_d = np.asarray(
            support_d,
            dtype=np.float64,
        )
        support_i = np.asarray(
            support_i,
            dtype=np.int64,
        )

        if support_d.ndim == 1:
            support_d = support_d[:, None]
            support_i = support_i[:, None]

        support_d = support_d.reshape(
            len(ce),
            len(ts),
            -1,
        )
        support_i = support_i.reshape(
            len(ce),
            len(ts),
            -1,
        )

        endpoint_mask = (
            (support_i == u_idx[:, None, None])
            | (support_i == v_idx[:, None, None])
        )

        support_d = support_d.copy()
        support_d[endpoint_mask] = np.inf

        nearest_third = np.min(
            support_d,
            axis=2,
        )

        # Worst-supported interior sample controls the cue.
        bridge_distance = np.max(
            nearest_third,
            axis=1,
        )
        bridge_ratio = (
            bridge_distance
            / local_scale
        )
        flag_bridge = (
            bridge_ratio
            > CROSS_BRIDGE_RATIO_MAX
        )

        # Cue D: common neighbors must geometrically support the segment,
        # not merely exist somewhere in both KNN sets.
        flag_common_geometry = np.ones(
            len(ce),
            dtype=bool,
        )

        for ei, (u, v) in enumerate(ce):
            common_ids = (
                small_sets[int(u)]
                & small_sets[int(v)]
            )

            if not common_ids:
                continue

            p0 = xyz[int(u)]
            segment = (
                xyz[int(v)]
                - p0
            )
            seg2 = float(
                np.dot(
                    segment,
                    segment,
                )
            )

            if seg2 <= 1e-20:
                flag_common_geometry[ei] = False
                continue

            common_arr = np.asarray(
                sorted(common_ids),
                dtype=np.int64,
            )
            rel = (
                xyz[common_arr]
                - p0[None, :]
            )

            tproj = (
                rel @ segment
                / seg2
            )
            tclip = np.clip(
                tproj,
                0.0,
                1.0,
            )
            closest = (
                p0[None, :]
                + tclip[:, None]
                * segment[None, :]
            )
            dseg = np.linalg.norm(
                xyz[common_arr]
                - closest,
                axis=1,
            )

            geometrically_supported = bool(
                np.any(
                    dseg
                    <= (
                        CROSS_COMMON_SEGMENT_RATIO_MAX
                        * local_scale[ei]
                    )
                )
            )

            flag_common_geometry[ei] = (
                not geometrically_supported
            )

        suspicious_count = (
            flag_long.astype(np.int8)
            + flag_geometry.astype(np.int8)
            + flag_bridge.astype(np.int8)
            + flag_common_geometry.astype(np.int8)
        )

        # Conservative decision:
        #   - at least two cues must agree;
        #   - normally one of them must be the empty-space bridge cue;
        #   - otherwise require >=3 independent abnormal cues.
        reject_mask = (
            (
                suspicious_count
                >= CROSS_MIN_SUSPICIOUS
            )
            & (
                flag_bridge
                | (
                    suspicious_count
                    >= 3
                )
            )
        )

        veto_flagged = int(
            np.count_nonzero(
                reject_mask
            )
        )

        kept_edges = [
            tuple(map(int, e))
            for e in ce[~reject_mask]
        ]

        rejected_idx = np.flatnonzero(
            reject_mask
        )

        # ----------------------------------------------------------
        # Connectivity safeguard.
        #
        # Never allow this new veto to split a connected component that was
        # connected after the ordinary continuity filter.  Start with the
        # kept edges, then restore the least-suspicious rejected edges only
        # when they reconnect two currently separated components.
        # ----------------------------------------------------------
        parent = np.arange(
            n,
            dtype=np.int64,
        )
        rank = np.zeros(
            n,
            dtype=np.int8,
        )

        def uf_find(x):
            x = int(x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = int(parent[x])
            return x

        def uf_union(x, y):
            rx = uf_find(x)
            ry = uf_find(y)

            if rx == ry:
                return False

            if rank[rx] < rank[ry]:
                rx, ry = ry, rx

            parent[ry] = rx

            if rank[rx] == rank[ry]:
                rank[rx] += 1

            return True

        for u, v in kept_edges:
            uf_union(u, v)

        restored_edges = []

        if len(rejected_idx):
            # Restore least suspicious bridges first.
            restore_order = sorted(
                rejected_idx.tolist(),
                key=lambda ei: (
                    int(suspicious_count[ei]),
                    float(bridge_ratio[ei]),
                    float(edge_ratio[ei]),
                    float(planarity_diff[ei]),
                ),
            )

            for ei in restore_order:
                u = int(ce[ei, 0])
                v = int(ce[ei, 1])

                if uf_union(u, v):
                    restored_edges.append(
                        (u, v)
                    )

        veto_restored = int(
            len(restored_edges)
        )
        veto_removed = int(
            veto_flagged
            - veto_restored
        )

        final_edges = (
            kept_edges
            + restored_edges
        )

        clean_edges = np.asarray(
            sorted(
                set(final_edges)
            ),
            dtype=np.int64,
        ) if final_edges else np.empty(
            (0, 2),
            dtype=np.int64,
        )

    else:
        clean_edges = np.empty(
            (0, 2),
            dtype=np.int64,
        )

    return (
        clean_edges,
        {
            'graph_k': int(graph_k),
            'tri_edges': int(len(edges)),
            'mutual': int(len(mutual_edges)),
            'distance': int(len(distance_edges)),
            'tangent': int(len(tangent_edges)),
            'continuity': int(len(continuity_edges)),
            'veto_flagged': int(veto_flagged),
            'veto_restored': int(veto_restored),
            'veto_removed': int(veto_removed),
            'clean': int(len(clean_edges)),
            'spacing_median': float(global_spacing),
        },
    )

def _edges_to_adjacency(n, edges):
    """
    Convert an undirected edge list to sorted per-vertex adjacency arrays.
    """
    buckets = [[] for _ in range(int(n))]
    for u, v in np.asarray(edges, dtype=np.int64):
        u = int(u)
        v = int(v)
        if u == v:
            continue
        buckets[u].append(v)
        buckets[v].append(u)
    adj = []
    for b in buckets:
        if b:
            adj.append(np.asarray(sorted(set(b)), dtype=np.int64))
        else:
            adj.append(np.empty(0, dtype=np.int64))
    return adj

def _reachable_nodes(clean_adj, sources):
    seen = set((int(s) for s in sources))
    stack = list(seen)
    while stack:
        u = stack.pop()
        for v in clean_adj[u]:
            v = int(v)
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen

def _clean_graph_root_to_apex_paths(xyz, clean_adj, root_idx, apex_indices):
    """
    Compute pure shortest paths from the EXACT selected Root Gaussian to each
    detected Apex on the filtered Clean Graph.

    Important:
      - source = root_idx (single exact Root), NOT the whole Root-source region;
      - targets = detected Apex points only;
      - edge weight = Euclidean length of the Clean-Graph edge;
      - no Temperature, Da, Dr, turn penalty, or cross-section constraint;
      - Dijkstra stops once all reachable Apex targets have been settled.

    The function returns only Root->Apex paths. It does not create or export
    paths from Root to every Gaussian.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    root_idx = int(root_idx)
    targets = sorted({int(a) for a in apex_indices if 0 <= int(a) < n and int(a) != root_idx})
    if not targets:
        return ([], [], [])
    target_set = set(targets)
    remaining = set(targets)
    dist = np.full(n, np.inf, dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int64)
    settled = np.zeros(n, dtype=bool)
    dist[root_idx] = 0.0
    heap = [(0.0, root_idx)]
    while heap and remaining:
        du, u = heapq.heappop(heap)
        u = int(u)
        if settled[u]:
            continue
        if du > dist[u] + 1e-12:
            continue
        settled[u] = True
        if u in remaining:
            remaining.remove(u)
            if not remaining:
                break
        nbr = clean_adj[u]
        if nbr is None:
            continue
        for v in np.asarray(nbr, dtype=np.int64):
            v = int(v)
            if settled[v]:
                continue
            edge_len = float(np.linalg.norm(xyz[v] - xyz[u]))
            if not np.isfinite(edge_len) or edge_len <= 1e-12:
                continue
            nd = du + edge_len
            if nd + 1e-12 < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    paths = []
    reached = []
    unreachable = []
    for apex_idx in targets:
        if not np.isfinite(dist[apex_idx]):
            unreachable.append(apex_idx)
            continue
        path = []
        cur = int(apex_idx)
        while cur >= 0:
            path.append(cur)
            if cur == root_idx:
                break
            cur = int(prev[cur])
        if not path or path[-1] != root_idx:
            unreachable.append(apex_idx)
            continue
        path.reverse()
        paths.append(path)
        reached.append(apex_idx)
    return (paths, reached, unreachable)

def save_root_apex(ply, xyz, root_idx, tips, clean_adj, graph_edges, path):
    """
    Save root_apex.ply:

      gray   = filtered Clean Graph
      colored= PURE shortest paths from exact Root -> each detected Apex
      white  = path segment shared by >=2 Root->Apex shortest paths
      magenta= detected Apex
      red    = exact selected Root

    These paths are deliberately independent of the current Apex->Root
    algorithm so the two can be compared visually.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    apex_idx = flatten(tips, n)
    paths, reached, unreachable = _clean_graph_root_to_apex_paths(xyz, clean_adj, root_idx, apex_idx)
    graph_edges = np.asarray(graph_edges, dtype=np.int64)
    v = ply['vertex'].data.copy()
    names = v.dtype.names or ()
    if all((k in names for k in ('f_dc_0', 'f_dc_1', 'f_dc_2'))):
        bg = np.full((n, 3), 0.045, dtype=np.float64)
        fdc = (bg - 0.5) / 0.28209479177387814
        v['f_dc_0'] = fdc[:, 0]
        v['f_dc_1'] = fdc[:, 1]
        v['f_dc_2'] = fdc[:, 2]
        for name in names:
            if name.startswith('f_rest_'):
                v[name] = 0
    if 'opacity' in names:
        v['opacity'] = -5.9
    base_sigma, _ = _root_path_render_scale(ply, xyz)
    graph_thickness = max(base_sigma * 0.12, 1e-10)
    path_thickness = max(base_sigma * 0.34, 1e-10)
    graph_colors = np.full((len(graph_edges), 3), 0.33, dtype=np.float64)
    graph_rows = _make_colored_edge_rows(v, xyz, graph_edges, graph_colors, graph_thickness, opacity_value=2.8)
    node_users = {}
    edge_users = {}
    for path_id, p in enumerate(paths):
        for node in p:
            node_users.setdefault(int(node), set()).add(path_id)
        for a, b in zip(p[:-1], p[1:]):
            a = int(a)
            b = int(b)
            if a == b:
                continue
            edge = (a, b) if a < b else (b, a)
            edge_users.setdefault(edge, set()).add(path_id)
    palette = path_palette(max(len(paths), 1))
    white = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    path_edge_list = []
    path_edge_colors = []
    shared_edge_count = 0
    for edge, users in edge_users.items():
        path_edge_list.append(edge)
        if len(users) >= 2:
            path_edge_colors.append(white)
            shared_edge_count += 1
        else:
            path_edge_colors.append(palette[next(iter(users))])
    if path_edge_list:
        path_edge_arr = np.asarray(path_edge_list, dtype=np.int64)
        path_edge_colors = np.asarray(path_edge_colors, dtype=np.float64)
    else:
        path_edge_arr = np.empty((0, 2), dtype=np.int64)
        path_edge_colors = np.empty((0, 3), dtype=np.float64)
    path_rows = _make_colored_edge_rows(v, xyz, path_edge_arr, path_edge_colors, path_thickness, opacity_value=6.0)
    marker_positions = []
    marker_colors = []
    for node, users in node_users.items():
        if int(node) == int(root_idx):
            continue
        if int(node) in set(map(int, apex_idx)):
            continue
        if len(users) >= 2:
            color = white
        else:
            color = palette[next(iter(users))]
        marker_positions.append(xyz[int(node)].astype(np.float64))
        marker_colors.append(color)
    marker_positions = np.asarray(marker_positions, dtype=np.float64) if marker_positions else np.empty((0, 3), dtype=np.float64)
    marker_colors = np.asarray(marker_colors, dtype=np.float64) if marker_colors else np.empty((0, 3), dtype=np.float64)
    marker_rows = _make_point_marker_rows(v, marker_positions, marker_colors, max(base_sigma * 0.9, 1e-10))
    key_positions = []
    key_colors = []
    if len(apex_idx):
        key_positions.append(xyz[apex_idx])
        key_colors.append(np.tile(np.array([[1.0, 0.0, 1.0]], dtype=np.float64), (len(apex_idx), 1)))
    key_positions.append(xyz[[int(root_idx)]])
    key_colors.append(np.array([[1.0, 0.05, 0.05]], dtype=np.float64))
    key_rows = _make_point_marker_rows(v, np.concatenate(key_positions, axis=0), np.concatenate(key_colors, axis=0), max(base_sigma * 1.65, 1e-10))
    out_vertex = np.concatenate([v, graph_rows, path_rows, marker_rows, key_rows])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(out_vertex, 'vertex')], text=False).write(str(path))
    print('      root_apex.ply =', path)
    print('      exact Root index          =', int(root_idx))
    print('      detected Apex count       =', len(apex_idx))
    print('      Root->Apex paths reached  =', len(reached))
    print('      unreachable Apex          =', len(unreachable))
    print('      shortest-path edges       =', len(path_edge_arr))
    print('      shared shortest edges     =', shared_edge_count)
    if unreachable:
        print('      unreachable Apex examples =', unreachable[:8])
    print('      legend: gray=Clean Graph, colored=pure Root->Apex shortest paths, white=shared path, magenta=Apex, red=exact Root')

def save_path_contact(ply, xyz, root_idx, graph_edges, graph_stats, triangle_count, tri_shape, raw_edge_count, path):
    """
    Layer 1:
    Save path_contact.ply as the filtered local-triangulation graph.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    edges = np.asarray(graph_edges, dtype=np.int64)
    stats = graph_stats
    base_sigma, _ = _root_path_render_scale(ply, xyz)
    edge_thickness = max(base_sigma * 0.18, 1e-10)
    v = ply['vertex'].data.copy()
    names = v.dtype.names or ()
    if all((k in names for k in ('f_dc_0', 'f_dc_1', 'f_dc_2'))):
        bg = np.full((n, 3), 0.055, dtype=np.float64)
        fdc = (bg - 0.5) / 0.28209479177387814
        v['f_dc_0'] = fdc[:, 0]
        v['f_dc_1'] = fdc[:, 1]
        v['f_dc_2'] = fdc[:, 2]
        for name in names:
            if name.startswith('f_rest_'):
                v[name] = 0
    if 'opacity' in names:
        v['opacity'] = -5.8
    edge_colors = np.full((len(edges), 3), 0.94, dtype=np.float64)
    edge_rows = _make_colored_edge_rows(v, xyz, edges, edge_colors, edge_thickness, opacity_value=5.5)
    node_colors = np.tile(np.array([[0.38, 0.55, 0.7]], dtype=np.float64), (n, 1))
    node_rows = _make_point_marker_rows(v, xyz, node_colors, max(base_sigma * 0.16, 1e-10))
    root_rows = _make_point_marker_rows(v, xyz[[int(root_idx)]], np.array([[1.0, 0.03, 0.03]], dtype=np.float64), max(base_sigma * 2.3, 1e-10))
    out_vertex = np.concatenate([v, edge_rows, node_rows, root_rows])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(out_vertex, 'vertex')], text=False).write(str(path))
    print('      path_contact.ply =', path)
    print('      graph vertices          =', n)
    print('      local triangulation     =', tri_shape)
    print('      valid local triangles   =', triangle_count)
    print('      raw graph edges         =', int(raw_edge_count))
    print('      after mutual-KNN        =', stats['mutual'])
    print('      after distance filter   =', stats['distance'])
    print('      after tangent filter    =', stats['tangent'])
    print('      after continuity filter =', stats['continuity'])
    print('      cross-organ flagged     =', stats['veto_flagged'])
    print('      connectivity restored   =', stats['veto_restored'])
    print('      cross-organ removed     =', stats['veto_removed'])
    print('      final Clean Graph edges =', stats['clean'])
    print('      rendered edge Gaussians =', len(edge_rows))

def save_apex_root(ply, xyz, root_idx, tips, bases, junctions, clusters, graph_edges, graph_stats, path):
    """
    Layer 2:
    Save Apex_root.ply showing the same filtered graph in faint gray and the
    ACTUAL selected Apex->Root paths overlaid in bright path colors.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    paths = _collect_apex_root_paths(clusters, n)
    if not paths:
        print('      Apex_root.ply skipped: no Apex->Root paths found')
        return
    graph_edges = np.asarray(graph_edges, dtype=np.int64)
    stats = graph_stats
    graph_edge_set = {(int(min(u, v)), int(max(u, v))) for u, v in graph_edges}
    node_users = {}
    edge_users = {}
    for path_id, p in enumerate(paths):
        for node in p:
            node_users.setdefault(int(node), set()).add(path_id)
        for a, b in zip(p[:-1], p[1:]):
            a = int(a)
            b = int(b)
            if a == b:
                continue
            edge = (a, b) if a < b else (b, a)
            edge_users.setdefault(edge, set()).add(path_id)
    outside_graph_edges = [edge for edge in edge_users if edge not in graph_edge_set]
    if outside_graph_edges:
        raise RuntimeError('Apex->Root contains %d edge(s) outside the clean graph. First examples: %s' % (len(outside_graph_edges), outside_graph_edges[:8]))
    palette = path_palette(len(paths))
    white = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    v = ply['vertex'].data.copy()
    names = v.dtype.names or ()
    if all((k in names for k in ('f_dc_0', 'f_dc_1', 'f_dc_2'))):
        bg = np.full((n, 3), 0.045, dtype=np.float64)
        fdc = (bg - 0.5) / 0.28209479177387814
        v['f_dc_0'] = fdc[:, 0]
        v['f_dc_1'] = fdc[:, 1]
        v['f_dc_2'] = fdc[:, 2]
        for name in names:
            if name.startswith('f_rest_'):
                v[name] = 0
    if 'opacity' in names:
        v['opacity'] = -5.9
    base_sigma, _ = _root_path_render_scale(ply, xyz)
    graph_thickness = max(base_sigma * 0.12, 1e-10)
    path_thickness = max(base_sigma * 0.3, 1e-10)
    graph_colors = np.full((len(graph_edges), 3), 0.33, dtype=np.float64)
    graph_rows = _make_colored_edge_rows(v, xyz, graph_edges, graph_colors, graph_thickness, opacity_value=2.8)
    path_edge_list = []
    path_edge_colors = []
    shared_edge_count = 0
    for edge, users in edge_users.items():
        path_edge_list.append(edge)
        if len(users) >= 2:
            path_edge_colors.append(white)
            shared_edge_count += 1
        else:
            path_edge_colors.append(palette[next(iter(users))])
    if path_edge_list:
        path_edge_arr = np.asarray(path_edge_list, dtype=np.int64)
        path_edge_colors = np.asarray(path_edge_colors, dtype=np.float64)
    else:
        path_edge_arr = np.empty((0, 2), dtype=np.int64)
        path_edge_colors = np.empty((0, 3), dtype=np.float64)
    path_rows = _make_colored_edge_rows(v, xyz, path_edge_arr, path_edge_colors, path_thickness, opacity_value=6.0)
    marker_positions = []
    marker_colors = []
    marker_sigma = max(base_sigma * 0.95, 1e-10)
    for node, users in node_users.items():
        if len(users) >= 2:
            color = white
        else:
            color = palette[next(iter(users))]
        marker_positions.append(xyz[int(node)].astype(np.float64))
        marker_colors.append(color)
    marker_positions = np.asarray(marker_positions, dtype=np.float64) if marker_positions else np.empty((0, 3), dtype=np.float64)
    marker_colors = np.asarray(marker_colors, dtype=np.float64) if marker_colors else np.empty((0, 3), dtype=np.float64)
    marker_rows = _make_point_marker_rows(v, marker_positions, marker_colors, marker_sigma)
    aidx = flatten(tips, n)
    bidx = flatten(bases, n)
    jidx = flatten(junctions, n)
    key_positions = []
    key_colors = []
    if len(aidx):
        key_positions.append(xyz[aidx])
        key_colors.append(np.tile(np.array([[1.0, 0.0, 1.0]], dtype=np.float64), (len(aidx), 1)))
    if len(bidx):
        key_positions.append(xyz[bidx])
        key_colors.append(np.tile(np.array([[0.05, 0.25, 1.0]], dtype=np.float64), (len(bidx), 1)))
    if len(jidx):
        key_positions.append(xyz[jidx])
        key_colors.append(np.tile(np.array([[1.0, 0.85, 0.0]], dtype=np.float64), (len(jidx), 1)))
    key_positions.append(xyz[[int(root_idx)]])
    key_colors.append(np.array([[1.0, 0.05, 0.05]], dtype=np.float64))
    key_rows = _make_point_marker_rows(v, np.concatenate(key_positions, axis=0), np.concatenate(key_colors, axis=0), max(base_sigma * 1.65, 1e-10))
    out_vertex = np.concatenate([v, graph_rows, path_rows, marker_rows, key_rows])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(out_vertex, 'vertex')], text=False).write(str(path))
    print('      Apex_root.ply =', path)
    print('      filtered graph edges    =', len(graph_edges))
    print('      selected path count     =', len(paths))
    print('      selected path edges     =', len(path_edge_arr))
    print('      shared path edges       =', shared_edge_count)
    print('      path edges outside graph=', len(outside_graph_edges))
    print('      graph filter final      =', stats['clean'])
    print('      legend: gray=Clean Graph, colored=turn-aware Dijkstra Apex->Root paths')

def clean_output(out):
    """Start clean; outputs: path_contact, root_apex, Apex_root, root_path, leaves, stem."""
    if out.exists():
        shutil.rmtree(str(out))
    out.mkdir(parents=True, exist_ok=True)

def main():
    args = args_parse()
    if args.tip_k is not None and args.tip_k < 2:
        raise ValueError('--tip-k must be >= 2')
    if args.path_k is not None and args.path_k < 2:
        raise ValueError('--path-k must be >= 2')
    inp = Path(args.input).resolve()
    core = Path(args.leaf_fit_root).resolve() / 'core'
    out = inp.parent / 'leaf_fit_segments'
    clean_output(out)
    print('[1/8] Read PLY')
    ply, xyz = read_ply(inp)
    n = len(xyz)
    gaussians = make_gaussians(xyz * float(args.geometry_scale))
    print('[2/8] Load LeafFit + select Root')
    auto = load_leaf_fit(core, args.tip_k, args.path_k, disable_apex_cluster=args.disable_apex_cluster, full_cross_section_span=args.full_cross_section_span)
    if args.pick_root:
        root_input = pick_root(xyz, ply, args.root_point_size)
        print('      root = visual selection:', root_input, xyz[root_input])
    else:
        root_input = int(np.argmin(xyz[:, 2]))
        print('      root = minimum-Z:', root_input, xyz[root_input])
    gaussians, root_idx, solver = auto.fix_plant_root_direction_legacy(gaussians, opacity_threshold=0.0, given_root_idx=int(root_input))
    root_idx = int(root_idx)
    dr0 = np.asarray(solver.compute_distance(root_idx), dtype=np.float64)
    sources = np.where(dr0 <= float(args.root_radius))[0].astype(np.int64)
    if len(sources) == 0:
        sources = np.asarray([root_idx], np.int64)
    Dr = solver.compute_distance_multisource(sources)
    print('[3/8] Detect Apex / Base / paths')
    ns = min(int(args.fps_points), n)
    sparse = fpsample.bucket_fps_kdline_sampling(gaussians.xyz, ns, h=7, start_idx=0).astype(np.int64)
    nn = NearestNeighbors(n_neighbors=1).fit(gaussians.xyz[sparse])
    _, mapping = nn.kneighbors(gaussians.xyz)
    mapping = mapping.flatten().astype(np.int64)
    tree = cKDTree(gaussians.xyz)
    print('      build clean path graph ...')
    raw_graph_edges, raw_triangle_count, raw_tri_shape = _pointcloud_heat_graph_edges(gaussians.xyz)
    clean_graph_edges, clean_graph_stats = _filter_graph_edges(gaussians.xyz, raw_graph_edges, args.path_k)
    if len(clean_graph_edges) == 0:
        raise RuntimeError('Clean graph has 0 edges; relax contact filters before path tracing.')
    clean_adj = _edges_to_adjacency(n, clean_graph_edges)
    auto.LEAFFIT_CLEAN_ADJ = clean_adj
    auto.LEAFFIT_CLEAN_EDGE_SET = {(int(min(u, v)), int(max(u, v))) for u, v in clean_graph_edges}
    auto.LEAFFIT_ROOT_SOURCE_SET = set(map(int, sources.tolist()))
    geodesic_root_distance = np.asarray(Dr, dtype=np.float64).copy()
    source_idx_arr = np.asarray(sources, dtype=np.int64)
    source_values = geodesic_root_distance[source_idx_arr]
    source_values = source_values[np.isfinite(source_values)]
    if len(source_values):
        geodesic_root_distance -= float(np.min(source_values))
    finite_gr_mask = np.isfinite(geodesic_root_distance)
    geodesic_root_distance[finite_gr_mask] = np.maximum(geodesic_root_distance[finite_gr_mask], 0.0)
    auto.LEAFFIT_ROOT_GRAPH_DISTANCE = geodesic_root_distance
    root_reachable = int(np.count_nonzero(np.isfinite(geodesic_root_distance)))
    root_unreachable = int(n - root_reachable)
    isolated_count = sum((1 for nbr in clean_adj if len(nbr) == 0))
    print('      raw graph edges         =', len(raw_graph_edges))
    print('      after mutual-KNN        =', clean_graph_stats['mutual'])
    print('      after distance filter   =', clean_graph_stats['distance'])
    print('      after tangent filter    =', clean_graph_stats['tangent'])
    print('      after continuity filter =', clean_graph_stats['continuity'])
    print('      cross-organ flagged     =', clean_graph_stats['veto_flagged'])
    print('      connectivity restored   =', clean_graph_stats['veto_restored'])
    print('      cross-organ removed     =', clean_graph_stats['veto_removed'])
    print('      final Clean Graph edges =', clean_graph_stats['clean'])
    print('      isolated vertices       =', isolated_count)
    print('      Geodesic-Gr finite      =', root_reachable, '/', n)
    print('      Geodesic-Gr nonfinite   =', root_unreachable)
    finite_gr_values = geodesic_root_distance[np.isfinite(geodesic_root_distance)]
    if len(finite_gr_values):
        print('      Geodesic-Gr range       = %.6f .. %.6f' % (float(np.min(finite_gr_values)), float(np.max(finite_gr_values))))
    print('      Apex->Root search       = CLEAN-GRAPH + SOFT GEODESIC-Gr + TURN')
    _, tips, bases, _, clusters, _ = auto.get_segment_mask(gaussians, sparse, mapping, solver, tree, sources.tolist(), 'geodesic_tip_graph', cached_root_distances=Dr, debug_vis=False)
    print('      detected Apex =', len(tips))
    print('      paths =', len(clusters))
    apex_idx_diag = flatten(tips, n)
    exact_root_reach = _reachable_nodes(clean_adj, [root_idx])
    root_region_reach = _reachable_nodes(clean_adj, sources.tolist())
    exact_unreachable = [int(a) for a in apex_idx_diag if int(a) not in exact_root_reach]
    region_unreachable = [int(a) for a in apex_idx_diag if int(a) not in root_region_reach]
    print('      Apex disconnected from exact Root =', len(exact_unreachable))
    print('      Apex disconnected from Root region=', len(region_unreachable))
    if region_unreachable:
        print('      disconnected Apex examples       =', region_unreachable[:8])
    root_source_set = set(map(int, sources.tolist()))
    reached_root_region = 0
    stopped_outside_root_region = 0
    for c in clusters:
        pth = [int(x) for x in c.get('path', [])]
        if pth and pth[-1] in root_source_set:
            reached_root_region += 1
        else:
            stopped_outside_root_region += 1
    print('      root source count          = %d' % len(root_source_set))
    print('      clean-path reached root region = %d/%d' % (reached_root_region, len(clusters)))
    print('      clean-path stopped outside root = %d' % stopped_outside_root_region)
    global_stats = getattr(auto, 'LEAFFIT_GLOBAL_PATH_STATS', [])
    if global_stats:
        reached = sum((1 for s in global_stats if s.get('status') in ('reached_root', 'already_in_root')))
        disconnected = sum((1 for s in global_stats if s.get('status') == 'clean_graph_disconnected'))
        root_search_failed = sum((1 for s in global_stats if s.get('status') == 'root_guided_search_failed'))
        mean_len = float(np.mean([s.get('path_len', 0) for s in global_stats]))
        mean_visited = float(np.mean([s.get('visited', 0) for s in global_stats]))
        mean_turn = float(np.mean([s.get('mean_turn_deg', 0.0) for s in global_stats]))
        max_turn = float(np.max([s.get('max_turn_deg', 0.0) for s in global_stats]))
        sharp_turns = int(np.sum([s.get('sharp_turns_60', 0) for s in global_stats]))
        print('      Dijkstra reached Root       = %d/%d' % (reached, len(global_stats)))
        print('      disconnected clean graph    = %d' % disconnected)
        print('      Root-guided search failed   = %d' % root_search_failed)
        print('      mean global path nodes       = %.1f' % mean_len)
        print('      mean Dijkstra visited states = %.1f' % mean_visited)
        print('      mean path turn angle         = %.1f deg' % mean_turn)
        print('      maximum path turn angle      = %.1f deg' % max_turn)
        print('      sharp turns >= 60 deg        = %d' % sharp_turns)
    print('      cross-section mode =', 'LOCAL Clean-Graph full maximum span' if args.full_cross_section_span else 'LOCAL Clean-Graph two-sided a---pi---b (Base/visualization only)')
    junctions = first_junctions_per_cluster(clusters, n)
    print('[4/8] Save path_contact.ply')
    save_path_contact(ply, xyz, root_idx, clean_graph_edges, clean_graph_stats, raw_triangle_count, raw_tri_shape, len(raw_graph_edges), out / 'path_contact.ply')
    print('[5/8] Save root_apex.ply')
    save_root_apex(ply, xyz, root_idx, tips, clean_adj, clean_graph_edges, out / 'root_apex.ply')
    print('[6/8] Save Apex_root.ply')
    save_apex_root(ply, xyz, root_idx, tips, bases, junctions, clusters, clean_graph_edges, clean_graph_stats, out / 'Apex_root.ply')
    print('[7/8] Save root_path.ply')
    save_root_paths(ply, xyz, root_idx, tips, bases, junctions, clusters, solver, Dr, clean_adj, out / 'root_path.ply', full_cross_section_span=args.full_cross_section_span)
    print('[8/8] Save leaves.ply + stem.ply')
    boundary_instances = extract_apex_to_nearest_boundary_regions(xyz, clusters, solver, Dr, tolerance_percentage=float(args.boundary_tol))
    valid = [np.asarray(idx, dtype=np.int64) for idx in boundary_instances if len(idx)]
    if not valid:
        raise RuntimeError('No leaf region was produced.')
    leaves_idx = np.unique(np.concatenate(valid))
    is_leaf = np.zeros(n, dtype=bool)
    is_leaf[leaves_idx] = True
    stem_idx = np.flatnonzero(~is_leaf).astype(np.int64)
    write_subset_original(ply, leaves_idx, out / 'leaves.ply')
    write_subset_original(ply, stem_idx, out / 'stem.ply')
    if len(leaves_idx) + len(stem_idx) != n:
        raise RuntimeError('leaves/stem split size mismatch.')
    print('\nDone:', out)
    print('  path_contact.ply')
    print('  root_apex.ply')
    print('  Apex_root.ply')
    print('  root_path.ply')
    print('  leaves.ply :', len(leaves_idx))
    print('  stem.ply   :', len(stem_idx))
if __name__ == '__main__':
    main()
