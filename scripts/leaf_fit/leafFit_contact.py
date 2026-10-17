#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LeafFit apex diagnostic exporter -- SuperSplat-compatible point-like Gaussians.

V9: adds the complete Dijkstra predecessor shortest-path tree and suspicious long-edge diagnostics.\n\nDIJ -> FPS -> local maximum -> apex visualization follows the LeafFit apex rule,
with Dr replaced by DIJ. Apex samples are removed from the base FPS layer before
adding enlarged purple markers, so SuperSplat cannot hide them behind duplicate cyan splats.

Purpose
-------
Do not change LeafFit segmentation logic. Only expose the stages used before
apex grouping so that the apex decision can be inspected visually.

Outputs (under <input_dir>/leafFit/)
-----------------------------------
1) Dr.ply
   - opacity-filtered plant used by the solver
   - synthetic Dr heatmap only (original RGB/SH removed)
   - green large marker = root

2) FPS.ply
   - FPS sparse points only
   - uniform cyan, enlarged

3) FPS_Dr.ply
   - FPS sparse points only
   - colored by the same Dr heatmap as Dr.ply

4) Apex.ply
   - very faint gray plant context
   - yellow large markers = detected apexes
   - green large marker = root

5) DIJ.ply
   - Dijkstra shortest-path distance from the same root to every Gaussian center
   - graph edges come from potpourri3d PointCloudLocalTriangulation
   - edge weight = Euclidean center-to-center distance
   - heatmap only; green large marker = root

6) FPS-apex.ply
   - same spatial FPS sparse points used by the Dr apex test
   - non-apex FPS points = small dark-cyan markers
   - purple very-large markers = apexes selected by the LeafFit local-maximum rule with Dr replaced by DIJ
   - apex points are removed from the cyan layer before purple markers are added (no duplicate splat overlap)
   - green large marker = root

7) DIJ-apex-path.ply
   - exact Dijkstra predecessor path from root to every DIJ-selected apex
   - every apex path uses a different color

8) DIJ-shortest-tree.ply
   - complete Dijkstra predecessor tree from the root
   - normal predecessor edges = cyan-blue
   - suspiciously long predecessor edges = red
   - green large marker = root

9) DIJ-long-edges.ply
   - only suspiciously long Dijkstra predecessor edges
   - red large endpoints + red edge traces
   - faint gray plant context

10) apex_debug.ply
   - very faint gray plant context
   - cyan points = all FPS points
   - yellow large points = detected apexes
   - orange points = rejected high-Dr FPS points closest to passing the apex test
   - red points = the strongest higher-Dr KNN competitor for those orange points
   - magenta links = rejected point -> strongest higher-Dr competitor
   - green large marker = root

All outputs preserve the original 3DGS/PGSR PLY schema so SuperSplat can open them.\nAll splats are converted to tiny isotropic Gaussians so they look like 3D points, not ellipsoids.\nOriginal Gaussian appearance is removed; only diagnostic colors are used.
"""

import os
import argparse
import colorsys
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra
from plyfile import PlyData, PlyElement
import potpourri3d as pp3d
import fpsample

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

SH_C0 = 0.28209479177387814

# ---------------- colors ----------------
GRAY = np.array([0.16, 0.16, 0.16], dtype=np.float32)
CYAN = np.array([0.05, 0.95, 1.00], dtype=np.float32)
YELLOW = np.array([1.00, 0.92, 0.00], dtype=np.float32)
GREEN = np.array([0.05, 1.00, 0.15], dtype=np.float32)
ORANGE = np.array([1.00, 0.34, 0.00], dtype=np.float32)
RED = np.array([1.00, 0.02, 0.02], dtype=np.float32)
MAGENTA = np.array([1.00, 0.00, 1.00], dtype=np.float32)
PURPLE = np.array([0.68, 0.12, 1.00], dtype=np.float32)
TREE_CYAN = np.array([0.06, 0.70, 1.00], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(
        description="Export separate LeafFit apex diagnostic PLY files."
    )
    p.add_argument("--input", required=True, type=str)
    p.add_argument("--leaf-fit-root", default="third_party/leaf_fit", type=str,
                   help="Accepted for command compatibility; this standalone script does not modify that folder.")
    p.add_argument("--disable-apex-cluster", action="store_true",
                   help="Accepted for compatibility. Apex clustering is never run here.")
    p.add_argument("--tip-k", default=128, type=int,
                   help="Euclidean KNN size used for FPS-space apex local-maximum testing.")
    p.add_argument("--path-k", default=256, type=int,
                   help="Accepted for compatibility. Root-path construction is not run here.")

    p.add_argument("--fps-count", default=8192, type=int,
                   help="Number of FPS points. Default: 8192.")
    p.add_argument("--fps-start-idx", default=0, type=int,
                   help="Start index in the opacity-filtered point set for deterministic FPS.")
    p.add_argument("--fps-method", choices=["vanilla", "bucket"], default="vanilla")

    p.add_argument("--opacity-threshold", default=0.10, type=float,
                   help="Remove the lowest opacity fraction before solver/FPS, matching LeafFit legacy preprocessing. Default: 0.10.")
    p.add_argument("--root-index", default=None, type=int,
                   help="Optional root index in the ORIGINAL input PLY. If omitted, use LeafFit-style automatic root selection.")

    # visualization only
    p.add_argument("--context-alpha", default=0.18, type=float,
                   help="Opacity of neutral-gray plant context in apex_debug.ply.")
    p.add_argument("--point-sigma-ratio", default=0.08, type=float,
                   help="Tiny isotropic Gaussian sigma as a fraction of median nearest-neighbor spacing. Default: 0.08.")
    p.add_argument("--fps-scale", default=1.8, type=float,
                   help="Point-size multiplier for FPS markers relative to the tiny base point sigma.")
    p.add_argument("--apex-scale", default=3.5, type=float,
                   help="Point-size multiplier for Dr-apex markers relative to the tiny base point sigma.")
    p.add_argument("--dij-apex-scale", default=8.0, type=float,
                   help="Point-size multiplier for purple DIJ-apex markers in FPS-apex.ply. Default: 8.0.")
    p.add_argument("--dij-path-point-scale", default=2.2, type=float,
                   help="Point-size multiplier for Dijkstra path vertices in DIJ-apex-path.ply. Default: 2.2.")
    p.add_argument("--dij-path-line-scale", default=0.85, type=float,
                   help="Point-size multiplier for tiny Gaussians used to draw Dijkstra path edges. Default: 0.85.")
    p.add_argument("--dij-path-apex-scale", default=7.0, type=float,
                   help="Point-size multiplier for colored DIJ-apex endpoints in DIJ-apex-path.ply. Default: 7.0.")
    p.add_argument("--dij-path-context-alpha", default=0.08, type=float,
                   help="Opacity of gray plant context in DIJ-apex-path.ply. Default: 0.08.")
    p.add_argument("--dij-path-step-ratio", default=0.35, type=float,
                   help="Interpolation spacing along displayed DIJ path edges as a fraction of median point spacing. Default: 0.35.")
    p.add_argument("--dij-tree-context-alpha", default=0.035, type=float,
                   help="Opacity of gray plant context in DIJ-shortest-tree.ply / DIJ-long-edges.ply. Default: 0.035.")
    p.add_argument("--dij-tree-line-scale", default=0.55, type=float,
                   help="Point-size multiplier for normal Dijkstra predecessor-tree edge traces. Default: 0.55.")
    p.add_argument("--dij-tree-long-line-scale", default=1.25, type=float,
                   help="Point-size multiplier for suspicious long predecessor-edge traces. Default: 1.25.")
    p.add_argument("--dij-tree-node-scale", default=0.90, type=float,
                   help="Point-size multiplier for predecessor-tree graph vertices. Default: 0.90.")
    p.add_argument("--dij-tree-long-endpoint-scale", default=3.8, type=float,
                   help="Point-size multiplier for endpoints of suspicious long predecessor edges. Default: 3.8.")
    p.add_argument("--dij-tree-step-ratio", default=0.45, type=float,
                   help="Interpolation spacing along predecessor-tree edges as a fraction of median point spacing. Default: 0.45.")
    p.add_argument("--dij-long-edge-factor", default=3.0, type=float,
                   help="Diagnostic threshold: predecessor edge is suspicious if length > factor * median nearest-neighbor spacing. Default: 3.0.")
    p.add_argument("--root-scale", default=3.5, type=float,
                   help="Point-size multiplier for root marker relative to the tiny base point sigma.")
    p.add_argument("--reject-scale", default=2.4, type=float,
                   help="Point-size multiplier for rejected/cause markers relative to the tiny base point sigma.")
    p.add_argument("--reject-dr-percentile", default=60.0, type=float,
                   help="Debug only: only rejected FPS points above this Dr percentile can receive rejection links.")
    p.add_argument("--max-reject-links", default=256, type=int,
                   help="Debug only: maximum number of rejection links.")
    return p.parse_args()


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def logit(alpha):
    a = float(np.clip(alpha, 1e-7, 1.0 - 1e-7))
    return float(np.log(a / (1.0 - a)))


def load_vertex_records(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise RuntimeError("PLY has no 'vertex' element")
    v = ply["vertex"].data.copy()
    names = set(v.dtype.names or [])
    for key in ("x", "y", "z"):
        if key not in names:
            raise RuntimeError("Input PLY must contain x, y, z")
    xyz = np.stack([
        np.asarray(v["x"], dtype=np.float64),
        np.asarray(v["y"], dtype=np.float64),
        np.asarray(v["z"], dtype=np.float64),
    ], axis=1)
    return v, xyz


def require_3dgs_fields(v):
    names = set(v.dtype.names or [])
    required = {
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    }
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(
            "Expected a 3DGS/PGSR PLY. Missing fields: " + str(missing)
        )


def zero_rest_sh(records):
    for name in records.dtype.names:
        if name.startswith("f_rest_"):
            records[name] = 0.0


def set_rgb(records, rgb):
    """Overwrite all visible color information. Original appearance is never retained."""
    if len(records) == 0:
        return
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim == 1:
        rgb = np.repeat(rgb[None, :], len(records), axis=0)
    dc = (rgb - 0.5) / SH_C0
    records["f_dc_0"] = dc[:, 0]
    records["f_dc_1"] = dc[:, 1]
    records["f_dc_2"] = dc[:, 2]
    zero_rest_sh(records)


def set_alpha(records, alpha):
    if len(records) == 0:
        return
    records["opacity"] = np.float32(logit(alpha))


def multiply_scale(records, factor):
    if len(records) == 0:
        return
    factor = np.asarray(factor, dtype=np.float64)
    if factor.ndim == 0:
        delta = np.float32(np.log(max(float(factor), 1e-8)))
        for name in ("scale_0", "scale_1", "scale_2"):
            records[name] = records[name] + delta
    else:
        delta = np.log(np.maximum(factor, 1e-8)).astype(np.float32)
        for name in ("scale_0", "scale_1", "scale_2"):
            records[name] = records[name] + delta


def set_point_shape(records, sigma):
    """Force each splat to a tiny isotropic Gaussian, i.e. a visible 3D point in SuperSplat."""
    if len(records) == 0:
        return
    sigma = float(max(sigma, 1e-8))
    raw_scale = np.float32(np.log(sigma))
    records["scale_0"] = raw_scale
    records["scale_1"] = raw_scale
    records["scale_2"] = raw_scale
    # Identity quaternion. The point is isotropic, but keep valid 3DGS rotation fields.
    records["rot_0"] = np.float32(1.0)
    records["rot_1"] = np.float32(0.0)
    records["rot_2"] = np.float32(0.0)
    records["rot_3"] = np.float32(0.0)


def apply_opacity_filter(vertices, fraction):
    n = len(vertices)
    if fraction <= 0 or "opacity" not in vertices.dtype.names:
        return np.arange(n, dtype=np.int64)

    # LeafFit upstream compares the stored GaussianData opacity values directly
    # against a percentile. GaussianData loaders typically activate opacity.
    # Here we use sigmoid(raw) to obtain the activated opacity explicitly.
    opacity = sigmoid(np.asarray(vertices["opacity"], dtype=np.float64))
    threshold = np.percentile(opacity, fraction * 100.0)
    keep = np.where(opacity > threshold)[0].astype(np.int64)
    if len(keep) < 10:
        raise RuntimeError("Opacity filtering left too few points; try --opacity-threshold 0")
    return keep


def make_solver(points):
    return pp3d.PointCloudHeatSolver(points, t_coef=1e8)


def auto_select_root(points, solver):
    """Standalone reproduction of LeafFit's legacy automatic-root selection."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = np.real(eigenvectors[:, order])

    asymmetries = []
    axis_info = []
    for vec in eigenvectors.T:
        projections = np.dot(centered, vec)
        pmin, pmax = projections.min(), projections.max()
        midpoint = (pmin + pmax) / 2.0
        left = int(np.sum(projections < midpoint))
        right = int(np.sum(projections >= midpoint))
        asymmetries.append(abs(left - right) / float(len(projections)))
        axis_info.append((projections, left, right))

    best = int(np.argmax(asymmetries))
    projections, _, _ = axis_info[best]

    p1 = np.percentile(projections, 1)
    p99 = np.percentile(projections, 99)
    robust_mid = (p1 + p99) / 2.0
    left = int(np.sum(projections < robust_mid))
    right = int(np.sum(projections >= robust_mid))

    if left < right:
        sparse_side = "left"
        threshold = np.percentile(projections, 5)
        candidate_mask = projections <= threshold
    else:
        sparse_side = "right"
        threshold = np.percentile(projections, 95)
        candidate_mask = projections >= threshold

    candidate_indices = np.where(candidate_mask)[0]
    candidate_proj = projections[candidate_mask]
    extreme_local = int(np.argmin(candidate_proj) if sparse_side == "left" else np.argmax(candidate_proj))
    extreme_idx = int(candidate_indices[extreme_local])

    try:
        d = np.asarray(solver.compute_distance(extreme_idx), dtype=np.float64)
        candidate_d = d[candidate_indices]
        dist_threshold = np.percentile(candidate_d, 15)
        dist_threshold = max(float(dist_threshold), float(d.mean()) * 0.05)
        neighborhood = np.where(d <= dist_threshold)[0]
        neighborhood_proj = projections[neighborhood]
    except Exception:
        neighborhood = candidate_indices
        neighborhood_proj = projections[neighborhood]

    local = int(np.argmin(neighborhood_proj) if sparse_side == "left" else np.argmax(neighborhood_proj))
    return int(neighborhood[local])


def run_fps(points, n_samples, method, start_idx):
    n_samples = int(min(max(1, n_samples), len(points)))
    if n_samples == len(points):
        return np.arange(len(points), dtype=np.int64)

    start_idx = int(np.clip(start_idx, 0, len(points) - 1))
    pc = np.ascontiguousarray(points, dtype=np.float64)
    if method == "bucket":
        idx = fpsample.bucket_fps_kdline_sampling(pc, n_samples, h=5, start_idx=start_idx)
    else:
        idx = fpsample.fps_sampling(pc, n_samples, start_idx=start_idx)
    return np.asarray(idx, dtype=np.int64)


def analyze_apexes(points, dr, sparse_idx, tip_k):
    """
    Diagnostic form of LeafFit's apex criterion:
      - build Euclidean KDTree on FPS points
      - query tip_k neighbors (self is normally the first result)
      - current FPS point passes iff no queried other FPS point has larger Dr

    The upstream code performs the same test through temperature = max(Dr)-Dr.
    """
    sparse_xyz = points[sparse_idx]
    sparse_dr = dr[sparse_idx]
    tree = cKDTree(sparse_xyz)

    if len(sparse_xyz) == 1:
        return {
            "sparse_dr": sparse_dr,
            "higher_count": np.zeros(1, dtype=np.int32),
            "killer_local": np.array([-1], dtype=np.int64),
            "apex_margin": np.array([np.inf], dtype=np.float64),
            "apex_local": np.array([0], dtype=np.int64),
            "query_k": 1,
        }

    query_k = int(min(max(2, tip_k), len(sparse_xyz)))
    _, nn = tree.query(sparse_xyz, k=query_k, workers=-1)
    if nn.ndim == 1:
        nn = nn[:, None]

    higher_count = np.zeros(len(sparse_idx), dtype=np.int32)
    killer_local = np.full(len(sparse_idx), -1, dtype=np.int64)
    apex_margin = np.full(len(sparse_idx), np.inf, dtype=np.float64)

    eps = max(float(np.ptp(sparse_dr)), 1.0) * 1e-12

    for i in range(len(sparse_idx)):
        # Match the normal KDTree behavior used by upstream: nearest hit is self.
        # To remain robust to ties/duplicates, remove every exact self index.
        neigh = nn[i]
        neigh = neigh[neigh != i]
        if len(neigh) == 0:
            continue

        nd = sparse_dr[neigh]
        max_pos = int(np.argmax(nd))
        killer_local[i] = int(neigh[max_pos])
        max_neigh_dr = float(nd[max_pos])

        higher = nd > (float(sparse_dr[i]) + eps)
        higher_count[i] = int(np.count_nonzero(higher))
        apex_margin[i] = float(sparse_dr[i]) - max_neigh_dr

    apex_local = np.where(higher_count == 0)[0].astype(np.int64)
    return {
        "sparse_dr": sparse_dr,
        "higher_count": higher_count,
        "killer_local": killer_local,
        "apex_margin": apex_margin,
        "apex_local": apex_local,
        "query_k": query_k,
    }



def analyze_apexes_finite(points, field, sparse_idx, tip_k):
    """
    Same FPS-space local-maximum rule as analyze_apexes(), but intended for a
    distance field that may contain +inf (DIJ can be disconnected).

    Rule for a sampled point i:
      1) field[i] must be finite;
      2) query the same Euclidean FPS KNN neighborhood;
      3) ignore neighbors whose field is non-finite;
      4) i is an apex iff no finite queried neighbor has a larger field value.

    Thus disconnected/unreachable DIJ points are never reported as apexes.
    """
    sparse_xyz = points[sparse_idx]
    sparse_field = np.asarray(field[sparse_idx], dtype=np.float64)
    tree = cKDTree(sparse_xyz)

    if len(sparse_xyz) == 1:
        finite = np.isfinite(sparse_field)
        return {
            "sparse_field": sparse_field,
            "higher_count": np.zeros(1, dtype=np.int32),
            "killer_local": np.array([-1], dtype=np.int64),
            "apex_margin": np.array([np.inf], dtype=np.float64),
            "apex_local": np.array([0], dtype=np.int64) if finite[0] else np.empty(0, dtype=np.int64),
            "query_k": 1,
        }

    query_k = int(min(max(2, tip_k), len(sparse_xyz)))
    _, nn = tree.query(sparse_xyz, k=query_k, workers=-1)
    if nn.ndim == 1:
        nn = nn[:, None]

    higher_count = np.zeros(len(sparse_idx), dtype=np.int32)
    killer_local = np.full(len(sparse_idx), -1, dtype=np.int64)
    apex_margin = np.full(len(sparse_idx), np.nan, dtype=np.float64)

    finite_field = field[np.isfinite(field)]
    field_scale = float(np.ptp(finite_field)) if len(finite_field) else 1.0
    eps = max(field_scale, 1.0) * 1e-12

    for i in range(len(sparse_idx)):
        if not np.isfinite(sparse_field[i]):
            # Unreachable in DIJ: never allow it to become a local maximum.
            higher_count[i] = -1
            continue

        neigh = nn[i]
        neigh = neigh[neigh != i]
        if len(neigh) == 0:
            apex_margin[i] = np.inf
            continue

        finite_neigh = neigh[np.isfinite(sparse_field[neigh])]
        if len(finite_neigh) == 0:
            apex_margin[i] = np.inf
            continue

        nd = sparse_field[finite_neigh]
        max_pos = int(np.argmax(nd))
        killer_local[i] = int(finite_neigh[max_pos])
        max_neigh = float(nd[max_pos])

        higher = nd > (float(sparse_field[i]) + eps)
        higher_count[i] = int(np.count_nonzero(higher))
        apex_margin[i] = float(sparse_field[i]) - max_neigh

    apex_local = np.where((higher_count == 0) & np.isfinite(sparse_field))[0].astype(np.int64)
    return {
        "sparse_field": sparse_field,
        "higher_count": higher_count,
        "killer_local": killer_local,
        "apex_margin": apex_margin,
        "apex_local": apex_local,
        "query_k": query_k,
    }


def analyze_dij_apex_leafit_style(points, dij, sparse_idx, tip_k):
    """
    LeafFit-style apex detection with one and only one conceptual replacement:

        original: Dr  -> temperature = max(Dr) - Dr -> local temperature minimum
        this test: DIJ -> temperature = max(DIJ) - DIJ -> local temperature minimum

    FPS itself is spatial FPS and does NOT depend on Dr or DIJ, exactly as in the
    original idea. The same sparse_idx is used here.

    For each FPS point i, query its Euclidean KNN among FPS points. After removing
    self, i is selected iff every finite neighbor has temperature >= temp(i), which
    is exactly equivalent to DIJ(i) >= DIJ(neighbor) for every finite neighbor.

    Non-finite DIJ points (disconnected from root) are excluded from candidacy and
    ignored as neighbors, because +inf is not a meaningful root distance.
    """
    sparse_xyz = points[sparse_idx]
    sparse_dij = np.asarray(dij[sparse_idx], dtype=np.float64)
    n = len(sparse_idx)
    if n == 0:
        return {
            "sparse_dij": sparse_dij,
            "temperature": np.empty(0, dtype=np.float64),
            "apex_local": np.empty(0, dtype=np.int64),
            "query_k": 0,
            "finite_mask": np.empty(0, dtype=bool),
        }

    finite_mask = np.isfinite(sparse_dij)
    if not np.any(finite_mask):
        return {
            "sparse_dij": sparse_dij,
            "temperature": np.full(n, np.nan, dtype=np.float64),
            "apex_local": np.empty(0, dtype=np.int64),
            "query_k": min(max(1, int(tip_k)), n),
            "finite_mask": finite_mask,
        }

    max_dij = float(np.max(sparse_dij[finite_mask]))
    temperature = np.full(n, np.nan, dtype=np.float64)
    temperature[finite_mask] = max_dij - sparse_dij[finite_mask]

    tree = cKDTree(sparse_xyz)
    query_k = int(min(max(2, int(tip_k)), n)) if n > 1 else 1
    _, nn = tree.query(sparse_xyz, k=query_k, workers=-1)
    if nn.ndim == 1:
        nn = nn[:, None]

    found = []
    higher_count = np.full(n, -1, dtype=np.int32)
    max_neighbor_dij = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if not finite_mask[i]:
            continue

        neigh = nn[i]
        # Same intent as LeafFit local_mask[1:]: remove the current FPS sample.
        neigh = neigh[neigh != i]
        if len(neigh) == 0:
            found.append(i)
            higher_count[i] = 0
            continue

        neigh = neigh[finite_mask[neigh]]
        if len(neigh) == 0:
            found.append(i)
            higher_count[i] = 0
            continue

        temp_i = temperature[i]
        temp_mask = temperature[neigh]
        higher_count[i] = int(np.count_nonzero(sparse_dij[neigh] > sparse_dij[i]))
        max_neighbor_dij[i] = float(np.max(sparse_dij[neigh]))

        # This is the same condition used by LeafFit find_local_tips(), but with
        # DIJ-derived temperature replacing Dr-derived temperature.
        if np.all(temp_mask >= temp_i):
            found.append(i)

    return {
        "sparse_dij": sparse_dij,
        "temperature": temperature,
        "apex_local": np.asarray(found, dtype=np.int64),
        "query_k": query_k,
        "finite_mask": finite_mask,
        "higher_count": higher_count,
        "max_neighbor_dij": max_neighbor_dij,
    }


def build_surface_dijkstra_graph(points):
    """
    Build an undirected weighted graph from potpourri3d's point-cloud local
    triangulation. Each valid triangle contributes its three edges. Duplicate
    edges are removed. Edge weight is Euclidean distance between Gaussian
    centers.

    This graph is a discrete surface-neighborhood baseline for Dijkstra. It is
    intentionally distinct from Heat Method Dr: DIJ sums edge lengths along
    a shortest graph path, whereas Dr is obtained by the heat-method PDE solve.
    """
    print("Building PointCloudLocalTriangulation for DIJ ...")
    tri_solver = pp3d.PointCloudLocalTriangulation(
        np.ascontiguousarray(points, dtype=np.float64),
        with_degeneracy_heuristic=True,
    )
    tris = np.asarray(tri_solver.get_local_triangulation(), dtype=np.int64)
    if tris.ndim != 3 or tris.shape[-1] != 3:
        raise RuntimeError(f"Unexpected local triangulation shape: {tris.shape}")

    flat = tris.reshape(-1, 3)
    valid = np.all(flat >= 0, axis=1) & np.all(flat < len(points), axis=1)
    flat = flat[valid]
    if len(flat) == 0:
        raise RuntimeError("PointCloudLocalTriangulation returned no valid triangles")

    e01 = flat[:, [0, 1]]
    e12 = flat[:, [1, 2]]
    e20 = flat[:, [2, 0]]
    edges = np.vstack([e01, e12, e20])
    edges.sort(axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.unique(edges, axis=0)
    if len(edges) == 0:
        raise RuntimeError("No valid surface-neighborhood edges for Dijkstra")

    u = edges[:, 0]
    v = edges[:, 1]
    w = np.linalg.norm(points[u] - points[v], axis=1).astype(np.float64)
    good = np.isfinite(w) & (w > 1e-12)
    u, v, w = u[good], v[good], w[good]

    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    data = np.concatenate([w, w])
    graph = csr_matrix((data, (rows, cols)), shape=(len(points), len(points)))

    degree = np.diff(graph.indptr)
    print("DIJ graph vertices:", len(points))
    print("DIJ graph undirected edges:", len(w))
    print("DIJ graph degree min/mean/max: {}/{:.2f}/{}".format(
        int(degree.min()) if len(degree) else 0,
        float(degree.mean()) if len(degree) else 0.0,
        int(degree.max()) if len(degree) else 0,
    ))
    return graph, edges


def compute_dijkstra_distance(points, root_idx):
    graph, edges = build_surface_dijkstra_graph(points)
    print("Computing DIJ shortest-path distances from root ...")
    dist, predecessors = scipy_dijkstra(
        graph,
        directed=False,
        indices=int(root_idx),
        return_predecessors=True,
    )
    dist = np.asarray(dist, dtype=np.float64)
    predecessors = np.asarray(predecessors, dtype=np.int64)
    finite = np.isfinite(dist)
    print("DIJ reachable points: {}/{}".format(int(finite.sum()), len(dist)))
    if finite.any():
        print("DIJ finite range: [{:.6f}, {:.6f}]".format(
            float(dist[finite].min()), float(dist[finite].max())
        ))
    return dist, predecessors, edges


def distance_colors_with_inf(values):
    """Turbo-style heatmap for finite distances; unreachable nodes are dark gray."""
    values = np.asarray(values, dtype=np.float64)
    rgb = np.repeat(GRAY[None, :], len(values), axis=0).astype(np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return rgb
    fv = values[finite]
    lo = float(np.percentile(fv, 1))
    hi = float(np.percentile(fv, 99))
    rgb[finite] = dr_colors(fv, lo, hi)
    return rgb

def dr_colors(values, lo=None, hi=None):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if lo is None:
        lo = float(np.percentile(values, 1))
    if hi is None:
        hi = float(np.percentile(values, 99))
    if hi <= lo + 1e-12:
        t = np.zeros(len(values), dtype=np.float64)
    else:
        t = np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    if plt is not None:
        return plt.get_cmap("turbo")(t)[:, :3].astype(np.float32)

    # fallback: blue -> cyan -> yellow -> red
    rgb = np.zeros((len(t), 3), dtype=np.float32)
    for i, x in enumerate(t):
        if x < 1.0 / 3.0:
            u = x * 3.0
            rgb[i] = [0.0, u, 1.0]
        elif x < 2.0 / 3.0:
            u = (x - 1.0 / 3.0) * 3.0
            rgb[i] = [u, 1.0, 1.0 - u]
        else:
            u = (x - 2.0 / 3.0) * 3.0
            rgb[i] = [1.0, 1.0 - u, 0.0]
    return rgb


def estimate_spacing(points):
    if len(points) < 2:
        return 1e-3
    tree = cKDTree(points)
    d, _ = tree.query(points, k=2, workers=-1)
    nn = d[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 1e-12)]
    return float(np.median(nn)) if len(nn) else 1e-3


def quat_x_to_direction(vec):
    vec = np.asarray(vec, dtype=np.float64)
    n = np.linalg.norm(vec)
    if n < 1e-12:
        return np.array([1, 0, 0, 0], dtype=np.float32)
    b = vec / n
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 0.999999:
        return np.array([1, 0, 0, 0], dtype=np.float32)
    if dot < -0.999999:
        return np.array([0, 0, 1, 0], dtype=np.float32)
    cross = np.cross(a, b)
    q = np.array([1.0 + dot, cross[0], cross[1], cross[2]], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q.astype(np.float32)


def make_link_point_records(template_vertices, source_orig_idx, p0, p1, sigma, spacing, rgb=MAGENTA):
    """Represent a debug link as a chain of tiny spherical Gaussians, never as an elongated splat."""
    if len(source_orig_idx) == 0:
        return np.empty(0, dtype=template_vertices.dtype)
    all_xyz = []
    all_src = []
    for src, a, b in zip(source_orig_idx, p0, p1):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        L = float(np.linalg.norm(b - a))
        n = max(2, int(np.ceil(L / max(float(spacing), 1e-8))) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        pts = a[None, :] * (1.0 - t) + b[None, :] * t
        all_xyz.append(pts)
        all_src.extend([int(src)] * n)
    xyz = np.vstack(all_xyz)
    rec = template_vertices[np.asarray(all_src, dtype=np.int64)].copy()
    rec["x"] = xyz[:, 0]
    rec["y"] = xyz[:, 1]
    rec["z"] = xyz[:, 2]
    set_point_shape(rec, sigma)
    set_rgb(rec, rgb)
    set_alpha(rec, 0.995)
    for name in ("nx", "ny", "nz"):
        if name in rec.dtype.names:
            rec[name] = 0.0
    return rec


def write_ply(path, records, comments):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    elem = PlyElement.describe(records, "vertex")
    PlyData([elem], text=False, comments=comments).write(path)


def make_root_layer(vertices, root_orig_idx, sigma):
    rec = vertices[[root_orig_idx]].copy()
    set_point_shape(rec, sigma)
    set_rgb(rec, GREEN)
    set_alpha(rec, 0.999)
    return rec


def trace_dijkstra_path(predecessors, root_idx, target_idx):
    """
    Recover the exact shortest-path predecessor chain produced by scipy.csgraph.dijkstra.

    predecessors[v] is the previous vertex on the shortest route from root to v.
    Returned indices are ordered root -> ... -> target.
    """
    predecessors = np.asarray(predecessors, dtype=np.int64)
    n = len(predecessors)
    root_idx = int(root_idx)
    target_idx = int(target_idx)

    if not (0 <= root_idx < n and 0 <= target_idx < n):
        return np.empty(0, dtype=np.int64)
    if target_idx == root_idx:
        return np.array([root_idx], dtype=np.int64)

    path = [target_idx]
    seen = {target_idx}
    cur = target_idx

    for _ in range(n + 1):
        if cur == root_idx:
            break
        prev = int(predecessors[cur])

        # scipy uses a negative sentinel (normally -9999) when no predecessor exists.
        if prev < 0 or prev >= n or prev in seen:
            return np.empty(0, dtype=np.int64)

        path.append(prev)
        seen.add(prev)
        cur = prev

        if cur == root_idx:
            break

    if path[-1] != root_idx:
        return np.empty(0, dtype=np.int64)

    path.reverse()
    return np.asarray(path, dtype=np.int64)


def distinct_path_color(path_id):
    """
    Deterministic bright colors for different root->apex paths.
    Golden-ratio hue stepping avoids adjacent paths receiving nearly identical colors.
    """
    h = (0.08 + float(path_id) * 0.6180339887498949) % 1.0
    s = 0.88
    v = 1.00
    return np.asarray(colorsys.hsv_to_rgb(h, s, v), dtype=np.float32)


def make_dijkstra_apex_path_layers(
    vertices,
    keep_orig_idx,
    points,
    predecessors,
    root_filtered_idx,
    dij_apex_filtered_idx,
    base_sigma,
    median_spacing,
    args,
):
    """
    Visualize the exact Dijkstra shortest path root -> apex for every DIJ-selected apex.

    Returns
    -------
    records : structured ndarray
        SuperSplat-compatible point-like Gaussian records.
    path_summaries : list of tuples
        (path_id, apex_filtered_idx, vertex_count, path_length)
    """
    layers = []

    # Low-opacity neutral context to show where the paths lie on the plant.
    context = make_context_layer(
        vertices,
        keep_orig_idx,
        base_sigma,
        float(args.dij_path_context_alpha),
    )
    layers.append(context)

    path_summaries = []

    for path_id, apex_filtered_idx in enumerate(np.asarray(dij_apex_filtered_idx, dtype=np.int64)):
        path_idx = trace_dijkstra_path(
            predecessors,
            root_filtered_idx,
            int(apex_filtered_idx),
        )
        if len(path_idx) == 0:
            print(
                "WARNING: could not recover DIJ predecessor path for apex filtered idx",
                int(apex_filtered_idx),
            )
            continue

        rgb = distinct_path_color(path_id)
        path_orig_idx = keep_orig_idx[path_idx]

        # Actual graph vertices on the Dijkstra route.
        path_pts = vertices[path_orig_idx].copy()
        set_point_shape(path_pts, base_sigma * float(args.dij_path_point_scale))
        set_rgb(path_pts, rgb)
        set_alpha(path_pts, 0.995)
        layers.append(path_pts)

        # Draw each graph edge as a dense chain of tiny isotropic Gaussians.
        if len(path_idx) >= 2:
            p0 = points[path_idx[:-1]]
            p1 = points[path_idx[1:]]
            edge_layer = make_link_point_records(
                vertices,
                path_orig_idx[:-1],
                p0,
                p1,
                sigma=base_sigma * float(args.dij_path_line_scale),
                spacing=max(
                    median_spacing * float(args.dij_path_step_ratio),
                    1e-8,
                ),
                rgb=rgb,
            )
            layers.append(edge_layer)

            path_length = float(
                np.sum(np.linalg.norm(p1 - p0, axis=1))
            )
        else:
            path_length = 0.0

        # Large colored endpoint for this apex.
        apex_orig_idx = int(keep_orig_idx[int(apex_filtered_idx)])
        apex_marker = vertices[[apex_orig_idx]].copy()
        set_point_shape(
            apex_marker,
            base_sigma * float(args.dij_path_apex_scale),
        )
        set_rgb(apex_marker, rgb)
        set_alpha(apex_marker, 0.9999)
        layers.append(apex_marker)

        path_summaries.append(
            (
                int(path_id),
                int(apex_filtered_idx),
                int(len(path_idx)),
                path_length,
            )
        )

    # Root is shown once, in green, above all path colors.
    root_orig_idx = int(keep_orig_idx[int(root_filtered_idx)])
    root_layer = make_root_layer(
        vertices,
        root_orig_idx,
        base_sigma * float(args.root_scale),
    )
    layers.append(root_layer)

    nonempty = [x for x in layers if len(x)]
    if not nonempty:
        return np.empty(0, dtype=vertices.dtype), path_summaries
    return np.concatenate(nonempty), path_summaries



def analyze_dijkstra_predecessor_tree(points, predecessors, root_idx, median_spacing, long_factor):
    """
    Convert scipy Dijkstra predecessors into the exact shortest-path tree edges.

    Every reachable non-root vertex v contributes exactly one tree edge:
        predecessors[v] -> v

    This is NOT all local-triangulation graph edges. It is only the subset
    actually selected by Dijkstra as the predecessor tree from the chosen root.
    """
    points = np.asarray(points, dtype=np.float64)
    predecessors = np.asarray(predecessors, dtype=np.int64)
    root_idx = int(root_idx)

    child = np.arange(len(points), dtype=np.int64)
    valid = (
        (child != root_idx)
        & (predecessors >= 0)
        & (predecessors < len(points))
    )

    child = child[valid]
    parent = predecessors[valid]

    if len(child) == 0:
        return {
            "parent": np.empty(0, dtype=np.int64),
            "child": np.empty(0, dtype=np.int64),
            "length": np.empty(0, dtype=np.float64),
            "long_mask": np.empty(0, dtype=bool),
            "threshold": float("inf"),
        }

    length = np.linalg.norm(points[child] - points[parent], axis=1)
    threshold = float(long_factor) * float(median_spacing)
    long_mask = np.isfinite(length) & (length > threshold)

    finite_len = length[np.isfinite(length)]
    print("DIJ predecessor-tree edges:", len(length))
    if len(finite_len):
        print(
            "DIJ predecessor edge length min/median/p95/p99/max: "
            "{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}".format(
                float(np.min(finite_len)),
                float(np.median(finite_len)),
                float(np.percentile(finite_len, 95)),
                float(np.percentile(finite_len, 99)),
                float(np.max(finite_len)),
            )
        )
    print(
        "Suspicious-long predecessor threshold: {:.6f} = {:.2f} * median spacing".format(
            threshold, float(long_factor)
        )
    )
    print(
        "Suspicious-long predecessor edges: {}/{}".format(
            int(np.count_nonzero(long_mask)), len(length)
        )
    )

    return {
        "parent": parent,
        "child": child,
        "length": length,
        "long_mask": long_mask,
        "threshold": threshold,
    }


def make_dijkstra_shortest_tree_layers(
    vertices,
    keep_orig_idx,
    points,
    tree_info,
    root_filtered_idx,
    base_sigma,
    median_spacing,
    args,
    long_only=False,
):
    """
    Export the exact Dijkstra predecessor tree used by DIJ.

    Normal predecessor edges are cyan-blue.
    Edges longer than --dij-long-edge-factor * median_spacing are red.

    When long_only=True, only suspicious long predecessor edges are exported,
    along with faint plant context and enlarged red endpoints.
    """
    parent = np.asarray(tree_info["parent"], dtype=np.int64)
    child = np.asarray(tree_info["child"], dtype=np.int64)
    long_mask = np.asarray(tree_info["long_mask"], dtype=bool)

    layers = []

    context = make_context_layer(
        vertices,
        keep_orig_idx,
        base_sigma,
        float(args.dij_tree_context_alpha),
    )
    layers.append(context)

    spacing = max(
        float(median_spacing) * float(args.dij_tree_step_ratio),
        1e-8,
    )

    if not long_only:
        # Show the actual original graph vertices used in the predecessor tree.
        tree_nodes_filtered = np.unique(
            np.concatenate([parent, child]) if len(parent) else np.array([root_filtered_idx], dtype=np.int64)
        )
        tree_nodes_orig = keep_orig_idx[tree_nodes_filtered]
        tree_nodes = vertices[tree_nodes_orig].copy()
        set_point_shape(tree_nodes, base_sigma * float(args.dij_tree_node_scale))
        set_rgb(tree_nodes, TREE_CYAN)
        set_alpha(tree_nodes, 0.78)
        layers.append(tree_nodes)

        normal_mask = ~long_mask
        if np.any(normal_mask):
            normal_parent = parent[normal_mask]
            normal_child = child[normal_mask]
            normal_edges = make_link_point_records(
                vertices,
                keep_orig_idx[normal_parent],
                points[normal_parent],
                points[normal_child],
                sigma=base_sigma * float(args.dij_tree_line_scale),
                spacing=spacing,
                rgb=TREE_CYAN,
            )
            set_alpha(normal_edges, 0.88)
            layers.append(normal_edges)

    # Long/suspicious edges: always red and thicker.
    if np.any(long_mask):
        lp = parent[long_mask]
        lc = child[long_mask]

        long_edges = make_link_point_records(
            vertices,
            keep_orig_idx[lp],
            points[lp],
            points[lc],
            sigma=base_sigma * float(args.dij_tree_long_line_scale),
            spacing=spacing,
            rgb=RED,
        )
        set_alpha(long_edges, 0.999)
        layers.append(long_edges)

        endpoints_filtered = np.unique(np.concatenate([lp, lc]))
        endpoints_orig = keep_orig_idx[endpoints_filtered]
        endpoint_layer = vertices[endpoints_orig].copy()
        set_point_shape(
            endpoint_layer,
            base_sigma * float(args.dij_tree_long_endpoint_scale),
        )
        set_rgb(endpoint_layer, RED)
        set_alpha(endpoint_layer, 0.9999)
        layers.append(endpoint_layer)

    root_orig_idx = int(keep_orig_idx[int(root_filtered_idx)])
    root_layer = make_root_layer(
        vertices,
        root_orig_idx,
        base_sigma * float(args.root_scale),
    )
    layers.append(root_layer)

    nonempty = [x for x in layers if len(x)]
    if not nonempty:
        return np.empty(0, dtype=vertices.dtype)
    return np.concatenate(nonempty)



def make_context_layer(vertices, keep_orig_idx, sigma, alpha):
    rec = vertices[keep_orig_idx].copy()
    set_point_shape(rec, sigma)
    set_rgb(rec, GRAY)
    set_alpha(rec, alpha)
    return rec


def select_rejected_debug_points(analysis, args):
    """
    Debug-only selection: among rejected FPS points with relatively high Dr,
    keep those whose apex_margin is closest to zero. These are the points most
    similar to a local maximum but rejected by at least one higher-Dr neighbor.
    """
    rejected = np.where(analysis["higher_count"] > 0)[0]
    if len(rejected) == 0:
        return rejected

    cutoff = np.percentile(analysis["sparse_dr"], args.reject_dr_percentile)
    rejected = rejected[analysis["sparse_dr"][rejected] >= cutoff]
    if len(rejected) == 0:
        return rejected

    # Larger margin means closer to 0 -> closer to passing the local-max test.
    order = np.argsort(analysis["apex_margin"][rejected])[::-1]
    rejected = rejected[order]
    return rejected[:max(0, int(args.max_reject_links))]


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input)
    out_dir = os.path.join(os.path.dirname(input_path), "leafFit")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("LeafFit Apex diagnostics v9 -- DIJ apexes + Dijkstra paths + predecessor tree")
    print("=" * 80)
    print("Input :", input_path)
    print("Output:", out_dir)

    vertices, xyz_full = load_vertex_records(input_path)
    require_3dgs_fields(vertices)
    print("Input Gaussian count:", len(vertices))

    keep_orig_idx = apply_opacity_filter(vertices, args.opacity_threshold)
    points = xyz_full[keep_orig_idx]
    print("After opacity filter:", len(points))

    median_spacing = estimate_spacing(points)
    base_sigma = max(median_spacing * float(args.point_sigma_ratio), 1e-8)
    print("Median nearest-neighbor spacing:", median_spacing)
    print("Point-like Gaussian sigma:", base_sigma)

    print("Building PointCloudHeatSolver(t_coef=1e8) ...")
    solver = make_solver(points)

    if args.root_index is None:
        root_filtered_idx = auto_select_root(points, solver)
        root_orig_idx = int(keep_orig_idx[root_filtered_idx])
        print("Auto root filtered idx:", root_filtered_idx)
        print("Auto root original idx:", root_orig_idx)
    else:
        hits = np.where(keep_orig_idx == int(args.root_index))[0]
        if len(hits) == 0:
            raise RuntimeError(
                "--root-index was removed by opacity filtering. Use --opacity-threshold 0 or another root."
            )
        root_filtered_idx = int(hits[0])
        root_orig_idx = int(args.root_index)
        print("Using root original idx:", root_orig_idx)
        print("Mapped root filtered idx:", root_filtered_idx)

    print("Computing Dr ...")
    dr = np.asarray(solver.compute_distance(root_filtered_idx), dtype=np.float64)
    print("Dr range: [{:.6f}, {:.6f}]".format(float(dr.min()), float(dr.max())))

    # Discrete shortest-path baseline on a surface-neighborhood graph.
    dij, dij_predecessor, dij_edges = compute_dijkstra_distance(points, root_filtered_idx)

    fps_count = min(int(args.fps_count), len(points))
    print("FPS count:", fps_count)
    sparse_idx = run_fps(points, fps_count, args.fps_method, args.fps_start_idx)
    sparse_orig_idx = keep_orig_idx[sparse_idx]

    print("Apex test with tip-k =", args.tip_k)
    analysis = analyze_apexes(points, dr, sparse_idx, args.tip_k)
    apex_local = analysis["apex_local"]
    apex_filtered_idx = sparse_idx[apex_local]
    apex_orig_idx = keep_orig_idx[apex_filtered_idx]
    print("Detected apex count:", len(apex_orig_idx))
    print("Detected apex original indices:", apex_orig_idx.tolist())

    # DIJ-based apex detection: reproduce the LeafFit local-tip logic, with Dr
    # replaced by DIJ. FPS itself remains the same spatial FPS sample set.
    print("DIJ -> FPS -> local-maximum apex test with tip-k =", args.tip_k)
    dij_apex_analysis = analyze_dij_apex_leafit_style(points, dij, sparse_idx, args.tip_k)
    dij_apex_local = dij_apex_analysis["apex_local"]
    dij_apex_filtered_idx = sparse_idx[dij_apex_local]
    dij_apex_orig_idx = keep_orig_idx[dij_apex_filtered_idx]
    print("Finite DIJ FPS points: {}/{}".format(
        int(np.count_nonzero(dij_apex_analysis["finite_mask"])), len(sparse_idx)))
    print("Detected DIJ-apex count:", len(dij_apex_orig_idx))
    print("Detected DIJ-apex original indices:", dij_apex_orig_idx.tolist())
    if len(dij_apex_orig_idx) == 0:
        print("WARNING: DIJ local-maximum test produced zero apexes; FPS-apex.ply will contain no purple markers.")

    dr_lo = float(np.percentile(dr, 1))
    dr_hi = float(np.percentile(dr, 99))

    # ------------------------------------------------------------------
    # 1) Dr.ply
    # ------------------------------------------------------------------
    dr_layer = vertices[keep_orig_idx].copy()
    set_point_shape(dr_layer, base_sigma)
    set_rgb(dr_layer, dr_colors(dr, dr_lo, dr_hi))
    set_alpha(dr_layer, 0.995)
    root_layer = make_root_layer(vertices, root_orig_idx, base_sigma * args.root_scale)
    dr_records = np.concatenate([dr_layer, root_layer])
    write_ply(
        os.path.join(out_dir, "Dr.ply"),
        dr_records,
        [
            "Synthetic Dr heatmap on tiny isotropic Gaussians; original Gaussian colors/shapes removed",
            "Green large marker = root",
            "Only opacity-filtered points used by the standalone solver are shown",
        ],
    )

    # ------------------------------------------------------------------
    # 2) FPS.ply
    # ------------------------------------------------------------------
    fps_layer = vertices[sparse_orig_idx].copy()
    set_point_shape(fps_layer, base_sigma * args.fps_scale)
    set_rgb(fps_layer, CYAN)
    set_alpha(fps_layer, 0.995)
    write_ply(
        os.path.join(out_dir, "FPS.ply"),
        fps_layer,
        [
            "FPS sparse centers represented as tiny isotropic Gaussians",
            "Uniform cyan; original Gaussian colors/shapes removed",
            f"FPS count = {len(fps_layer)}",
        ],
    )

    # ------------------------------------------------------------------
    # 3) FPS_Dr.ply
    # ------------------------------------------------------------------
    fps_dr_layer = vertices[sparse_orig_idx].copy()
    set_point_shape(fps_dr_layer, base_sigma * args.fps_scale)
    set_rgb(fps_dr_layer, dr_colors(analysis["sparse_dr"], dr_lo, dr_hi))
    set_alpha(fps_dr_layer, 0.995)
    write_ply(
        os.path.join(out_dir, "FPS_Dr.ply"),
        fps_dr_layer,
        [
            "FPS centers represented as tiny isotropic Gaussians, colored by global Dr scale",
            "Original Gaussian colors/shapes removed",
        ],
    )

    # ------------------------------------------------------------------
    # 4) Apex.ply
    # ------------------------------------------------------------------
    apex_layer = vertices[apex_orig_idx].copy()
    set_point_shape(apex_layer, base_sigma * args.apex_scale)
    set_rgb(apex_layer, YELLOW)
    set_alpha(apex_layer, 0.999)
    root_layer = make_root_layer(vertices, root_orig_idx, base_sigma * args.root_scale)
    apex_records = np.concatenate([apex_layer, root_layer])
    write_ply(
        os.path.join(out_dir, "Apex.ply"),
        apex_records,
        [
            "Only tiny point-like Gaussians are shown; original Gaussian appearance removed",
            "Yellow = detected apexes",
            "Green large marker = root",
            "No apex grouping is run",
        ],
    )

    # ------------------------------------------------------------------
    # 5) DIJ.ply
    # ------------------------------------------------------------------
    dij_layer = vertices[keep_orig_idx].copy()
    set_point_shape(dij_layer, base_sigma)
    set_rgb(dij_layer, distance_colors_with_inf(dij))
    set_alpha(dij_layer, 0.995)
    root_layer = make_root_layer(vertices, root_orig_idx, base_sigma * args.root_scale)
    dij_records = np.concatenate([dij_layer, root_layer])
    write_ply(
        os.path.join(out_dir, "DIJ.ply"),
        dij_records,
        [
            "Dijkstra shortest-path distance heatmap from root",
            "Surface-neighborhood graph = union of edges from potpourri3d PointCloudLocalTriangulation",
            "Edge weight = Euclidean distance between Gaussian centers",
            "Dark gray = unreachable from root in this discrete graph",
            "Green large marker = root",
            "This is a discrete graph-geodesic baseline; it is not the Heat Method Dr field",
        ],
    )

    # ------------------------------------------------------------------
    # 6) FPS-apex.ply -- DIJ -> FPS -> LeafFit-style local maximum -> apex
    # ------------------------------------------------------------------
    # IMPORTANT FOR SUPERSPLAT:
    # Do NOT keep an ordinary cyan FPS splat at the same coordinate as a purple
    # apex splat. Two transparent Gaussian splats at identical centers can blend
    # or sort in a way that makes the purple marker look absent. Therefore the
    # DIJ apex samples are removed from the base FPS layer first.
    non_apex_mask = np.ones(len(sparse_orig_idx), dtype=bool)
    non_apex_mask[dij_apex_local] = False
    non_apex_orig_idx = sparse_orig_idx[non_apex_mask]

    fps_apex_layer = vertices[non_apex_orig_idx].copy()
    set_point_shape(fps_apex_layer, base_sigma * args.fps_scale)
    # Keep non-apex FPS points visually quiet so purple apexes dominate.
    set_rgb(fps_apex_layer, np.array([0.02, 0.48, 0.55], dtype=np.float32))
    set_alpha(fps_apex_layer, 0.72)

    dij_apex_layer = vertices[dij_apex_orig_idx].copy()
    set_point_shape(dij_apex_layer, base_sigma * args.dij_apex_scale)
    set_rgb(dij_apex_layer, PURPLE)
    set_alpha(dij_apex_layer, 0.9999)

    root_fps_apex = make_root_layer(vertices, root_orig_idx, base_sigma * args.root_scale)
    fps_apex_records = np.concatenate([
        fps_apex_layer,
        dij_apex_layer,
        root_fps_apex,
    ])
    write_ply(
        os.path.join(out_dir, "FPS-apex.ply"),
        fps_apex_records,
        [
            "DIJ -> spatial FPS -> LeafFit-style local maximum apex selection",
            "The original LeafFit idea is preserved; only Dr is replaced by DIJ for the apex test",
            "Small dark-cyan = non-apex FPS points",
            "Purple very-large = DIJ local-maximum apex",
            "Green large = root",
            "Purple apex samples are removed from the cyan FPS layer before overlay, avoiding duplicate-splat hiding in SuperSplat",
            f"tip-k = {dij_apex_analysis['query_k']}",
            f"DIJ apex count = {len(dij_apex_orig_idx)}",
            "Unreachable DIJ points are excluded from apex candidacy",
            "No apex grouping is run",
        ],
    )

    # ------------------------------------------------------------------
    # 7) DIJ-apex-path.ply
    #    Exact Dijkstra predecessor route root -> each DIJ-selected apex.
    # ------------------------------------------------------------------
    dij_path_records, dij_path_summaries = make_dijkstra_apex_path_layers(
        vertices=vertices,
        keep_orig_idx=keep_orig_idx,
        points=points,
        predecessors=dij_predecessor,
        root_filtered_idx=root_filtered_idx,
        dij_apex_filtered_idx=dij_apex_filtered_idx,
        base_sigma=base_sigma,
        median_spacing=median_spacing,
        args=args,
    )
    write_ply(
        os.path.join(out_dir, "DIJ-apex-path.ply"),
        dij_path_records,
        [
            "Exact Dijkstra predecessor paths from root to all DIJ-selected apexes",
            "Each apex path has a different bright color",
            "Large endpoint marker uses the same color as its path",
            "Green large marker = root",
            "Gray faint points = opacity-filtered plant-center context",
            "Paths come directly from scipy dijkstra return_predecessors on the DIJ surface-neighborhood graph",
            f"DIJ apex count = {len(dij_apex_filtered_idx)}",
            "No LeafFit path-k tracing is used in this file",
        ],
    )

    print("Recovered DIJ root->apex paths:", len(dij_path_summaries))
    for path_id, apex_fidx, n_vertices, path_length in dij_path_summaries:
        print(
            "  path_{:03d}: apex_filtered_idx={} vertices={} graph_length={:.6f}".format(
                path_id + 1,
                apex_fidx,
                n_vertices,
                path_length,
            )
        )

    # ------------------------------------------------------------------
    # 8) DIJ-shortest-tree.ply
    #    Exact predecessor tree selected by the root-to-all-points Dijkstra.
    # ------------------------------------------------------------------
    dij_tree_info = analyze_dijkstra_predecessor_tree(
        points=points,
        predecessors=dij_predecessor,
        root_idx=root_filtered_idx,
        median_spacing=median_spacing,
        long_factor=args.dij_long_edge_factor,
    )

    dij_tree_records = make_dijkstra_shortest_tree_layers(
        vertices=vertices,
        keep_orig_idx=keep_orig_idx,
        points=points,
        tree_info=dij_tree_info,
        root_filtered_idx=root_filtered_idx,
        base_sigma=base_sigma,
        median_spacing=median_spacing,
        args=args,
        long_only=False,
    )
    write_ply(
        os.path.join(out_dir, "DIJ-shortest-tree.ply"),
        dij_tree_records,
        [
            "Exact Dijkstra predecessor shortest-path tree from root",
            "Every reachable non-root Gaussian has one predecessor edge selected by scipy Dijkstra",
            "Cyan-blue = normal predecessor edges / tree vertices",
            "Red = suspiciously long predecessor edges",
            "Green = root",
            "Gray faint = opacity-filtered plant-center context",
            "Suspicious-long threshold = edge_length > factor * median nearest-neighbor spacing",
            f"factor = {args.dij_long_edge_factor}",
            f"threshold = {dij_tree_info['threshold']:.9f}",
            f"tree edge count = {len(dij_tree_info['length'])}",
            f"red long-edge count = {int(np.count_nonzero(dij_tree_info['long_mask']))}",
            "This tree is the exact predecessor structure used to recover DIJ root->point shortest paths",
        ],
    )

    # ------------------------------------------------------------------
    # 9) DIJ-long-edges.ply
    #    Only suspiciously long predecessor edges, for easy cross-leaf diagnosis.
    # ------------------------------------------------------------------
    dij_long_records = make_dijkstra_shortest_tree_layers(
        vertices=vertices,
        keep_orig_idx=keep_orig_idx,
        points=points,
        tree_info=dij_tree_info,
        root_filtered_idx=root_filtered_idx,
        base_sigma=base_sigma,
        median_spacing=median_spacing,
        args=args,
        long_only=True,
    )
    write_ply(
        os.path.join(out_dir, "DIJ-long-edges.ply"),
        dij_long_records,
        [
            "Only suspiciously long edges from the exact Dijkstra predecessor tree",
            "Red traces = suspicious predecessor edges",
            "Red large points = both endpoints of those edges",
            "Green = root",
            "Gray faint = plant-center context",
            f"factor = {args.dij_long_edge_factor}",
            f"threshold = {dij_tree_info['threshold']:.9f}",
            f"red long-edge count = {int(np.count_nonzero(dij_tree_info['long_mask']))}",
            "Length alone does not prove an edge is cross-leaf; this file is a diagnostic aid",
        ],
    )

    # ------------------------------------------------------------------
    # 10) apex_debug.ply
    # ------------------------------------------------------------------
    context_dbg = make_context_layer(vertices, keep_orig_idx, base_sigma, args.context_alpha)

    fps_dbg = vertices[sparse_orig_idx].copy()
    set_point_shape(fps_dbg, base_sigma * max(1.0, args.fps_scale * 0.75))
    set_rgb(fps_dbg, CYAN)
    set_alpha(fps_dbg, 0.85)

    apex_dbg = vertices[apex_orig_idx].copy()
    set_point_shape(apex_dbg, base_sigma * args.apex_scale)
    set_rgb(apex_dbg, YELLOW)
    set_alpha(apex_dbg, 0.999)

    reject_local = select_rejected_debug_points(analysis, args)
    reject_filtered_idx = sparse_idx[reject_local]
    reject_orig_idx = keep_orig_idx[reject_filtered_idx]
    reject_layer = vertices[reject_orig_idx].copy()
    set_point_shape(reject_layer, base_sigma * args.reject_scale)
    set_rgb(reject_layer, ORANGE)
    set_alpha(reject_layer, 0.999)

    killer_local = analysis["killer_local"][reject_local]
    valid = killer_local >= 0
    reject_local = reject_local[valid]
    reject_orig_idx_valid = keep_orig_idx[sparse_idx[reject_local]]
    killer_local = killer_local[valid]
    killer_orig_idx = keep_orig_idx[sparse_idx[killer_local]]

    killer_layer = vertices[killer_orig_idx].copy()
    set_point_shape(killer_layer, base_sigma * args.reject_scale)
    set_rgb(killer_layer, RED)
    set_alpha(killer_layer, 0.999)

    spacing = estimate_spacing(points)
    p0 = points[sparse_idx[reject_local]] if len(reject_local) else np.empty((0, 3))
    p1 = points[sparse_idx[killer_local]] if len(killer_local) else np.empty((0, 3))
    link_layer = make_link_point_records(
        vertices,
        reject_orig_idx_valid,
        p0,
        p1,
        sigma=base_sigma * 0.75,
        spacing=max(spacing * 0.35, 1e-8),
        rgb=MAGENTA,
    )

    root_dbg = make_root_layer(vertices, root_orig_idx, base_sigma * args.root_scale)
    debug_records = np.concatenate([
        context_dbg,
        fps_dbg,
        reject_layer,
        killer_layer,
        link_layer,
        apex_dbg,
        root_dbg,
    ])
    write_ply(
        os.path.join(out_dir, "apex_debug.ply"),
        debug_records,
        [
            "Gray tiny Gaussians = plant-center context; original colors and shapes removed",
            "Cyan = all FPS points",
            "Yellow large = detected apex",
            "Orange = rejected high-Dr FPS point close to passing apex test (debug selection only)",
            "Red = strongest higher-Dr KNN competitor of orange point",
            "Magenta tiny-Gaussian chain = orange rejected point -> red competitor",
            "Green large = root",
            f"tip-k = {analysis['query_k']}",
            "Apex rule = no queried FPS neighbor has larger Dr",
            "No apex grouping and no root-path construction are run",
        ],
    )

    print("=" * 80)
    print("DONE. Files:")
    for name in ("Dr.ply", "DIJ.ply", "FPS.ply", "FPS_Dr.ply", "Apex.ply", "FPS-apex.ply", "DIJ-apex-path.ply", "DIJ-shortest-tree.ply", "DIJ-long-edges.ply", "apex_debug.ply"):
        print("  ", os.path.join(out_dir, name))
    print("Debug reject links:", len(link_layer))
    print("=" * 80)


if __name__ == "__main__":
    main()
