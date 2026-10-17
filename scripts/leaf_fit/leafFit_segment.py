#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import colorsys
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


gm = types.ModuleType("gaussian_utils")
gm.GaussianData = GaussianData
sys.modules["gaussian_utils"] = gm


def args_parse():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--leaf-fit-root", default="third_party/leaf_fit")
    p.add_argument("--output", default=None)

    g = p.add_mutually_exclusive_group()
    g.add_argument("--root-index", type=int)
    g.add_argument("--pick-root", action="store_true")

    p.add_argument("--root-point-size", type=float, default=4.0)
    p.add_argument("--fps-points", type=int, default=8192)
    p.add_argument("--tip-k", type=int, default=None)
    p.add_argument("--path-k", type=int, default=None)
    p.add_argument("--root-radius", type=float, default=0.1)
    p.add_argument("--geometry-scale", type=float, default=1.0)
    p.add_argument(
        "--cut-sigma",
        type=float,
        default=2.0,
        help=(
            "Leaf-facing Base Gaussian support used for the hard cut. "
            "The cut plane is tangent to the Base Gaussian k-sigma ellipsoid "
            "instead of passing through its center. Default: 2.0."
        ),
    )
    p.add_argument(
        "--disable-apex-cluster",
        action="store_true",
        help=(
            "Disable LeafFit apex grouping. Each detected Apex is treated as "
            "one independent single-tip leaf candidate."
        ),
    )
    return p.parse_args()


def read_ply(path):
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    names = v.dtype.names or ()
    for k in ("x", "y", "z"):
        if k not in names:
            raise RuntimeError("PLY missing property: %s" % k)
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)
    return ply, xyz


def make_gaussians(xyz):
    n = len(xyz)
    return GaussianData(
        xyz=xyz.astype(np.float32, copy=False),
        rot=np.tile(np.array([[1, 0, 0, 0]], np.float32), (n, 1)),
        scale=np.ones((n, 3), np.float32),
        opacity=np.ones((n, 1), np.float32),
        sh=np.zeros((n, 3), np.float32),
        nxnynz=np.zeros((n, 3), np.float32),
        filter_3Ds=np.zeros((n, 1), np.float32),
    )


def rgb_from_ply(ply):
    v = ply["vertex"].data
    names = v.dtype.names or ()
    if not all(k in names for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
        return None
    sh = np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]])
    return np.clip(0.28209479177387814 * sh + 0.5, 0.0, 1.0)


def write_full_colored(ply, colors, path, mask=None):
    v = ply["vertex"].data.copy()
    names = v.dtype.names or ()
    if not all(k in names for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
        raise RuntimeError("PLY has no f_dc_0..2.")

    colors = np.clip(np.asarray(colors, np.float64), 0.0, 1.0)
    if mask is None:
        mask = np.ones(len(v), dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    fdc = (colors - 0.5) / 0.28209479177387814
    v["f_dc_0"][mask] = fdc[mask, 0]
    v["f_dc_1"][mask] = fdc[mask, 1]
    v["f_dc_2"][mask] = fdc[mask, 2]
    for name in names:
        if name.startswith("f_rest_"):
            v[name][mask] = 0

    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(path))


def write_subset_colored(ply, indices, colors, path):
    indices = np.asarray(indices, np.int64)
    colors = np.asarray(colors, np.float64)
    v = ply["vertex"].data[indices].copy()
    names = v.dtype.names or ()

    fdc = (np.clip(colors, 0.0, 1.0) - 0.5) / 0.28209479177387814
    v["f_dc_0"] = fdc[:, 0]
    v["f_dc_1"] = fdc[:, 1]
    v["f_dc_2"] = fdc[:, 2]
    for name in names:
        if name.startswith("f_rest_"):
            v[name] = 0

    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(path))


def dr_colors(dr):
    d = np.asarray(dr, np.float64)
    lo, hi = float(np.min(d)), float(np.max(d))
    t = np.zeros(len(d)) if hi <= lo else np.clip((d - lo) / (hi - lo), 0, 1)

    x = np.array([0, 0.25, 0.5, 0.75, 1.0])
    c = np.array([
        [0.05, 0.20, 1.00],
        [0.00, 0.90, 1.00],
        [0.10, 0.85, 0.20],
        [1.00, 0.90, 0.05],
        [1.00, 0.05, 0.05],
    ])
    out = np.empty((len(d), 3))
    for j in range(3):
        out[:, j] = np.interp(t, x, c[:, j])
    return out


def leaf_palette(n):
    out, h = [], 0.08
    for _ in range(n):
        h = (h + 0.618033988749895) % 1.0
        out.append(colorsys.hsv_to_rgb(h, 0.82, 0.96))
    return np.asarray(out)



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
        # Keep path colors bright and saturated.
        out.append(colorsys.hsv_to_rgb(h, 0.90, 1.00))
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
        raise RuntimeError("Open3D is required for --pick-root.")

    while True:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, np.float64))
        rgb = rgb_from_ply(ply)
        if rgb is not None:
            pcd.colors = o3d.utility.Vector3dVector(rgb)

        print("\nShift+左键选 Root，Shift+右键撤销，Q 结束。")
        vis = o3d.visualization.VisualizerWithEditing()
        if not vis.create_window("LeafFit - Pick Root", 1280, 900):
            raise RuntimeError("Open3D window creation failed.")
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
        print("Root candidate: %d, xyz=%s" % (idx, xyz[idx]))
        ans = input("确认使用？ [Y/n/r]: ").strip().lower()
        if ans in ("", "y", "yes"):
            return idx


def load_leaf_fit(
    core,
    tip_k,
    path_k,
    disable_apex_cluster=False,
):
    sys.path.insert(0, str(core))

    # Fix LeafFit's short-path argmin([]) edge case in memory.
    pp = core / "petiole_detection.py"
    s = pp.read_text(encoding="utf-8")
    pat = (
        r"(?P<i>[ \t]*)min_idx_after_protection\s*=\s*"
        r"np\.argmin\(selected_dists\[protection_period:\]\)"
        r"\s*\+\s*protection_period"
    )

    def repl(m):
        i = m.group("i")
        return (
            i + "search_start = min(protection_period, len(selected_dists) - 1)\n"
            + i + "tail_dists = selected_dists[search_start:]\n"
            + i + "min_idx_after_protection = np.argmin(tail_dists) + search_start"
        )

    s = re.sub(pat, repl, s, count=1)
    pm = types.ModuleType("petiole_detection")
    pm.__file__ = str(pp)
    pm.__package__ = ""
    sys.modules["petiole_detection"] = pm
    exec(compile(s, str(pp), "exec"), pm.__dict__)

    ap = core / "auto_segment.py"
    s = ap.read_text(encoding="utf-8")
    if "from __future__ import annotations" not in s:
        s = "from __future__ import annotations\n" + s

    s, a = re.subn(
        r"k\s*=\s*len\(sparse_indices\)\s*//\s*64",
        "k=(LEAFFIT_TIP_K if LEAFFIT_TIP_K is not None else len(sparse_indices)//64)",
        s, count=1,
    )
    s, b = re.subn(
        r"k\s*=\s*len\(sparse_indices\)\s*//\s*32",
        "k=(LEAFFIT_PATH_K if LEAFFIT_PATH_K is not None else len(sparse_indices)//32)",
        s, count=1,
    )
    if a != 1 or b != 1:
        raise RuntimeError("Could not patch LeafFit tip/path KNN expressions.")

    am = types.ModuleType("auto_segment")
    am.__file__ = str(ap)
    am.__package__ = ""
    am.LEAFFIT_TIP_K = tip_k
    am.LEAFFIT_PATH_K = path_k
    sys.modules["auto_segment"] = am
    exec(compile(s, str(ap), "exec"), am.__dict__)

    # --------------------------------------------------------------
    # Base selection override: cross-section WIDTH TREND.
    #
    # Keep LeafFit's original Da + Root/temperature-field construction
    # and the original source-call parameters. Only change:
    #
    #   (1) cross-section width:
    #       Wi = max_{u,v in Si} ||xu - xv||_2
    #
    #   (2) Base decision:
    #       find the first transition from a descending width trend
    #       to a continuously stable/flat width region.
    #
    # There is deliberately NO fixed "width drops to X%" condition.
    # --------------------------------------------------------------
    if not hasattr(am, "find_base_idx_by_geodesic_tip_graph"):
        raise RuntimeError(
            "LeafFit auto_segment.py has no find_base_idx_by_geodesic_tip_graph()."
        )

    _original_base_detector = am.find_base_idx_by_geodesic_tip_graph

    # Internal method constants. They are not exposed as command-line
    # parameters because they belong to the new Base detector itself.
    BASE_EQUAL_DA_SAMPLES = 48
    BASE_PRE_WINDOW = 4
    BASE_POST_WINDOW = 5

    # "Stable" is a trend criterion, not a width-size criterion.
    # The right-side slope must be much flatter than the preceding descent,
    # and the right-side local variation must stay small.
    BASE_FLAT_SLOPE_RATIO = 0.30
    BASE_STABLE_CV = 0.15
    BASE_DESCENDING_FRACTION = 0.67

    def _exact_farthest_pair_width(coords, block_size=256):
        """
        Exact cross-section diameter:
            max ||xi - xj||_2
        over all candidate points, computed in blocks to avoid storing
        one full NxN distance matrix.
        """
        coords = np.asarray(coords, dtype=np.float64)
        m = len(coords)
        if m < 2:
            return np.nan

        max_d2 = 0.0
        for i0 in range(0, m, block_size):
            a = coords[i0:i0 + block_size]
            diff = a[:, None, :] - coords[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", diff, diff)
            local = float(np.max(d2))
            if local > max_d2:
                max_d2 = local

        return float(np.sqrt(max_d2))

    def _equal_da_path_samples(path, da_field, max_samples):
        """
        Approximately equally spaced samples in Da along Apex->Root path.
        Each equal-Da target is mapped back to the nearest discrete path node.
        """
        path = [int(x) for x in path]
        if len(path) <= 2:
            return path

        values = np.asarray(
            [float(da_field[idx]) for idx in path],
            dtype=np.float64,
        )

        finite = np.isfinite(values)
        if not np.all(finite):
            path = [p for p, ok in zip(path, finite) if ok]
            values = values[finite]

        if len(path) <= 2:
            return path

        # Da should increase from Apex toward Root. Use a monotone envelope
        # only for resampling so small numerical reversals do not create
        # duplicated/backward samples.
        mono = np.maximum.accumulate(values)
        if float(mono[-1] - mono[0]) <= 1e-10:
            return path

        count = max(3, min(int(max_samples), len(path)))
        targets = np.linspace(mono[0], mono[-1], count)

        positions = []
        for target in targets:
            j = int(np.searchsorted(mono, target, side="left"))

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

    def _cross_section_candidates(
        path_idx,
        da_field,
        root_field,
        tolerance_percentage,
    ):
        """
        Construct the same conceptual cross-section as LeafFit:

            Da(q) ~= Da(path_idx)
            RootField(q) ~= RootField(path_idx)

        The source tolerance passed by LeafFit is reused directly.
        If fewer than two points survive, both tolerances are enlarged
        together a few times only to avoid a degenerate empty section.
        """
        da0 = float(da_field[path_idx])
        r0 = float(root_field[path_idx])

        da_range = float(np.nanmax(da_field) - np.nanmin(da_field))
        r_range = float(np.nanmax(root_field) - np.nanmin(root_field))

        da_floor = max(da_range * 1e-4, 1e-8)
        r_floor = max(r_range * 1e-4, 1e-8)

        for scale in (1.0, 1.5, 2.0, 3.0):
            da_tol = max(
                abs(da0) * float(tolerance_percentage) * scale,
                da_floor * scale,
            )
            r_tol = max(
                abs(r0) * float(tolerance_percentage) * scale,
                r_floor * scale,
            )

            mask = (
                (np.abs(da_field - da0) <= da_tol)
                & (np.abs(root_field - r0) <= r_tol)
            )
            idx = np.where(mask)[0]

            if len(idx) >= 2:
                return idx

        return np.empty((0,), dtype=np.int64)

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

    def _find_base_by_width_trend(
        cluster_info_item,
        gaussians,
        dense_solver,
        temperature_field,
        min_distance_threshold=0.05,
        tolerance_percentage=0.02,
        last_virtual_path_distance_factor=2.5,
        protection_period_ratio=0.25,
        debug_vis=False,
    ):
        tips = cluster_info_item.get("tips", [])
        path = [
            int(x)
            for x in cluster_info_item.get("potential_base_idx", [])
        ]

        final_tip = int(
            cluster_info_item.get(
                "final_tip",
                tips[0] if tips else (path[0] if path else 0),
            )
        )

        if not path:
            return final_tip

        # Keep LeafFit's valid search interval: Apex -> LCA/Junction.
        lca = cluster_info_item.get("lca", None)
        if lca is not None and int(lca) in path:
            path = path[:path.index(int(lca)) + 1]

        min_needed = BASE_PRE_WINDOW + BASE_POST_WINDOW + 1
        if len(path) < min_needed:
            return _original_base_detector(
                cluster_info_item,
                gaussians,
                dense_solver,
                temperature_field,
                min_distance_threshold=min_distance_threshold,
                tolerance_percentage=tolerance_percentage,
                last_virtual_path_distance_factor=(
                    last_virtual_path_distance_factor
                ),
                protection_period_ratio=protection_period_ratio,
                debug_vis=debug_vis,
            )

        da_field = dense_solver.compute_distance(final_tip)
        root_field = np.asarray(
            temperature_field,
            dtype=np.float64,
        )

        sampled_path = _equal_da_path_samples(
            path,
            da_field,
            BASE_EQUAL_DA_SAMPLES,
        )

        if len(sampled_path) < min_needed:
            return _original_base_detector(
                cluster_info_item,
                gaussians,
                dense_solver,
                temperature_field,
                min_distance_threshold=min_distance_threshold,
                tolerance_percentage=tolerance_percentage,
                last_virtual_path_distance_factor=(
                    last_virtual_path_distance_factor
                ),
                protection_period_ratio=protection_period_ratio,
                debug_vis=debug_vis,
            )

        widths = []
        candidate_counts = []

        for path_idx in sampled_path:
            candidate_idx = _cross_section_candidates(
                path_idx,
                da_field,
                root_field,
                tolerance_percentage,
            )

            candidate_counts.append(len(candidate_idx))

            if len(candidate_idx) < 2:
                widths.append(np.nan)
                continue

            coords = gaussians.xyz[candidate_idx]
            widths.append(_exact_farthest_pair_width(coords))

        widths = np.asarray(widths, dtype=np.float64)
        smooth = _median_smooth_widths(widths)

        finite_idx = np.where(np.isfinite(smooth))[0]
        if len(finite_idx) < min_needed:
            print(
                "    [Base-trend] tip=%d: too few valid widths; "
                "fallback to original Base detector." % final_tip
            )
            return _original_base_detector(
                cluster_info_item,
                gaussians,
                dense_solver,
                temperature_field,
                min_distance_threshold=min_distance_threshold,
                tolerance_percentage=tolerance_percentage,
                last_virtual_path_distance_factor=(
                    last_virtual_path_distance_factor
                ),
                protection_period_ratio=protection_period_ratio,
                debug_vis=debug_vis,
            )

        # A leaf normally grows wider after Apex and then narrows toward
        # the petiole. Therefore do not search for Base before the maximum
        # cross-section width.
        max_pos = int(finite_idx[np.argmax(smooth[finite_idx])])

        n = len(sampled_path)

        # Preserve LeafFit's own protected Apex-side fraction.
        protection = int(
            np.ceil(n * float(protection_period_ratio))
        )

        search_start = max(
            max_pos + 1,
            protection,
            BASE_PRE_WINDOW,
        )

        search_end = n - BASE_POST_WINDOW + 1

        chosen = None
        chosen_stats = None

        for i in range(search_start, search_end):
            before = smooth[i - BASE_PRE_WINDOW:i]
            after = smooth[i:i + BASE_POST_WINDOW]

            if (
                not np.all(np.isfinite(before))
                or not np.all(np.isfinite(after))
            ):
                continue

            # BEFORE Base: width should have a descending trend.
            before_slope = _normalized_slope(before)
            diffs_before = np.diff(before)
            descending_fraction = float(
                np.mean(diffs_before < 0.0)
            )

            if (
                not np.isfinite(before_slope)
                or before_slope >= 0.0
                or descending_fraction < BASE_DESCENDING_FRACTION
            ):
                continue

            # AFTER Base: the petiole width should become locally flat/stable.
            after_slope = _normalized_slope(after)
            after_mean = float(np.mean(after))

            if (
                not np.isfinite(after_slope)
                or after_mean <= 1e-10
            ):
                continue

            after_cv = float(
                np.std(after) / after_mean
            )

            # No absolute width or drop percentage is used here.
            # The right side only has to become substantially flatter than
            # the descending left side and remain locally stable.
            flatter_than_before = (
                abs(after_slope)
                <= BASE_FLAT_SLOPE_RATIO * abs(before_slope)
            )
            stable_after = after_cv <= BASE_STABLE_CV

            # A plateau should not jump back above the preceding local level.
            # This is only an ordering check, not a percentage threshold.
            not_rebounding = (
                float(np.median(after))
                <= float(np.median(before))
            )

            if (
                flatter_than_before
                and stable_after
                and not_rebounding
            ):
                chosen = i
                chosen_stats = (
                    before_slope,
                    after_slope,
                    descending_fraction,
                    after_cv,
                )
                break

        if chosen is not None:
            base_idx = int(sampled_path[chosen])
            before_slope, after_slope, desc_frac, after_cv = chosen_stats

            print(
                "    [Base-trend] tip=%d base=%d sample=%d/%d "
                "Wmax@%d slope_before=%.5f slope_after=%.5f "
                "descending=%.2f stable_cv=%.3f candidates=%d"
                % (
                    final_tip,
                    base_idx,
                    chosen,
                    n,
                    max_pos,
                    before_slope,
                    after_slope,
                    desc_frac,
                    after_cv,
                    candidate_counts[chosen],
                )
            )
            return base_idx

        print(
            "    [Base-trend] tip=%d: no descending->stable transition; "
            "fallback to original geodesic_tip_graph Base detector."
            % final_tip
        )

        return _original_base_detector(
            cluster_info_item,
            gaussians,
            dense_solver,
            temperature_field,
            min_distance_threshold=min_distance_threshold,
            tolerance_percentage=tolerance_percentage,
            last_virtual_path_distance_factor=(
                last_virtual_path_distance_factor
            ),
            protection_period_ratio=protection_period_ratio,
            debug_vis=debug_vis,
        )

    am.find_base_idx_by_geodesic_tip_graph = _find_base_by_width_trend

    # --------------------------------------------------------------
    # Optional per-tip mode:
    # bypass LeafFit's apex clustering while preserving the rest of
    # get_segment_mask(), Base detection and leaf extraction.
    #
    # We replace the function in the loaded module at runtime, so the
    # third_party/leaf_fit source on disk is not modified.
    # --------------------------------------------------------------
    _original_group_apexes = am.group_apexes_by_inequality

    def _group_apexes_per_tip(
        found_tips,
        pathes,
        root_cahced_distance=None,
        overlap_cut=0.8,
        triangle_cut=0.62,
        slack_eps=1e-9,
        max_iters=50,
        dense_solver=None,
    ):
        if not disable_apex_cluster:
            return _original_group_apexes(
                found_tips,
                pathes,
                root_cahced_distance=root_cahced_distance,
                overlap_cut=overlap_cut,
                triangle_cut=triangle_cut,
                slack_eps=slack_eps,
                max_iters=max_iters,
                dense_solver=dense_solver,
            )

        cluster_info = []
        for tip, path in zip(found_tips, pathes):
            tip = int(tip)
            path = [int(x) for x in path]

            # For a singleton Apex, use the path root as the initial LCA.
            # get_segment_mask() will subsequently recompute the earliest
            # intersection between independent paths before Base detection.
            lca = int(path[-1]) if path else tip

            if root_cahced_distance is not None and len(path):
                tip_to_lca = float(
                    root_cahced_distance[tip]
                    - root_cahced_distance[lca]
                )
            else:
                tip_to_lca = 0.0

            cluster_info.append({
                "tips": [tip],
                "tips_dist_to_lca": {tip: tip_to_lca},
                "final_tip": tip,
                "lca": lca,
                "potential_base_idx": path,
                "pathes": [path],
                "path": path,
                "type": "single_tip",
            })

        return cluster_info

    am.group_apexes_by_inequality = _group_apexes_per_tip

    return am


def junction_indices(clusters, fn, n):
    paths = [
        [int(x) for x in c.get("path", [])]
        for c in clusters if c.get("path", [])
    ]
    if len(paths) < 2:
        return np.empty(0, np.int64)

    details = fn(paths, forbid_tip=True, return_detail=True)
    nodes = [
        int(d["node"]) for d in details
        if d.get("node") is not None and 0 <= int(d["node"]) < n
    ]
    return np.unique(np.asarray(nodes, np.int64)) if nodes else np.empty(0, np.int64)


def save_structure(
    ply,
    xyz,
    root_idx,
    tips,
    bases,
    junctions,
    path,
):
    """
    High-visibility keypoint-only debug visualization.

    debug/02_structure_colored_gaussians.ply:
      magenta = Apex
      blue    = Base
      yellow  = Junction
      red     = Root

    Apex->Root paths are intentionally NOT visualized.

    Non-keypoint plant Gaussians are dark and nearly transparent so the
    structural keypoints are easy to inspect.

    Base/Junction are exact single Gaussian indices, not KNN-expanded.
    """
    n = len(xyz)

    # Dark background for the whole plant.
    colors = np.full((n, 3), 0.10, dtype=np.float64)
    changed = np.ones(n, dtype=bool)

    # Exact structural keypoints.
    aidx = flatten(tips, n)
    bidx = flatten(bases, n)
    jidx = flatten(junctions, n)

    colors[jidx] = [1.00, 0.85, 0.00]        # Junction
    colors[bidx] = [0.05, 0.25, 1.00]        # Base
    colors[aidx] = [1.00, 0.00, 1.00]        # Apex
    colors[int(root_idx)] = [1.00, 0.05, 0.05]  # Root

    write_full_colored(ply, colors, path, changed)

    # Make only keypoints strongly visible in this DEBUG PLY.
    dbg = PlyData.read(str(path))
    v = dbg["vertex"].data.copy()
    names = v.dtype.names or ()

    visible = np.zeros(n, dtype=bool)
    visible[aidx] = True
    visible[bidx] = True
    visible[jidx] = True
    visible[int(root_idx)] = True

    if "opacity" in names:
        # Background plant is only a faint spatial reference.
        v["opacity"][~visible] = -5.0   # sigmoid ~= 0.0067
        v["opacity"][visible] = 5.0     # sigmoid ~= 0.993

    # In the debug file only, make keypoint Gaussians small/isotropic so one
    # anisotropic Gaussian does not look like a long colored "region".
    if all(k in names for k in ("scale_0", "scale_1", "scale_2")):
        key_idx = np.unique(
            np.concatenate([
                aidx,
                bidx,
                jidx,
                np.asarray([int(root_idx)], dtype=np.int64),
            ])
        )
        if len(key_idx):
            logs = np.column_stack([
                v["scale_0"][key_idx],
                v["scale_1"][key_idx],
                v["scale_2"][key_idx],
            ]).astype(np.float64)

            median_log = np.median(logs, axis=1)
            marker_log = np.minimum(median_log - 1.25, -2.5)

            v["scale_0"][key_idx] = marker_log
            v["scale_1"][key_idx] = marker_log
            v["scale_2"][key_idx] = marker_log

    PlyData(
        [PlyElement.describe(v, "vertex")],
        text=False,
    ).write(str(path))

    print("      structure debug = keypoints only")
    print("      magenta=Apex, blue=Base, yellow=Junction, red=Root")
    print("      Apex->Root paths are NOT visualized")


def _quat_wxyz_to_matrix(q):
    q = np.asarray(q, np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def base_gaussian_projected_sigma(ply, base_idx, direction):
    """
    Project the anisotropic Base Gaussian covariance onto 'direction'.

    Standard 3DGS / PGSR PLY convention:
      scale_0..2 store log standard deviations
      rot_0..3 store quaternion (w, x, y, z)

    sigma(direction) = sqrt(direction^T Sigma direction)

    Returns None when the required Gaussian fields are unavailable.
    """
    v = ply["vertex"].data
    names = v.dtype.names or ()
    scale_names = ("scale_0", "scale_1", "scale_2")
    rot_names = ("rot_0", "rot_1", "rot_2", "rot_3")

    if not all(k in names for k in scale_names + rot_names):
        return None

    log_s = np.array([float(v[k][base_idx]) for k in scale_names])
    s = np.exp(log_s)

    q = np.array([float(v[k][base_idx]) for k in rot_names])
    R = _quat_wxyz_to_matrix(q)

    # Sigma = R diag(s^2) R^T.
    # More stable/economical form for the projected standard deviation:
    local_dir = np.dot(R.T, np.asarray(direction, np.float64))
    sigma = float(np.sqrt(np.sum((s * local_dir) ** 2)))

    if not np.isfinite(sigma) or sigma <= 0:
        return None
    return sigma


def base_plane(xyz, ply, cluster, root_idx, cut_sigma):
    """
    Construct an Apex-facing support plane of the detected Base Gaussian.

    Old version:
        plane passed through Base Gaussian center.

    New version:
        1) estimate local Apex->Root path direction n near Base;
        2) compute the Base Gaussian projected sigma along n;
        3) move the plane from the Gaussian center toward the Apex by
           cut_sigma * sigma(n);
        4) use a plane perpendicular to n.

    Geometrically this is a tangent/support plane to the finite k-sigma
    ellipsoid of the Base Gaussian in the Apex direction.
    """
    n_points = len(xyz)
    tip = int(cluster["selected_tip"])
    base = int(cluster["base_idx"])
    path = [int(x) for x in cluster.get("path", []) if 0 <= int(x) < n_points]

    if not (0 <= tip < n_points and 0 <= base < n_points):
        return None

    # Local path tangent around Base.
    if len(path) >= 2:
        if base in path:
            pos = path.index(base)
        else:
            q = xyz[np.asarray(path)]
            pos = int(np.argmin(np.linalg.norm(q - xyz[base], axis=1)))

        # Small symmetric window suppresses one-edge noise.
        i0 = max(0, pos - 2)
        i1 = min(len(path) - 1, pos + 2)
        normal = (
            xyz[path[i1]].astype(np.float64)
            - xyz[path[i0]].astype(np.float64)
        )
    else:
        normal = (
            xyz[root_idx].astype(np.float64)
            - xyz[tip].astype(np.float64)
        )

    norm = float(np.linalg.norm(normal))
    if norm < 1e-10:
        normal = (
            xyz[root_idx].astype(np.float64)
            - xyz[base].astype(np.float64)
        )
        norm = float(np.linalg.norm(normal))
    if norm < 1e-10:
        return None
    normal /= norm

    base_center = xyz[base].astype(np.float64)
    apex_signed_from_center = float(
        np.dot(xyz[tip].astype(np.float64) - base_center, normal)
    )
    if abs(apex_signed_from_center) < 1e-10:
        return None

    # Direction from Base center toward Apex along the path normal.
    apex_sign = 1.0 if apex_signed_from_center > 0 else -1.0

    # Use the actual PGSR Gaussian scale+rotation to locate its Apex-facing edge.
    sigma_along_path = base_gaussian_projected_sigma(ply, base, normal)
    if sigma_along_path is None:
        # Safe fallback: old center plane if PLY fields are unavailable.
        offset = 0.0
    else:
        offset = float(cut_sigma) * sigma_along_path

    # This is the key change:
    # cut plane is moved from Base center to its Apex-facing support boundary.
    cut_center = base_center + apex_sign * offset * normal

    apex_side = float(
        np.dot(xyz[tip].astype(np.float64) - cut_center, normal)
    )

    # If a very large cut_sigma moved past the Apex, fall back to center plane.
    if abs(apex_side) < 1e-10 or apex_side * apex_signed_from_center <= 0:
        cut_center = base_center
        apex_side = apex_signed_from_center
        offset = 0.0

    return base, tip, cut_center, normal, apex_side, offset


def cut_at_base(xyz, ply, candidate, cluster, root_idx, cut_sigma):
    """
    Keep only LeafFit candidate Gaussian centers on the Apex side of the
    Apex-facing Base-Gaussian support plane.

    The Base Gaussian itself is always removed.
    """
    candidate = np.unique(np.asarray(candidate, np.int64))
    if len(candidate) == 0:
        return candidate, 0.0

    plane = base_plane(
        xyz,
        ply,
        cluster,
        root_idx,
        cut_sigma,
    )
    if plane is None:
        return candidate, 0.0

    base, tip, cut_center, normal, apex_side, offset = plane
    signed = np.dot(
        xyz[candidate].astype(np.float64) - cut_center,
        normal,
    )

    # Keep only the leaf/Apex half-space.
    cut = candidate[signed * apex_side > 1e-10]
    cut = cut[cut != base]

    if tip in candidate and tip not in cut:
        cut = np.unique(
            np.concatenate([cut, np.asarray([tip], np.int64)])
        )

    return cut, float(offset)



def clean_output(out):
    for name in ("debug", "semantic", "instances", "instances_colored", "indices"):
        p = out / name
        if p.exists():
            shutil.rmtree(str(p))
    p = out / "segmentation.npz"
    if p.exists():
        p.unlink()


def main():
    args = args_parse()
    if args.tip_k is not None and args.tip_k < 2:
        raise ValueError("--tip-k must be >= 2")
    if args.path_k is not None and args.path_k < 2:
        raise ValueError("--path-k must be >= 2")

    inp = Path(args.input).resolve()
    core = Path(args.leaf_fit_root).resolve() / "core"
    out = Path(args.output).resolve() if args.output else inp.parent / "leaf_fit_segments"

    clean_output(out)
    debug = out / "debug"
    semantic = out / "semantic"
    debug.mkdir(parents=True, exist_ok=True)
    semantic.mkdir(parents=True, exist_ok=True)

    print("[1/6] Read PLY")
    ply, xyz = read_ply(inp)
    n = len(xyz)
    gaussians = make_gaussians(xyz * float(args.geometry_scale))

    print("[2/6] Load LeafFit + select Root")
    auto = load_leaf_fit(
        core,
        args.tip_k,
        args.path_k,
        disable_apex_cluster=args.disable_apex_cluster,
    )

    root_input = args.root_index
    if args.pick_root:
        root_input = pick_root(xyz, ply, args.root_point_size)

    gaussians, root_idx, solver = auto.fix_plant_root_direction_legacy(
        gaussians, opacity_threshold=0.0, given_root_idx=root_input
    )
    root_idx = int(root_idx)
    print("      root_idx =", root_idx, "xyz =", xyz[root_idx])

    dr0 = solver.compute_distance(root_idx)
    sources = np.where(dr0 <= float(args.root_radius))[0].astype(np.int64)
    if len(sources) == 0:
        sources = np.asarray([root_idx], np.int64)
    Dr = solver.compute_distance_multisource(sources)

    print("[3/6] FPS + segmentation")
    ns = min(int(args.fps_points), n)
    sparse = fpsample.bucket_fps_kdline_sampling(
        gaussians.xyz, ns, h=7, start_idx=0
    ).astype(np.int64)

    nn = NearestNeighbors(n_neighbors=1).fit(gaussians.xyz[sparse])
    _, m = nn.kneighbors(gaussians.xyz)
    mapping = m.flatten().astype(np.int64)
    tree = cKDTree(gaussians.xyz)

    print("      sparse =", len(sparse))
    print("      tip-k =", args.tip_k if args.tip_k is not None else len(sparse)//64)
    print("      path-k =", args.path_k if args.path_k is not None else len(sparse)//32)
    print(
        "      apex clustering =",
        "DISABLED (1 Apex = 1 leaf candidate)"
        if args.disable_apex_cluster
        else "LeafFit default",
    )
    print(
        "      Base selection = cross-section width trend "
        "(descending -> stable)"
    )

    masks, tips, bases, _, clusters, _ = auto.get_segment_mask(
        gaussians,
        sparse,
        mapping,
        solver,
        tree,
        sources.tolist(),
        "geodesic_tip_graph",
        cached_root_distances=Dr,
        debug_vis=False,
    )
    print("      leaf clusters =", len(masks))
    print("      detected Apex =", len(tips))

    print("[4/6] Save debug PLYs")
    write_full_colored(
        ply,
        dr_colors(Dr),
        debug / "01_Dr_colored_gaussians.ply",
    )
    junctions = junction_indices(
        clusters,
        auto.find_earliest_intersection_indices,
        n,
    )
    save_structure(
        ply,
        xyz,
        root_idx,
        tips,
        bases,
        junctions,
        debug / "02_structure_colored_gaussians.ply",
    )

    print("[5/6] Base-Gaussian boundary cut")
    print("      cut-sigma =", args.cut_sigma)
    cut_instances = []
    for i in range(min(len(masks), len(clusters))):
        candidate = np.unique(np.asarray(masks[i], np.int64))
        cut, plane_offset = cut_at_base(
            xyz,
            ply,
            candidate,
            clusters[i],
            root_idx,
            args.cut_sigma,
        )
        print(
            "      leaf_%03d: %d -> %d, cut_offset=%.6f"
            % (i + 1, len(candidate), len(cut), plane_offset)
        )
        if len(cut):
            cut_instances.append(cut)

    if not cut_instances:
        raise RuntimeError("No cut leaf instances produced.")

    print("[6/6] Save colored leaf-only PLY")
    palette = leaf_palette(len(cut_instances))
    used = set()
    all_idx, all_color = [], []

    for idx, color in zip(cut_instances, palette):
        keep = []
        for x in idx:
            x = int(x)
            if x not in used:
                used.add(x)
                keep.append(x)
        if not keep:
            continue

        keep = np.asarray(keep, np.int64)
        all_idx.append(keep)
        all_color.append(np.tile(color[None, :], (len(keep), 1)))

    write_subset_colored(
        ply,
        np.concatenate(all_idx),
        np.concatenate(all_color, axis=0),
        semantic / "leaves_cut_instance_colored.ply",
    )

    print("\nDone. Only keep:")
    print("  debug/01_Dr_colored_gaussians.ply")
    print("  debug/02_structure_colored_gaussians.ply")
    print("  semantic/leaves_cut_instance_colored.ply")
    print("\n02_structure legend:")
    print("  background plant = dark + nearly transparent")
    print("  magenta=Apex, blue=Base, yellow=Junction, red=Root")
    print("  Apex->Root paths are NOT output in this debug file")
    print(
        "  apex clustering =",
        "disabled"
        if args.disable_apex_cluster
        else "LeafFit default",
    )


if __name__ == "__main__":
    main()