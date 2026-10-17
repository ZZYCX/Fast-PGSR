#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
render_ortho.py
===============

Fast-PGSR Gaussian PLY -> multi-view orthographic renderer.

What this version outputs by default
------------------------------------
1) Top-down sequence (16 by default):
   - topdown_00_rgb.png ... topdown_15_rgb.png
   Meaning: progressively clip away the plant from top to bottom.

2) Upper-side ring views (8 by default):
   - upper_00_rgb.png ... upper_07_rgb.png
   Meaning: 8 orthographic oblique views around the plant, looking slightly downward.

3) Side ring views (8 by default):
   - side_00_rgb.png ... side_07_rgb.png
   Meaning: 8 orthographic pure side views around the plant.

Optional:
4) full_top_rgb.png (enabled by default, can be disabled with --no-full-top)

Default output folders
----------------------
render/rgb/
render/aux_data/   (only if --save-aux is provided)

Key interpretation
------------------
- "topdown" sequence:
  the camera conceptually moves downward along the height axis,
  so frame i keeps only Gaussians with height <= cutoff_i.

- "upper" and "side":
  full-plant orthographic renders from additional angles around the plant.

Recommended run
---------------
python scripts/render/render_ortho.py \
  --input output/mungbean2_0804/target_exports/plant_only/point_cloud.ply \
  --steps 16 \
  --num-upper-views 8 \
  --num-side-views 8 \
  --height-axis z \
  --width 1024 \
  --height 1024
"""

import argparse
import json
import math
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Python 3.7 compatibility for gsplat 1.5.3
# Must be defined BEFORE importing gsplat.
# ----------------------------------------------------------------------
if not hasattr(math, "prod"):
    def _math_prod(iterable, start=1):
        result = start
        for value in iterable:
            result *= value
        return result
    math.prod = _math_prod

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.gaussian_model import GaussianModel
from utils.sh_utils import SH2RGB

try:
    from gsplat.rendering import rasterization as gsplat_rasterization
except Exception as exc:
    raise ImportError(
        "Failed to import gsplat. Make sure third_party/gsplat is installed with "
        "`pip install -e . --no-build-isolation --no-deps`.\n"
        "Original error: {}".format(exc)
    )

EPS = 1e-8


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < EPS:
        raise ValueError("Cannot normalize near-zero vector: {}".format(v))
    return v / n


def axis_index(axis):
    axis = axis.lower()
    table = {"x": 0, "y": 1, "z": 2}
    if axis not in table:
        raise ValueError("height axis must be one of x/y/z, got {!r}".format(axis))
    return table[axis]


def infer_sh_degree(ply_path):
    ply = PlyData.read(str(ply_path))
    props = [p.name for p in ply.elements[0].properties]

    required = {
        "x", "y", "z",
        "opacity",
        "f_dc_0", "f_dc_1", "f_dc_2",
    }
    missing = sorted(required - set(props))
    scale_names = [p for p in props if p.startswith("scale_")]
    rot_names = [p for p in props if p.startswith("rot")]
    rest_names = [p for p in props if p.startswith("f_rest_")]

    if missing or len(scale_names) != 3 or len(rot_names) != 4:
        raise ValueError(
            "Input does not look like a full Gaussian PLY. "
            "missing={}, scale_fields={}, rot_fields={}".format(
                missing, len(scale_names), len(rot_names)
            )
        )

    coeffs_per_channel = (len(rest_names) + 3) / 3.0
    degree = int(round(math.sqrt(coeffs_per_channel) - 1))
    expected_rest = 3 * (degree + 1) ** 2 - 3

    if expected_rest != len(rest_names):
        raise ValueError(
            "Cannot infer SH degree safely: found {} f_rest_* fields, but "
            "degree={} would require {}".format(
                len(rest_names), degree, expected_rest
            )
        )
    return degree


def load_gaussians(ply_path):
    sh_degree = infer_sh_degree(ply_path)
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(str(ply_path))

    if hasattr(gaussians, "active_sh_degree"):
        try:
            gaussians.active_sh_degree = sh_degree
        except Exception:
            pass

    n = int(gaussians.get_xyz.shape[0])
    if n == 0:
        raise ValueError("Loaded zero Gaussians from {}".format(ply_path))

    print("[Load] SH degree={}, Gaussians={}".format(sh_degree, n))
    return gaussians, sh_degree


def bbox_from_xyz(xyz):
    arr = xyz.detach().cpu().numpy()
    return arr.min(axis=0), arr.max(axis=0)


def bbox_corners(bmin, bmax):
    corners = []
    for x in (bmin[0], bmax[0]):
        for y in (bmin[1], bmax[1]):
            for z in (bmin[2], bmax[2]):
                corners.append([x, y, z])
    return np.asarray(corners, dtype=np.float64)


def print_dc_color_diagnostics(gaussians):
    dc = gaussians.get_features_dc
    if dc.ndim == 3:
        dc = dc[:, 0, :]
    rgb = SH2RGB(dc).detach().float().cpu().numpy()
    rgb = np.clip(rgb, 0.0, 1.0)

    p05 = np.percentile(rgb, 5, axis=0)
    p50 = np.percentile(rgb, 50, axis=0)
    p95 = np.percentile(rgb, 95, axis=0)

    delta = np.mean(
        np.abs(rgb[:, 0] - rgb[:, 1]) +
        np.abs(rgb[:, 1] - rgb[:, 2])
    )

    print(
        "[Color/DC] p05={}, p50={}, p95={}, channel_delta={:.6f}".format(
            np.round(p05, 3),
            np.round(p50, 3),
            np.round(p95, 3),
            float(delta),
        )
    )
    if delta < 1e-4:
        print("[WARNING] DC color is almost grayscale. If RGB output is gray, "
              "the exported PLY itself may already have lost plant color.")


def world_up_vector(height_axis):
    basis = np.eye(3, dtype=np.float64)
    return basis[axis_index(height_axis)]


def choose_up_reference(forward, preferred_up):
    f = normalize(forward)
    u = normalize(preferred_up)
    if abs(np.dot(f, u)) < 0.95:
        return u

    # pick a fallback axis not parallel to forward
    basis = np.eye(3, dtype=np.float64)
    best = None
    best_score = None
    for cand in basis:
        score = abs(np.dot(f, cand))
        if best is None or score < best_score:
            best = cand
            best_score = score
    return normalize(best)


def make_ortho_camera_from_forward(
    bmin,
    bmax,
    forward_world,
    width,
    height,
    margin,
    device,
    preferred_up_world,
):
    """
    Generic orthographic camera that fully covers the whole plant bbox for the given
    viewing direction.
    """
    forward = normalize(forward_world)
    up_ref = choose_up_reference(forward, preferred_up_world)

    right = normalize(np.cross(forward, up_ref))
    true_up = normalize(np.cross(right, forward))
    down = -true_up

    center = 0.5 * (bmin + bmax)
    corners = bbox_corners(bmin, bmax)
    rel = corners - center[None, :]

    x_vals = rel @ right
    y_vals = rel @ down
    d_vals = rel @ forward

    span_x = max(float(x_vals.max() - x_vals.min()), EPS)
    span_y = max(float(y_vals.max() - y_vals.min()), EPS)
    depth_half = max(abs(float(d_vals.min())), abs(float(d_vals.max())), EPS)

    usable_w = max(width * (1.0 - 2.0 * margin), 1.0)
    usable_h = max(height * (1.0 - 2.0 * margin), 1.0)
    pixels_per_unit = min(usable_w / span_x, usable_h / span_y)

    K = np.array(
        [
            [pixels_per_unit, 0.0, width * 0.5],
            [0.0, pixels_per_unit, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    scene_extent = max(float(np.max(bmax - bmin)), EPS)
    camera_distance = depth_half + 3.0 * scene_extent + 1e-3
    eye = center - forward * camera_distance

    R = np.stack([right, down, forward], axis=0)
    t = -R @ eye

    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R.astype(np.float32)
    viewmat[:3, 3] = t.astype(np.float32)

    near_plane = max(1e-4, camera_distance - depth_half - 2.0 * scene_extent)
    far_plane = camera_distance + depth_half + 2.0 * scene_extent

    return {
        "viewmat": torch.from_numpy(viewmat).to(device=device),
        "K": torch.from_numpy(K).to(device=device),
        "eye_world": eye.astype(np.float32),
        "center_world": center.astype(np.float32),
        "forward_world": forward.astype(np.float32),
        "right_world": right.astype(np.float32),
        "up_world": true_up.astype(np.float32),
        "pixels_per_unit": float(pixels_per_unit),
        "near_plane": float(near_plane),
        "far_plane": float(far_plane),
        "width": int(width),
        "height": int(height),
        "margin": float(margin),
    }


def make_top_ortho_camera(bmin, bmax, height_axis, width, height, margin, device):
    h_axis = world_up_vector(height_axis)
    forward = -h_axis
    preferred_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return make_ortho_camera_from_forward(
        bmin=bmin,
        bmax=bmax,
        forward_world=forward,
        width=width,
        height=height,
        margin=margin,
        device=device,
        preferred_up_world=preferred_up,
    )


def make_ring_direction(height_axis, azim_deg, elev_deg):
    """
    Build a camera forward vector:
    - azimuth rotates around the height axis
    - elevation:
        0   -> pure horizontal side view
        >0  -> camera is above the horizon and looks downward
    """
    hi = axis_index(height_axis)
    basis = np.eye(3, dtype=np.float64)
    h_axis = basis[hi]

    others = [i for i in range(3) if i != hi]
    u = basis[others[0]]
    v = basis[others[1]]

    az = np.deg2rad(float(azim_deg))
    el = np.deg2rad(float(elev_deg))

    # Horizontal direction in the plane perpendicular to height axis.
    horizontal = math.cos(az) * u + math.sin(az) * v
    horizontal = normalize(horizontal)

    # Forward points from camera to object center.
    # elev = 0   -> horizontal view
    # elev > 0   -> downward-looking oblique view
    forward = math.cos(el) * horizontal - math.sin(el) * h_axis
    return normalize(forward)


def camera_to_json(camera):
    return {
        "viewmat_world_to_camera": camera["viewmat"].detach().cpu().numpy().tolist(),
        "K_orthographic": camera["K"].detach().cpu().numpy().tolist(),
        "eye_world": camera["eye_world"].tolist(),
        "center_world": camera["center_world"].tolist(),
        "forward_world": camera["forward_world"].tolist(),
        "right_world": camera["right_world"].tolist(),
        "up_world": camera["up_world"].tolist(),
        "pixels_per_world_unit": float(camera["pixels_per_unit"]),
        "near_plane": float(camera["near_plane"]),
        "far_plane": float(camera["far_plane"]),
        "width": int(camera["width"]),
        "height": int(camera["height"]),
        "margin": float(camera["margin"]),
    }


def build_topdown_planes(xyz, height_axis, steps):
    ai = axis_index(height_axis)
    h = xyz[:, ai]
    hmin = float(h.min().item())
    hmax = float(h.max().item())
    extent = hmax - hmin

    if extent <= EPS:
        raise ValueError(
            "Scene has almost zero extent along {}: [{}, {}]".format(
                height_axis, hmin, hmax
            )
        )

    dh = extent / float(steps)

    planes = []
    for i in range(steps):
        cutoff = hmax - i * dh
        planes.append({
            "index": i,
            "camera_plane": float(cutoff),
            "keep_below_or_equal": float(cutoff),
            "removed_from_top": float(i * dh),
            "remaining_height_from_bottom": float(cutoff - hmin),
        })
    return hmin, hmax, dh, planes


def select_below_plane(xyz, height_axis, cutoff):
    ai = axis_index(height_axis)
    h = xyz[:, ai]
    keep = (h <= cutoff)
    return torch.nonzero(keep, as_tuple=False).squeeze(1)


@torch.no_grad()
def render_rgb(
    gaussians,
    indices,
    camera,
    sh_degree,
    background_rgb,
    color_mode="sh",
):
    H = int(camera["height"])
    W = int(camera["width"])

    if indices.numel() == 0:
        bg = np.asarray(background_rgb, dtype=np.float32)
        rgb = np.broadcast_to(bg[None, None, :], (H, W, 3)).copy()
        alpha = np.zeros((H, W), dtype=np.float32)
        return rgb, alpha

    means = gaussians.get_xyz[indices].contiguous()
    quats = gaussians.get_rotation[indices].contiguous()
    scales = gaussians.get_scaling[indices].contiguous()
    opacities = gaussians.get_opacity[indices].squeeze(-1).contiguous()

    if color_mode == "sh":
        colors = gaussians.get_features[indices].contiguous()
        use_sh_degree = int(sh_degree)
    elif color_mode == "dc":
        dc = gaussians.get_features_dc[indices]
        if dc.ndim == 3:
            dc = dc[:, 0, :]
        colors = torch.clamp(SH2RGB(dc), 0.0, 1.0).contiguous()
        use_sh_degree = None
    else:
        raise ValueError("color_mode must be 'sh' or 'dc'")

    backgrounds = torch.tensor(
        [background_rgb],
        dtype=means.dtype,
        device=means.device,
    )

    rendered, alpha, _meta = gsplat_rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=camera["viewmat"][None, ...],
        Ks=camera["K"][None, ...],
        width=W,
        height=H,
        near_plane=float(camera["near_plane"]),
        far_plane=float(camera["far_plane"]),
        sh_degree=use_sh_degree,
        backgrounds=backgrounds,
        render_mode="RGB",
        camera_model="ortho",
        packed=False,
        rasterize_mode="classic",
    )

    rgb = rendered[0, ..., :3].clamp(0.0, 1.0)
    alpha = alpha[0, ..., 0].clamp(0.0, 1.0)

    return (
        rgb.detach().cpu().numpy().astype(np.float32),
        alpha.detach().cpu().numpy().astype(np.float32),
    )


@torch.no_grad()
def render_expected_depth(
    gaussians,
    indices,
    camera,
):
    H = int(camera["height"])
    W = int(camera["width"])

    if indices.numel() == 0:
        return np.zeros((H, W), dtype=np.float32)

    means = gaussians.get_xyz[indices].contiguous()
    quats = gaussians.get_rotation[indices].contiguous()
    scales = gaussians.get_scaling[indices].contiguous()
    opacities = gaussians.get_opacity[indices].squeeze(-1).contiguous()

    dummy = torch.zeros(
        (means.shape[0], 1),
        dtype=means.dtype,
        device=means.device,
    )

    rendered, _alpha, _meta = gsplat_rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=dummy,
        viewmats=camera["viewmat"][None, ...],
        Ks=camera["K"][None, ...],
        width=W,
        height=H,
        near_plane=float(camera["near_plane"]),
        far_plane=float(camera["far_plane"]),
        sh_degree=None,
        backgrounds=None,
        render_mode="ED",
        camera_model="ortho",
        packed=False,
        rasterize_mode="classic",
    )

    depth = rendered[0, ..., 0]
    return depth.detach().cpu().numpy().astype(np.float32)


def save_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(str(path))


def save_gray(path, gray01):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(gray01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(str(path))


def save_depth_vis(path, depth, alpha, alpha_threshold):
    valid = ((alpha >= alpha_threshold) & np.isfinite(depth) & (depth > 0))
    vis = np.zeros_like(depth, dtype=np.float32)

    if np.any(valid):
        vals = depth[valid]
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.0))
        if hi <= lo + EPS:
            hi = lo + 1.0
        vis[valid] = 1.0 - np.clip((depth[valid] - lo) / (hi - lo), 0.0, 1.0)

    save_gray(path, vis)


def save_aux_bundle(
    stem,
    out_aux_dir,
    alpha,
    depth,
    indices_np,
    meta,
    alpha_threshold,
):
    save_gray(out_aux_dir / (stem + "_alpha.png"), alpha)
    mask = (alpha >= alpha_threshold).astype(np.float32)
    save_gray(out_aux_dir / (stem + "_mask.png"), mask)

    np.save(str(out_aux_dir / (stem + "_indices.npy")), indices_np.astype(np.int64))

    if depth is not None:
        depth_clean = depth.copy()
        depth_clean[(alpha < alpha_threshold) | (~np.isfinite(depth_clean))] = 0.0
        np.save(str(out_aux_dir / (stem + "_depth.npy")), depth_clean.astype(np.float32))
        save_depth_vis(
            out_aux_dir / (stem + "_depth_vis.png"),
            depth_clean,
            alpha,
            alpha_threshold,
        )

    with open(str(out_aux_dir / (stem + "_meta.json")), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def render_and_save_view(
    stem,
    gaussians,
    indices,
    camera,
    sh_degree,
    background_rgb,
    color_mode,
    out_rgb_dir,
    out_aux_dir,
    save_aux,
    alpha_threshold,
    meta,
):
    rgb, alpha = render_rgb(
        gaussians=gaussians,
        indices=indices,
        camera=camera,
        sh_degree=sh_degree,
        background_rgb=background_rgb,
        color_mode=color_mode,
    )
    save_rgb(out_rgb_dir / (stem + "_rgb.png"), rgb)

    if save_aux:
        depth = render_expected_depth(
            gaussians=gaussians,
            indices=indices,
            camera=camera,
        )
        save_aux_bundle(
            stem=stem,
            out_aux_dir=out_aux_dir,
            alpha=alpha,
            depth=depth,
            indices_np=indices.detach().cpu().numpy(),
            meta=meta,
            alpha_threshold=alpha_threshold,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Top-down + upper-side + side orthographic renderer for Fast-PGSR Gaussian PLY."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="output/mungbean2_0804/target_exports/plant_only/point_cloud.ply",
        help="Input Gaussian PLY.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Default: <input_ply_parent>/render",
    )

    parser.add_argument("--steps", type=int, default=16, help="Number of top-down frames.")
    parser.add_argument("--height-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--margin", type=float, default=0.06)

    parser.add_argument("--num-upper-views", type=int, default=8)
    parser.add_argument("--num-side-views", type=int, default=8)
    parser.add_argument("--upper-elev-deg", type=float, default=35.0,
                        help="Oblique upper-side elevation in degrees. 0=horizontal, larger=more top-down.")
    parser.add_argument("--upper-start-deg", type=float, default=0.0)
    parser.add_argument("--side-start-deg", type=float, default=0.0)

    parser.add_argument("--background", choices=["black", "white"], default="black")
    parser.add_argument("--color-mode", choices=["sh", "dc"], default="sh")
    parser.add_argument("--save-aux", action="store_true")
    parser.add_argument("--alpha-threshold", type=float, default=0.01)
    parser.add_argument("--no-full-top", action="store_true",
                        help="If set, do not write full_top_rgb.png")

    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.num_upper_views < 0 or args.num_side_views < 0:
        raise ValueError("--num-upper-views and --num-side-views must be >= 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    if not (0.0 <= args.margin < 0.45):
        raise ValueError("--margin must be in [0, 0.45)")

    ply_path = Path(args.input).expanduser().resolve()
    if not ply_path.exists():
        raise FileNotFoundError(str(ply_path))

    out_root = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else ply_path.parent / "render"
    )
    out_rgb_dir = out_root / "rgb"
    out_aux_dir = out_root / "aux_data"

    out_rgb_dir.mkdir(parents=True, exist_ok=True)
    if args.save_aux:
        out_aux_dir.mkdir(parents=True, exist_ok=True)

    bg = (0.0, 0.0, 0.0) if args.background == "black" else (1.0, 1.0, 1.0)

    print("=" * 88)
    print("Fast-PGSR Multi-View Orthographic Renderer")
    print("=" * 88)
    print("[Input]          {}".format(ply_path))
    print("[Output RGB]     {}".format(out_rgb_dir))
    if args.save_aux:
        print("[Output AUX]     {}".format(out_aux_dir))
    print("[Topdown steps]  {}".format(args.steps))
    print("[Upper views]    {}".format(args.num_upper_views))
    print("[Side views]     {}".format(args.num_side_views))
    print("[Height axis]    {}".format(args.height_axis))
    print("[Color mode]     {}".format(args.color_mode))
    print("[Resolution]     {} x {}".format(args.width, args.height))
    print("[Upper elev deg] {}".format(args.upper_elev_deg))

    gaussians, sh_degree = load_gaussians(ply_path)
    print_dc_color_diagnostics(gaussians)

    xyz = gaussians.get_xyz
    device = xyz.device
    all_indices = torch.arange(xyz.shape[0], dtype=torch.long, device=device)

    bmin, bmax = bbox_from_xyz(xyz)
    print("[BBox] min={}, max={}, extent={}".format(bmin, bmax, bmax - bmin))

    top_camera = make_top_ortho_camera(
        bmin=bmin,
        bmax=bmax,
        height_axis=args.height_axis,
        width=args.width,
        height=args.height,
        margin=args.margin,
        device=device,
    )
    print(
        "[Top Camera] px/unit={:.4f}, forward={}".format(
            top_camera["pixels_per_unit"], top_camera["forward_world"]
        )
    )

    global_meta = {
        "input_ply": str(ply_path),
        "projection": "orthographic",
        "height_axis": args.height_axis,
        "steps": int(args.steps),
        "num_upper_views": int(args.num_upper_views),
        "num_side_views": int(args.num_side_views),
        "upper_elev_deg": float(args.upper_elev_deg),
        "upper_start_deg": float(args.upper_start_deg),
        "side_start_deg": float(args.side_start_deg),
        "color_mode": args.color_mode,
        "sh_degree": int(sh_degree),
        "bbox_min": bmin.tolist(),
        "bbox_max": bmax.tolist(),
        "background": args.background,
    }

    # ------------------------------------------------------------------
    # Optional full top render
    # ------------------------------------------------------------------
    if not args.no_full_top:
        print("[Render] full_top_rgb, N={}".format(int(all_indices.numel())))
        meta = dict(global_meta)
        meta.update({
            "scope": "full_top",
            "camera": camera_to_json(top_camera),
            "num_gaussians": int(all_indices.numel()),
        })
        render_and_save_view(
            stem="full_top",
            gaussians=gaussians,
            indices=all_indices,
            camera=top_camera,
            sh_degree=sh_degree,
            background_rgb=bg,
            color_mode=args.color_mode,
            out_rgb_dir=out_rgb_dir,
            out_aux_dir=out_aux_dir,
            save_aux=args.save_aux,
            alpha_threshold=args.alpha_threshold,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Top-down sequence
    # ------------------------------------------------------------------
    hmin, hmax, dh, planes = build_topdown_planes(
        xyz=xyz,
        height_axis=args.height_axis,
        steps=args.steps,
    )
    print("[Height] min={:.6f}, max={:.6f}, step_size={:.6f}".format(hmin, hmax, dh))

    global_meta["topdown_planes"] = planes
    global_meta["top_camera"] = camera_to_json(top_camera)

    # ------------------------------------------------------------------
    # Upper-side cameras
    # ------------------------------------------------------------------
    upper_view_meta = []
    preferred_up = world_up_vector(args.height_axis)
    for i in range(args.num_upper_views):
        az = args.upper_start_deg + i * (360.0 / float(max(args.num_upper_views, 1)))
        forward = make_ring_direction(args.height_axis, azim_deg=az, elev_deg=args.upper_elev_deg)
        cam = make_ortho_camera_from_forward(
            bmin=bmin,
            bmax=bmax,
            forward_world=forward,
            width=args.width,
            height=args.height,
            margin=args.margin,
            device=device,
            preferred_up_world=preferred_up,
        )
        upper_view_meta.append({
            "index": i,
            "azimuth_deg": float(az),
            "elevation_deg": float(args.upper_elev_deg),
            "camera": camera_to_json(cam),
        })

    # ------------------------------------------------------------------
    # Side cameras
    # ------------------------------------------------------------------
    side_view_meta = []
    for i in range(args.num_side_views):
        az = args.side_start_deg + i * (360.0 / float(max(args.num_side_views, 1)))
        forward = make_ring_direction(args.height_axis, azim_deg=az, elev_deg=0.0)
        cam = make_ortho_camera_from_forward(
            bmin=bmin,
            bmax=bmax,
            forward_world=forward,
            width=args.width,
            height=args.height,
            margin=args.margin,
            device=device,
            preferred_up_world=preferred_up,
        )
        side_view_meta.append({
            "index": i,
            "azimuth_deg": float(az),
            "elevation_deg": 0.0,
            "camera": camera_to_json(cam),
        })

    global_meta["upper_views"] = upper_view_meta
    global_meta["side_views"] = side_view_meta

    with open(str(out_root / "render_meta.json"), "w", encoding="utf-8") as f:
        json.dump(global_meta, f, ensure_ascii=False, indent=2)

    # Write topdown sequence
    for plane in planes:
        idx = int(plane["index"])
        cutoff = float(plane["keep_below_or_equal"])
        chosen = select_below_plane(
            xyz=xyz,
            height_axis=args.height_axis,
            cutoff=cutoff,
        )
        stem = "topdown_{:02d}".format(idx)
        print("[Render] {}_rgb: cutoff={:.6f}, kept_gaussians={}".format(
            stem, cutoff, int(chosen.numel())
        ))

        meta = dict(global_meta)
        meta.update({
            "scope": "topdown_sequence",
            "frame_index": idx,
            "frame_name": stem,
            "camera_plane": float(plane["camera_plane"]),
            "keep_below_or_equal": float(plane["keep_below_or_equal"]),
            "removed_from_top": float(plane["removed_from_top"]),
            "remaining_height_from_bottom": float(plane["remaining_height_from_bottom"]),
            "step_size": float(dh),
            "num_gaussians": int(chosen.numel()),
            "camera": camera_to_json(top_camera),
        })

        render_and_save_view(
            stem=stem,
            gaussians=gaussians,
            indices=chosen,
            camera=top_camera,
            sh_degree=sh_degree,
            background_rgb=bg,
            color_mode=args.color_mode,
            out_rgb_dir=out_rgb_dir,
            out_aux_dir=out_aux_dir,
            save_aux=args.save_aux,
            alpha_threshold=args.alpha_threshold,
            meta=meta,
        )

    # Write upper-side ring views
    for item in upper_view_meta:
        idx = int(item["index"])
        stem = "upper_{:02d}".format(idx)
        print("[Render] {}_rgb: azimuth={:.3f}, elevation={:.3f}, N={}".format(
            stem, item["azimuth_deg"], item["elevation_deg"], int(all_indices.numel())
        ))
        cam = make_ortho_camera_from_forward(
            bmin=bmin,
            bmax=bmax,
            forward_world=item["camera"]["forward_world"],
            width=args.width,
            height=args.height,
            margin=args.margin,
            device=device,
            preferred_up_world=preferred_up,
        )

        meta = dict(global_meta)
        meta.update({
            "scope": "upper_ring",
            "frame_index": idx,
            "frame_name": stem,
            "azimuth_deg": float(item["azimuth_deg"]),
            "elevation_deg": float(item["elevation_deg"]),
            "num_gaussians": int(all_indices.numel()),
            "camera": camera_to_json(cam),
        })
        render_and_save_view(
            stem=stem,
            gaussians=gaussians,
            indices=all_indices,
            camera=cam,
            sh_degree=sh_degree,
            background_rgb=bg,
            color_mode=args.color_mode,
            out_rgb_dir=out_rgb_dir,
            out_aux_dir=out_aux_dir,
            save_aux=args.save_aux,
            alpha_threshold=args.alpha_threshold,
            meta=meta,
        )

    # Write pure side ring views
    for item in side_view_meta:
        idx = int(item["index"])
        stem = "side_{:02d}".format(idx)
        print("[Render] {}_rgb: azimuth={:.3f}, elevation={:.3f}, N={}".format(
            stem, item["azimuth_deg"], item["elevation_deg"], int(all_indices.numel())
        ))
        cam = make_ortho_camera_from_forward(
            bmin=bmin,
            bmax=bmax,
            forward_world=item["camera"]["forward_world"],
            width=args.width,
            height=args.height,
            margin=args.margin,
            device=device,
            preferred_up_world=preferred_up,
        )

        meta = dict(global_meta)
        meta.update({
            "scope": "side_ring",
            "frame_index": idx,
            "frame_name": stem,
            "azimuth_deg": float(item["azimuth_deg"]),
            "elevation_deg": float(item["elevation_deg"]),
            "num_gaussians": int(all_indices.numel()),
            "camera": camera_to_json(cam),
        })
        render_and_save_view(
            stem=stem,
            gaussians=gaussians,
            indices=all_indices,
            camera=cam,
            sh_degree=sh_degree,
            background_rgb=bg,
            color_mode=args.color_mode,
            out_rgb_dir=out_rgb_dir,
            out_aux_dir=out_aux_dir,
            save_aux=args.save_aux,
            alpha_threshold=args.alpha_threshold,
            meta=meta,
        )

    print("=" * 88)
    print("[Done]")
    print("[RGB] {}".format(out_rgb_dir))
    if args.save_aux:
        print("[AUX_DATA] {}".format(out_aux_dir))
    total_main = args.steps + args.num_upper_views + args.num_side_views
    if not args.no_full_top:
        total_main += 1
    print("Main outputs written: {}".format(total_main))
    print("Requested sequence count = topdown({}) + upper({}) + side({})".format(
        args.steps, args.num_upper_views, args.num_side_views
    ))
    if not args.no_full_top:
        print("Plus one extra helper image: full_top_rgb.png")
    print("=" * 88)


if __name__ == "__main__":
    main()
