#
# VEA: visibility-filtered semantic evidence accumulation.
#
# For each training view (after warm-up), project the semantically-bound Gaussians
# (unknown only by default; unknown + known targets plant/rubik_1/rubik_2 when
# --enable_all_class_semantic_revision), keep the raster-visible / in-front /
# inside-image / depth-consistent ones, read a local patch class distribution from
# camera.object_mask, and accumulate 4-class evidence into
# GaussianModel._semantic_evidence_sum / _semantic_obs_count.
#
# background (object_id == 0) is a TERMINAL semantic state: it never enters the VEA
# candidate set, never runs the patch statistics, and never gains evidence.
#
# Patch modes:
#   * fixed equal-weight kernel (original): --enable_dynamic_multiscale_semantic_patch
#     OFF -> the old integer 5x5 (semantic_patch_size) avg-pool ratio map is used.
#   * dynamic multi-scale centre-weighted voting: switch ON -> per-Gaussian patch
#     sizes are derived from the renderer's 2D radii (large = 2*ceil(radius)+1),
#     each scale is centre-weighted by a Gaussian kernel around the float projection,
#     and the scales vote with [small, medium, large] weights.
#
# Only evidence is collected; no label/probability/confidence is changed.
#
# Depth definition (verified against diff-plane-rasterization forward.cu):
#   render_pkg["plane_depth"][pix] = All_map[4] / -(n.ray)  = ray-parameter t to the
#   alpha-composited Gaussian tangent plane, in world units.  For a Gaussian CENTER,
#   the ray-parameter to the center equals its camera-space Z (pts_in_cam[:,2]) because
#   the pixel ray has z=1.  Hence z_gaussian == pts_in_cam[:,2] and z_render == plane_depth
#   are the same quantity (world-unit ray distance) and can be compared directly.
#

import math
import os

import numpy as np
import torch
import torch.nn.functional as F

# module-level state for the optional visual debug export (capped view set)
_EXPORTED_VIEWS = set()


def project_gaussians(camera, xyz):
    """Project Gaussian centers into the training camera.

    Returns (u, v, Z): float pixel coords (x right, y down) and camera-space Z.
    u = X/Z*Fx + Cx, v = Y/Z*Fy + Cy with pts_in_cam = xyz @ W2C^T[:3,:3] + W2C^T[3,:3].
    """
    W2C = camera.world_view_transform
    pts_cam = xyz @ W2C[:3, :3] + W2C[3, :3]
    Z = pts_cam[:, 2]
    safe_Z = torch.where(Z.abs() < 1e-9, torch.full_like(Z, 1e-9), Z)
    u = pts_cam[:, 0] / safe_Z * camera.Fx + camera.Cx
    v = pts_cam[:, 1] / safe_Z * camera.Fy + camera.Cy
    return u, v, Z


def compute_patch_class_ratios(mask, patch_size):
    """Edge-corrected local class ratio map (fixed equal-weight kernel).

    mask: uint8 [H,W] (labels 0..3) on cuda.
    Returns (ratios [4,H,W] float, denom [H,W] int) where ratios[c, v, u] is the
    fraction of class c inside the patch_size x patch_size window centred at (u,v),
    normalised by the number of IN-IMAGE pixels in that window (out-of-image padding
    is never counted as background).
    """
    k = int(patch_size)
    pad = k // 2
    H, W = mask.shape
    one_hot = F.one_hot(mask.to(torch.int64).clamp(0, 3), num_classes=4).float()  # [H,W,4]
    one_hot = one_hot.permute(2, 0, 1).unsqueeze(0)  # [1,4,H,W]
    ones = torch.ones((1, 1, H, W), dtype=torch.float32, device=mask.device)
    sums = F.avg_pool2d(one_hot, kernel_size=k, stride=1, padding=pad,
                        divisor_override=1)[0]                       # [4,H,W] sums in window
    denom = F.avg_pool2d(ones, kernel_size=k, stride=1, padding=pad,
                         divisor_override=1)[0, 0]                   # [H,W] in-image pixel count
    ratios = sums / denom.clamp_min(1.0)
    return ratios, denom


# --------------------------------------------------------------------------- #
# dynamic multi-scale centre-weighted patch helpers
# --------------------------------------------------------------------------- #
def odd_ceil(x):
    """Ceil to the nearest ODD integer (vectorised for tensors, scalar for python ints)."""
    if torch.is_tensor(x):
        n = torch.ceil(x.float()).long()
        return n + ((n + 1) % 2)
    n = int(math.ceil(float(x)))
    return n if n % 2 == 1 else n + 1


def compute_dynamic_scales(large_size, ratios, base_weights, max_size=0):
    """Derive (sizes, weights) for one Gaussian's dynamic patch set.

    large_size: odd int >= 3 (2*ceil(radius)+1).  ratios: ascending positive floats.
    base_weights: same length as ratios.  Every size is clamped to [3, large_size] and
    made odd; duplicate sizes merge their weights and the weights are re-normalised
    (a single surviving scale ends with weight 1.0).  max_size>0 is only a safety cap
    applied to large_size itself, never a fixed-scale rule.
    """
    if max_size and max_size > 0:
        large_size = min(large_size, int(max_size))
    sizes, weights = [], []
    for r, w in zip(ratios, base_weights):
        s = odd_ceil(large_size * r)
        s = max(3, s)
        s = min(s, large_size)
        merged = False
        for j in range(len(sizes)):
            if sizes[j] == s:
                weights[j] += w
                merged = True
                break
        if not merged:
            sizes.append(s)
            weights.append(w)
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("[VEA] 动态尺度权重和必须 > 0")
    return sizes, [wt / total for wt in weights]


def _center_weighted_scale_prob(mask, u, v, size, sigma, chunk=4096):
    """Per-Gaussian centre-weighted class probabilities inside an odd `size` patch.

    mask: uint8 [H,W] labels 0..3 (cuda). u,v: [G] FLOAT pixel centres (not rounded).
    sigma: scalar >= 1.0. For every Gaussian the patch is the size x size window
    anchored at the ROUNDED centre, but the weight of each pixel is
        exp(-((x-u)^2 + (y-v)^2) / (2*sigma^2))
    computed from the FLOAT centre, so the kernel is never a symmetric integer kernel.
    Returns (prob [G,4] float32, valid [G] bool). prob is renormalised over the
    IN-IMAGE pixels only; `valid` marks scales with at least one in-image pixel.
    """
    G = u.shape[0]
    out_prob = torch.zeros((G, 4), dtype=torch.float32, device=mask.device)
    out_valid = torch.zeros((G,), dtype=torch.bool, device=mask.device)
    H, W = mask.shape
    h = (size - 1) // 2
    offs = torch.arange(-h, h + 1, device=mask.device).float()
    ox, oy = torch.meshgrid(offs, offs, indexing="xy")
    ox = ox.reshape(-1)
    oy = oy.reshape(-1)                                   # [P]
    for s in range(0, G, chunk):
        e = min(s + chunk, G)
        uc = u[s:e].contiguous()
        vc = v[s:e].contiguous()
        cu = uc.round()
        cv = vc.round()
        px = cu[:, None] + ox[None, :]                    # [g, P]
        py = cv[:, None] + oy[None, :]
        valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        w = torch.exp(-((px - uc[:, None]) ** 2 + (py - vc[:, None]) ** 2)
                      / (2.0 * sigma * sigma))
        w = w * valid
        denom = w.sum(dim=1)                              # [g]
        pxc = px.clamp(0, W - 1).long()
        pyc = py.clamp(0, H - 1).long()
        labels = mask[pyc, pxc].long()                    # [g, P] (out-of-image rows carry w=0)
        onehot = F.one_hot(labels, num_classes=4).float()
        weighted = (w.unsqueeze(-1) * onehot).sum(dim=1)  # [g, 4]
        prob = weighted / denom.clamp_min(1e-9).unsqueeze(-1)
        out_prob[s:e] = prob
        out_valid[s:e] = denom > 0
    return out_prob, out_valid


def dynamic_multiscale_view_prob(mask, u, v, radii, ratios, base_weights, sigma_ratio,
                                 max_size=0):
    """Per-Gaussian multi-scale centre-weighted view probability.

    radii: [G] renderer 2D radii (must be valid; caller filters invalid radii first).
    Derives large = 2*ceil(radius)+1 per Gaussian, derives the scale set, and votes
    with the merged per-plan weights. A Gaussian with NO valid scale is marked invalid
    and must not update evidence. Returns (view_prob [G,4], valid [G] bool).
    """
    G = u.shape[0]
    large = torch.ceil(radii.float()).long() * 2 + 1      # always odd, >= 3
    if max_size and max_size > 0:
        large = large.clamp(max=int(max_size))
    view_prob = torch.zeros((G, 4), dtype=torch.float32, device=mask.device)
    view_valid = torch.zeros((G,), dtype=torch.bool, device=mask.device)

    plans = {}
    for lv in torch.unique(large).tolist():
        plans[int(lv)] = compute_dynamic_scales(int(lv), ratios, base_weights)

    for lv, (sizes, weights) in plans.items():
        sel = large == lv
        if not bool(sel.any()):
            continue
        idx = sel.nonzero().flatten()
        us = u[idx]
        vs = v[idx]
        accum = torch.zeros((idx.numel(), 4), dtype=torch.float32, device=mask.device)
        accum_w = torch.zeros((idx.numel(),), dtype=torch.float32, device=mask.device)
        for size, sw in zip(sizes, weights):
            sigma = max(1.0, float(sigma_ratio) * size)
            prob, vvalid = _center_weighted_scale_prob(mask, us, vs, size, sigma)
            # invalid scale (no in-image pixel) contributes nothing; surviving scale
            # weights are re-normalised through accum_w.
            accum += sw * prob * vvalid.unsqueeze(-1)
            accum_w += sw * vvalid.float()
        ok = accum_w > 0
        vp = accum / accum_w.clamp_min(1e-9).unsqueeze(-1)
        vp = vp / vp.sum(dim=-1, keepdim=True).clamp_min(1e-9)   # final safe renormalisation
        view_prob[idx] = vp
        view_valid[idx] = ok
    return view_prob, view_valid


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def run_semantic_evidence_accumulation(opt, iteration, camera, gaussians, render_pkg,
                                       debug_dir=None):
    """Run VEA for the current camera. Called after render, before topology changes.

    Only accumulates evidence; never modifies labels/probabilities/confidence.
    Safe no-op when semantics/mask/visibility/depth data are missing.
    """
    if not opt.enable_semantic_evidence:
        return None
    if not gaussians._semantics_attached:
        return None
    if camera.object_mask is None:
        return None
    if iteration < opt.semantic_warmup_iterations:
        return None
    if "plane_depth" not in render_pkg:
        return None

    dynamic = bool(getattr(opt, "enable_dynamic_multiscale_semantic_patch", False))
    all_class = bool(getattr(opt, "enable_all_class_semantic_revision", False))
    if not dynamic:
        k = int(opt.semantic_patch_size)
        if k <= 0 or k % 2 == 0:
            raise ValueError(f"[VEA] semantic_patch_size 必须为正奇数，当前 {k}")
    else:
        ratios = list(getattr(opt, "semantic_multiscale_ratios", (0.333333, 0.666667, 1.0)))
        base_weights = list(getattr(opt, "semantic_multiscale_weights", (0.55, 0.30, 0.15)))
        sigma_ratio = float(getattr(opt, "semantic_multiscale_sigma_ratio", 0.25))
        max_size = int(getattr(opt, "semantic_dynamic_patch_max_size", 0))
        if (len(ratios) == 0 or len(base_weights) != len(ratios)
                or any(r <= 0 for r in ratios) or any(w <= 0 for w in base_weights)):
            raise ValueError("[VEA] semantic_multiscale_ratios/weights 必须非空、同长度、全为正数")

    counters = {"out_of_frustum": 0, "behind_camera": 0, "outside_image": 0,
                "invalid_depth": 0, "depth_occluded": 0, "invalid_mask_patch": 0,
                "raster_visible": 0, "camera_front": 0, "inside_image": 0,
                "valid_render_depth": 0, "depth_consistent": 0, "evidence_updated": 0,
                "invalid_radius": 0}
    stats = {"unknown_total": int((gaussians._object_id == -1).sum()),
             "candidate_total": 0, "all_class": bool(all_class),
             "background_excluded": 0,
             "mean_patch_probability": None, "obs_count_min": None,
             "obs_count_mean": None, "obs_count_max": None,
             "zero_observation_unknown_count": None}
    stats_all = None

    with torch.no_grad():
        object_id = gaussians._object_id
        visibility = render_pkg["visibility_filter"]

        # 1) candidate screen: non-background classes (all_class) or unknown only (default).
        #    background (object_id == 0) is a TERMINAL semantic state and never enters VEA,
        #    so it neither runs the dynamic multi-scale patch nor gains evidence.
        if all_class:
            non_background = ((object_id == -1) | (object_id == 1)
                              | (object_id == 2) | (object_id == 3))
            cand = non_background & visibility
            counters["out_of_frustum"] = int((non_background & ~visibility).sum())
            stats["background_excluded"] = int((object_id == 0).sum())
        else:
            unknown = object_id == -1
            cand = unknown & visibility
            counters["out_of_frustum"] = int((unknown & ~visibility).sum())
        cand_idx = cand.nonzero().flatten()
        counters["raster_visible"] = int(cand_idx.numel())
        stats["candidate_total"] = int(cand_idx.numel())
        if cand_idx.numel() == 0:
            _debug_log(opt, iteration, counters, stats, None)
            return None

        # 2) project + in-front + inside-image (keep FLOAT centres for dynamic patch)
        u, v, Z = project_gaussians(camera, gaussians._xyz.detach()[cand_idx])
        front = Z > 0
        counters["behind_camera"] = int((~front).sum())
        counters["camera_front"] = int(front.sum())
        uf, vf, Zf = u[front], v[front], Z[front]
        idx_f = cand_idx[front]
        inside = (uf >= 0) & (uf < camera.image_width) & (vf >= 0) & (vf < camera.image_height)
        counters["outside_image"] = int((~inside).sum())
        counters["inside_image"] = int(inside.sum())
        ui = torch.round(uf[inside]).clamp(0, camera.image_width - 1).long()
        vi = torch.round(vf[inside]).clamp(0, camera.image_height - 1).long()
        idx_i = idx_f[inside]
        u_float = uf[inside]
        v_float = vf[inside]
        Zi = Zf[inside]
        if idx_i.numel() == 0:
            _debug_log(opt, iteration, counters, stats, None)
            return None

        # 3) sample render depth + alpha at the projected pixel
        plane_depth = render_pkg["plane_depth"].detach().squeeze(0)   # [H,W]
        z_render = plane_depth[vi, ui]
        valid_depth = torch.isfinite(z_render) & (z_render > 0)
        alpha_ok = torch.ones_like(valid_depth)
        if opt.semantic_min_render_alpha > 0 and "rendered_alpha" in render_pkg:
            alpha_map = render_pkg["rendered_alpha"].detach().squeeze(0)
            alpha_ok = alpha_map[vi, ui] >= opt.semantic_min_render_alpha
        valid_depth = valid_depth & alpha_ok
        counters["invalid_depth"] = int((~valid_depth).sum())
        counters["valid_render_depth"] = int(valid_depth.sum())

        # 4) depth consistency vs plane_depth
        depth_diff = (Zi - z_render).abs()
        tol = torch.maximum(torch.full_like(z_render, float(opt.semantic_depth_abs_tolerance)),
                            float(opt.semantic_depth_rel_tolerance) * z_render.abs())
        depth_ok = valid_depth & (depth_diff <= tol)
        counters["depth_occluded"] = int((valid_depth & (depth_diff > tol)).sum())
        counters["depth_consistent"] = int(depth_ok.sum())
        idx_d = idx_i[depth_ok]
        u_float_d = u_float[depth_ok]
        v_float_d = v_float[depth_ok]
        if idx_d.numel() == 0:
            _debug_log(opt, iteration, counters, stats, None)
            return None

        # 5) local patch class distribution (dynamic multi-scale OR fixed kernel).
        # Track the per-candidate z / z_render / depth_diff through the final filter
        # so stats_all stays row-aligned with idx_up in BOTH paths.
        Zi_c = Zi[depth_ok]
        zr_c = z_render[depth_ok]
        dd_c = depth_diff[depth_ok]
        if dynamic:
            if "radii" not in render_pkg:
                raise RuntimeError("[VEA] 动态多尺度需要 render_pkg['radii']")
            radii_c = render_pkg["radii"].detach()[idx_d]
            rvalid = torch.isfinite(radii_c) & (radii_c > 0)
            counters["invalid_radius"] = int((~rvalid).sum())
            idx_v = idx_d[rvalid]
            if idx_v.numel() == 0:
                _debug_log(opt, iteration, counters, stats, None)
                return None
            u_dyn = u_float_d[rvalid]
            v_dyn = v_float_d[rvalid]
            radii_v = radii_c[rvalid]
            Zi_v, zr_v, dd_v = Zi_c[rvalid], zr_c[rvalid], dd_c[rvalid]
            view_prob, vp_valid = dynamic_multiscale_view_prob(
                camera.object_mask, u_dyn, v_dyn, radii_v,
                ratios, base_weights, sigma_ratio, max_size)
            counters["invalid_mask_patch"] = int((~vp_valid).sum())
            idx_up = idx_v[vp_valid]
            evidence = view_prob[vp_valid]
            u_fin, v_fin = u_dyn[vp_valid], v_dyn[vp_valid]
            Zi_fin, zr_fin, dd_fin = Zi_v[vp_valid], zr_v[vp_valid], dd_v[vp_valid]
            if idx_up.numel() == 0:
                _debug_log(opt, iteration, counters, stats, None)
                return None
        else:
            ratios_map, denom = compute_patch_class_ratios(camera.object_mask, k)
            ud = ui[depth_ok]
            vd = vi[depth_ok]
            patch_prob = ratios_map[:, vd, ud].t()                 # [K,4]
            patch_valid = denom[vd, ud] > 0
            counters["invalid_mask_patch"] = int((~patch_valid).sum())
            final = patch_valid
            idx_up = idx_d[final]
            evidence = patch_prob[final]
            u_fin = ud[final]
            v_fin = vd[final]
            Zi_fin, zr_fin, dd_fin = Zi_c[final], zr_c[final], dd_c[final]
            if idx_up.numel() == 0:
                _debug_log(opt, iteration, counters, stats, None)
                return None

        # 6) accumulate
        gaussians.accumulate_semantic_evidence(idx_up, evidence, enable_all_class=all_class)
        counters["evidence_updated"] = int(idx_up.numel())
        stats["mean_patch_probability"] = evidence.mean(dim=0).detach().cpu().numpy()
        obs = gaussians._semantic_obs_count
        obs_up = obs[idx_up]
        stats["obs_count_min"] = int(obs_up.min().item())
        stats["obs_count_mean"] = float(obs_up.float().mean().item())
        stats["obs_count_max"] = int(obs_up.max().item())
        stats["zero_observation_unknown_count"] = int(((gaussians._object_id == -1) & (obs == 0)).sum())
        stats_all = {"indices": idx_up.detach().cpu().numpy(),
                     "source_point3D_id": gaussians._source_point3D_id[idx_up].detach().cpu().numpy(),
                     "u": u_fin.detach().cpu().numpy(),
                     "v": v_fin.detach().cpu().numpy(),
                     "z_gaussian": Zi_fin.detach().cpu().numpy(),
                     "z_render": zr_fin.detach().cpu().numpy(),
                     "depth_diff": dd_fin.detach().cpu().numpy(),
                     "patch_prob": evidence.detach().cpu().numpy(),
                     "updated": np.ones(idx_up.numel(), dtype=bool)}

    _debug_log(opt, iteration, counters, stats, stats_all)
    if opt.semantic_evidence_debug_export and debug_dir:
        _export_view_debug(opt, iteration, camera, gaussians, render_pkg, debug_dir,
                           idx_up, counters)
    return stats_all


# --------------------------------------------------------------------------- #
def _debug_log(opt, iteration, counters, stats, stats_all):
    if not opt.semantic_evidence_debug:
        return
    if iteration % opt.semantic_evidence_debug_interval != 0:
        return
    mp = stats["mean_patch_probability"]
    mp_str = np.round(mp, 4).tolist() if mp is not None else "n/a"
    mode = "NON-BACKGROUND-REVISION" if stats["all_class"] else "unknown"
    print("[VEA] iteration={} mode={} unknown_total={} candidate_total={} background_excluded={} "
          "raster_visible={} camera_front={} inside_image={} valid_render_depth={} "
          "depth_consistent={} evidence_updated={} mean_patch_probability={} obs_count_min={} "
          "obs_count_mean={:.2f} obs_count_max={} zero_observation_unknown_count={}".format(
              iteration, mode, stats["unknown_total"], stats["candidate_total"],
              stats.get("background_excluded", 0),
              counters["raster_visible"], counters["camera_front"], counters["inside_image"],
              counters["valid_render_depth"], counters["depth_consistent"],
              counters["evidence_updated"], mp_str, stats["obs_count_min"],
              stats["obs_count_mean"], stats["obs_count_max"],
              stats["zero_observation_unknown_count"]))
    print("[VEA]   rejected: out_of_frustum={} behind_camera={} outside_image={} "
          "invalid_depth={} depth_occluded={} invalid_mask_patch={} invalid_radius={}".format(
              counters["out_of_frustum"], counters["behind_camera"],
              counters["outside_image"], counters["invalid_depth"],
              counters["depth_occluded"], counters["invalid_mask_patch"],
              counters["invalid_radius"]))


def _export_view_debug(opt, iteration, camera, gaussians, render_pkg, debug_dir,
                       idx_up, counters):
    """Export (image, coloured mask, projected overlays) + a per-Gaussian
    JSON/NPZ record for up to a few views. Pure debug: never touches training tensors."""
    import cv2
    global _EXPORTED_VIEWS
    if camera.image_name in _EXPORTED_VIEWS or len(_EXPORTED_VIEWS) >= 5:
        return
    _EXPORTED_VIEWS.add(camera.image_name)
    os.makedirs(debug_dir, exist_ok=True)

    img, _ = camera.get_image()
    img_np = img.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    H, W = img_np.shape[:2]
    mask_np = camera.object_mask.cpu().numpy()
    color_map = {0: (70, 130, 180), 1: (0, 200, 0), 2: (220, 20, 60), 3: (255, 215, 0)}
    colored = np.zeros((H, W, 3), dtype=np.uint8)
    for cl, cc in color_map.items():
        colored[mask_np == cl] = cc

    base = os.path.join(debug_dir, f"iter{iteration:06d}_{camera.image_name}")
    cv2.imwrite(base + "_image.jpg", (img_np * 255).astype(np.uint8)[:, :, ::-1])
    cv2.imwrite(base + "_mask.png", colored)

    # overlays: all unknown (yellow), raster-visible (blue), updated (green), occluded (red)
    with torch.no_grad():
        xyz = gaussians._xyz.detach()
        unknown = gaussians._object_id == -1
        u_all, v_all, Z_all = project_gaussians(camera, xyz)
        in_front = Z_all > 0
        ui_all = torch.round(u_all[in_front & unknown]).clamp(0, W - 1).long()
        vi_all = torch.round(v_all[in_front & unknown]).clamp(0, H - 1).long()
        vis = render_pkg["visibility_filter"]

        ov = (img_np * 0.5 * 255).astype(np.uint8).copy()   # dim background
        _scatter(ov, ui_all, vi_all, (255, 255, 0))          # yellow: all unknown (front)
        vis_idx = (in_front & unknown & vis).nonzero().flatten()
        _scatter(ov, torch.round(u_all[vis_idx]).clamp(0, W - 1).long(),
                 torch.round(v_all[vis_idx]).clamp(0, H - 1).long(), (0, 0, 255))  # blue
        _scatter(ov, torch.round(u_all[idx_up]).clamp(0, W - 1).long(),
                 torch.round(v_all[idx_up]).clamp(0, H - 1).long(), (0, 255, 0))    # green
        rejected = (in_front & unknown & vis)
        if "plane_depth" in render_pkg:
            pd = render_pkg["plane_depth"].detach().squeeze(0)
            ui_r = torch.round(u_all[rejected]).clamp(0, W - 1).long()
            vi_r = torch.round(v_all[rejected]).clamp(0, H - 1).long()
            zr = pd[vi_r, ui_r]
            diff = (Z_all[rejected] - zr).abs()
            tol = torch.maximum(torch.full_like(zr, float(opt.semantic_depth_abs_tolerance)),
                                float(opt.semantic_depth_rel_tolerance) * zr.abs())
            occ = rejected.clone()
            occ[rejected] = (zr > 0) & torch.isfinite(zr) & (diff > tol)
            _scatter(ov, torch.round(u_all[occ]).clamp(0, W - 1).long(),
                     torch.round(v_all[occ]).clamp(0, H - 1).long(), (255, 0, 0))   # red
        cv2.imwrite(base + "_overlay.jpg", ov[:, :, ::-1])


def _scatter(rgb, u, v, color):
    u = u.detach().cpu().numpy()
    v = v.detach().cpu().numpy()
    rgb[v, u, 0] = color[0]
    rgb[v, u, 1] = color[1]
    rgb[v, u, 2] = color[2]
