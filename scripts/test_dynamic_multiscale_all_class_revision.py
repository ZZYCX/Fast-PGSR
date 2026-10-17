#
# VEA+ / VGER+ acceptance test:
#   A. dynamic projection-scale multi-scale centre-weighted Mask voting
#   B. all-class training-time dynamic semantic revision
#
# Covers the 24 requested scenarios: scale derivation tables (3/5/7/9/11/15),
# duplicate-scale weight merging, single-scale weight=1, centre-weighted kernels,
# multi-scale fusion sums to 1, obs_count +1 once per view, background-near-plant,
# known-class revert-to-unknown over 2 windows, one-shot conflict non-modification,
# candidate-change reset, unknown->background, background->unknown->plant, no-observation
# invariance, candidate SLS (clone/split reset, prune sync), SSP (checkpoint + sidecar),
# switches-off legacy parity, no optimizer pollution and no sustained memory growth.
#
# Run: python scripts/test_dynamic_multiscale_all_class_revision.py  (needs CUDA)
#

import argparse
import io
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from arguments import OptimizationParams
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
from utils.semantic_evidence_utils import (odd_ceil, compute_dynamic_scales,
                                           _center_weighted_scale_prob,
                                           run_semantic_evidence_accumulation)

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[VEA+][{'PASS' if cond else 'FAIL'}]  {name}{(' | ' + detail) if detail else ''}")
    assert cond, f"VEA+ FAIL: {name}"


def close(a, b, atol=1e-4):
    return np.allclose(a, b, atol=atol)


RATIOS = [0.333333, 0.666667, 1.0]
WEIGHTS = [0.55, 0.30, 0.15]


def make_opt(**over):
    base = dict(enable_semantic_evidence=True, semantic_warmup_iterations=0,
                semantic_patch_size=5, semantic_depth_abs_tolerance=0.1,
                semantic_depth_rel_tolerance=0.02, semantic_min_render_alpha=0.5,
                semantic_evidence_debug=False, semantic_evidence_debug_interval=100,
                semantic_evidence_debug_export=False,
                enable_dynamic_multiscale_semantic_patch=True,
                semantic_multiscale_ratios=list(RATIOS),
                semantic_multiscale_weights=list(WEIGHTS),
                semantic_multiscale_sigma_ratio=0.25,
                semantic_dynamic_patch_max_size=0,
                enable_all_class_semantic_revision=True,
                semantic_known_keep_threshold=0.55,
                semantic_known_reject_threshold=0.35,
                semantic_revision_candidate_confidence=0.70,
                semantic_revision_margin=0.20,
                semantic_revision_min_observations=6,
                semantic_revision_stable_windows=2,
                semantic_background_candidate_confidence=0.75,
                semantic_background_candidate_max_target_prob=0.25,
                semantic_prior_strength=5.0,
                semantic_target_min_observations=5,
                semantic_target_confidence=0.75,
                semantic_target_margin=0.20,
                semantic_resolution_debug=False)
    base.update(over)
    return argparse.Namespace(**base)


def make_camera(W=200, H=200, Fx=100.0, Fy=100.0):
    FoVx = 2 * np.arctan(W / (2 * Fx))
    FoVy = 2 * np.arctan(H / (2 * Fy))
    cam = Camera(colmap_id=0, R=np.eye(3), T=np.zeros(3), FoVx=FoVx, FoVy=FoVy,
                 image_width=W, image_height=H, image_path="", image_name="testcam",
                 uid=0, preload_img=False, data_device="cuda")
    cam.Fx, cam.Fy, cam.Cx, cam.Cy = Fx, Fy, W / 2.0, H / 2.0
    return cam


def build_model(points, object_id):
    pts = np.asarray(points, dtype=np.float32)
    N = len(pts)
    rng = np.random.RandomState(0)
    pcd = BasicPointCloud(points=pts, colors=rng.rand(N, 3).astype(np.float32),
                          normals=np.zeros((N, 3), np.float32))
    sem = {"point3D_ids": np.arange(1000, 1000 + N, dtype=np.int64),
           "object_id": np.asarray(object_id, dtype=np.int64),
           "object_probs": np.tile([0.25, 0.25, 0.25, 0.25], (N, 1)).astype(np.float32),
           "confidence": np.full(N, 0.25, np.float32),
           "stats": {"raw_num_points": N, "filtered_num_points": N, "npz_total": N,
                     "matched": N, "missing_id_count": 0, "unused_npz": 0,
                     "max_xyz_alignment_error": 1e-7}}
    g = GaussianModel(3)
    g.create_from_pcd(pcd, 1.0, semantics=sem)
    return g


def make_render_pkg(cam, n, depth_value=10.0, alpha=1.0, radii_val=1.0, vis=None):
    H, W = cam.image_height, cam.image_width
    pd = torch.full((1, H, W), depth_value, device="cuda")
    al = torch.full((1, H, W), alpha, device="cuda")
    vis = torch.ones(n, dtype=bool, device="cuda") if vis is None else vis
    radii = torch.full((n,), float(radii_val), device="cuda")
    return {"plane_depth": pd, "rendered_alpha": al,
            "visibility_filter": vis, "radii": radii}


def set_window_all(g, row, window_prob, obs, object_id=None):
    """Set ev=obs*window_prob and obs for one row; optionally rewrite its object_id."""
    wp = np.asarray(window_prob, np.float32)
    n = g._xyz.shape[0]
    ev = g._semantic_evidence_sum.cpu().numpy().copy()
    oc = g._semantic_obs_count.cpu().numpy().copy()
    ev[row] = wp * obs
    oc[row] = obs
    g._semantic_evidence_sum = torch.tensor(ev, dtype=torch.float32, device="cuda")
    g._semantic_obs_count = torch.tensor(oc, dtype=torch.int64, device="cuda")
    if object_id is not None:
        oid = g._object_id.cpu().numpy().copy()
        oid[row] = object_id
        g._object_id = torch.tensor(oid, dtype=torch.int64, device="cuda")


def set_candidate(g, row, cid, cst):
    c = g._semantic_candidate_id.cpu().numpy().copy()
    s = g._semantic_candidate_stable_count.cpu().numpy().copy()
    c[row] = cid
    s[row] = cst
    g._semantic_candidate_id = torch.tensor(c, dtype=torch.int64, device="cuda")
    g._semantic_candidate_stable_count = torch.tensor(s, dtype=torch.int64, device="cuda")


def setup_training(g):
    pa = argparse.ArgumentParser()
    op = OptimizationParams(pa)
    opt_args = op.extract(pa.parse_args([]))
    g.training_setup(opt_args)
    for opti in [g.optimizer, g.shoptimizer]:
        for pg in opti.param_groups:
            p = pg["params"][0]
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        opti.step()
        opti.zero_grad()
    return opt_args


# =========================================================================== #
print("=" * 78)
print("T1-T6  动态尺度表（large=3/5/7/9/11/15）")
print("=" * 78)
EXPECTED = {3: [3], 5: [3, 5], 7: [3, 5, 7], 9: [3, 7, 9], 11: [5, 9, 11], 15: [5, 11, 15]}
for large, exp_sizes in EXPECTED.items():
    sizes, weights = compute_dynamic_scales(large, RATIOS, WEIGHTS)
    check(f"T{list(EXPECTED).index(large) + 1} large={large} scales == {exp_sizes}",
          sizes == exp_sizes, f"got {sizes} weights {[round(w, 4) for w in weights]}")
    check(f"T{list(EXPECTED).index(large) + 1} large={large} 权重和≈1", abs(sum(weights) - 1.0) < 1e-9)
    for s in sizes:
        check(f"T{list(EXPECTED).index(large) + 1} large={large} 尺度{s}为奇数", s % 2 == 1)
        check(f"T{list(EXPECTED).index(large) + 1} large={large} 尺度{s}∈[3,large]", 3 <= s <= large)

# =========================================================================== #
print("=" * 78)
print("T7/T8  重复尺度权重合并；单尺度权重=1")
print("=" * 78)
sizes, weights = compute_dynamic_scales(5, RATIOS, WEIGHTS)          # [3,5,5] -> [3,5]
check("T7 large=5 去重后为 [3,5]", sizes == [3, 5])
check("T7 合并后权重 [0.55, 0.45]", close(weights, [0.55, 0.45], 1e-4), f"{[round(w, 4) for w in weights]}")
sizes, weights = compute_dynamic_scales(3, RATIOS, WEIGHTS)          # [3,3,3] -> [3]
check("T8 large=3 单尺度 [3]", sizes == [3])
check("T8 单尺度权重==1.0", close([weights[0]], [1.0]))
check("T7 odd_ceil(2)==3", odd_ceil(2) == 3)
check("T7 odd_ceil(3)==3", odd_ceil(3) == 3)
check("T7 odd_ceil(4)==5", odd_ceil(4) == 5)
check("T7 odd_ceil(7.3)==9", odd_ceil(7.3) == 9)

# =========================================================================== #
print("=" * 78)
print("T9  尺度内部中心加权（浮点中心核，与解析高斯参考逐像素一致）")
print("=" * 78)
mask = torch.zeros((9, 9), dtype=torch.uint8, device="cuda")
mask[4, 4] = 1                                  # single plant pixel at mask center
u = torch.tensor([4.0, 4.5], device="cuda")
v = torch.tensor([4.0, 4.0], device="cuda")
prob, valid = _center_weighted_scale_prob(mask, u, v, 5, sigma=1.0)
check("T9 两个中心 scale 均有效", bool(valid.all()))


def ref_gaussian_patch(mask_np, u0, v0, size, sigma):
    H, W = mask_np.shape
    h = (size - 1) // 2
    cu, cv = round(u0), round(v0)
    wsum = np.zeros(4, np.float64)
    for dy in range(-h, h + 1):
        for dx in range(-h, h + 1):
            px, py = cu + dx, cv + dy
            if 0 <= px < W and 0 <= py < H:
                w = np.exp(-((px - u0) ** 2 + (py - v0) ** 2) / (2.0 * sigma * sigma))
                wsum[mask_np[py, px]] += w
    return wsum / wsum.sum()


mn = mask.cpu().numpy()
p0 = ref_gaussian_patch(mn, 4.0, 4.0, 5, 1.0)
p1 = ref_gaussian_patch(mn, 4.5, 4.0, 5, 1.0)
check("T9 中心(4,4) plant权重 == 解析参考", close(prob[0].cpu().numpy(), p0))
check("T9 中心(4.5,4) plant权重 == 解析参考（浮点中心核）", close(prob[1].cpu().numpy(), p1))
check("T9 浮点中心 shift 0.5px 改变 plant 概率", abs(p1[1] - p0[1]) > 1e-3)
check("T9 概率和为1", close(prob.sum(-1).cpu().numpy(), np.ones(2)))

# =========================================================================== #
print("=" * 78)
print("T10/T11  动态多尺度融合：证据行和=1，obs_count 每视角+1（非每尺度）")
print("=" * 78)
g = build_model([[0.0, 0.0, 10.0], [0.0, 0.0, 10.1]], [-1, -1])
cam = make_camera()
mask = torch.zeros((cam.image_height, cam.image_width), dtype=torch.uint8, device="cuda")
mask[95:106, 95:106] = 1                      # plant block covering every scale patch
cam.object_mask = mask
opt = make_opt()
rp = make_render_pkg(cam, 2, depth_value=10.0, radii_val=3.0)   # large=7 -> scales [3,5,7]
run_semantic_evidence_accumulation(opt, 1, cam, g, rp)
oc = g._semantic_obs_count.cpu().numpy()
ev = g._semantic_evidence_sum.cpu().numpy()
check("T11 动态多尺度 obs_count == 1（非3）", int(oc[0]) == 1 and int(oc[1]) == 1)
check("T10 evidence 行和≈1", close(ev.sum(-1), np.ones(2), 1e-4), f"{ev.sum(-1)}")
check("T10 plant 概率为主（plant块覆盖所有尺度）", ev[0, 1] > 0.9, f"{ev[0]}")

# =========================================================================== #
print("=" * 78)
print("T12  background中心靠近plant时融合仍倾向background")
print("=" * 78)
g = build_model([[0.0, 0.0, 10.0]], [-1])
cam = make_camera()
mask = torch.zeros((cam.image_height, cam.image_width), dtype=torch.uint8, device="cuda")
mask[102, 100] = 1                            # plant 2px right of the bg center (100,100)
cam.object_mask = mask
opt = make_opt()
run_semantic_evidence_accumulation(opt, 1, cam, g, make_render_pkg(cam, 1, 10.0, radii_val=3.0))
ev = g._semantic_evidence_sum.cpu().numpy()[0]
check("T12 背景像素融合仍倾向 background", ev[0] > 0.5, f"{ev}")
check("T12 行和≈1", close(ev.sum(), 1.0, 1e-4))

# =========================================================================== #
print("=" * 78)
print("T13/T14/T15  all-class 修正：已知plant 冲突窗口 / 单次不修改 / 候选切换重置")
print("=" * 78)
# T13+T14: plant row, two consecutive background-conflict windows
g = build_model([[0.0, 0.0, 10.0]], [1])
BG_WP = [0.80, 0.10, 0.05, 0.05]
set_window_all(g, 0, BG_WP, 6)
ret = g.resolve_unknown_semantics(make_opt(), iteration=1)
check("T13 窗口1 冲突：仍 plant", int(g._object_id.cpu().numpy()[0]) == 1)
check("T14 一次冲突不修改标签（仍 plant）", int(g._object_id.cpu().numpy()[0]) == 1)
check("T13 窗口1 candidate_id==0(bg)", int(g._semantic_candidate_id.cpu().numpy()[0]) == 0)
check("T13 窗口1 candidate_stable==1", int(g._semantic_candidate_stable_count.cpu().numpy()[0]) == 1)
set_window_all(g, 0, BG_WP, 6)
ret = g.resolve_unknown_semantics(make_opt(), iteration=2)
check("T13 窗口2 连续冲突 -> 退回 unknown", int(g._object_id.cpu().numpy()[0]) == -1,
      f"oid={g._object_id.cpu().numpy()[0]}")
check("T13 退回后 candidate 重置", int(g._semantic_candidate_id.cpu().numpy()[0]) == -1
      and int(g._semantic_candidate_stable_count.cpu().numpy()[0]) == 0)
check("T13 退回后 object_prob == window_prob", close(g._object_prob.cpu().numpy()[0], BG_WP, 1e-5))

# T15: plant row, window1 conflict top1=bg, window2 conflict top1=rubik_1 -> stable reset
g = build_model([[0.0, 0.0, 10.0]], [1])
set_window_all(g, 0, [0.80, 0.10, 0.05, 0.05], 6)
g.resolve_unknown_semantics(make_opt(), iteration=1)
check("T15 候选1 top1=bg stable=1", int(g._semantic_candidate_id.cpu().numpy()[0]) == 0
      and int(g._semantic_candidate_stable_count.cpu().numpy()[0]) == 1)
set_window_all(g, 0, [0.10, 0.05, 0.80, 0.05], 6)     # top1=rubik_1
g.resolve_unknown_semantics(make_opt(), iteration=2)
check("T15 候选类别变化 -> candidate_id=2, stable重置为1",
      int(g._semantic_candidate_id.cpu().numpy()[0]) == 2
      and int(g._semantic_candidate_stable_count.cpu().numpy()[0]) == 1)

# =========================================================================== #
print("=" * 78)
print("T16  unknown -> background（连续两窗口）")
print("=" * 78)
g = build_model([[0.0, 0.0, 10.0]], [-1])
BG2 = [0.85, 0.05, 0.05, 0.05]
set_window_all(g, 0, BG2, 6)
g.resolve_unknown_semantics(make_opt(), iteration=1)
check("T16 窗口1 bg候选 仍unknown stable=1",
      int(g._object_id.cpu().numpy()[0]) == -1 and int(g._semantic_candidate_stable_count.cpu().numpy()[0]) == 1)
set_window_all(g, 0, BG2, 6)
g.resolve_unknown_semantics(make_opt(), iteration=2)
check("T16 窗口2 -> background", int(g._object_id.cpu().numpy()[0]) == 0)
check("T16 转bg后 candidate 重置", int(g._semantic_candidate_stable_count.cpu().numpy()[0]) == 0)

# =========================================================================== #
print("=" * 78)
print("T17  background -> unknown(2窗口) -> plant")
print("=" * 78)
g = build_model([[0.0, 0.0, 10.0]], [0])
PLANT_WP = [0.10, 0.80, 0.05, 0.05]
set_window_all(g, 0, PLANT_WP, 6)
g.resolve_unknown_semantics(make_opt(), iteration=1)
check("T17 窗口1 bg 被plant反对 仍background", int(g._object_id.cpu().numpy()[0]) == 0)
set_window_all(g, 0, PLANT_WP, 6)
g.resolve_unknown_semantics(make_opt(), iteration=2)
check("T17 窗口2 连续反对 -> 退回unknown", int(g._object_id.cpu().numpy()[0]) == -1)
set_window_all(g, 0, PLANT_WP, 6)
g.resolve_unknown_semantics(make_opt(), iteration=3)
check("T17 窗口3 unknown -> plant", int(g._object_id.cpu().numpy()[0]) == 1)
check("T17 未直接 background->plant", True)

# =========================================================================== #
print("=" * 78)
print("T18  无有效观测不修改任何标签")
print("=" * 78)
g = build_model([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0]], [1, 0])
oid0 = g._object_id.clone()
ret = g.resolve_unknown_semantics(make_opt(), iteration=1)
check("T18 obs=0 标签不变", torch.equal(g._object_id, oid0))
check("T18 obs=0 返回统计 observed=0", ret["observed_gaussians"] == 0)

# =========================================================================== #
print("=" * 78)
print("T19  candidate状态 clone/split 时清零")
print("=" * 78)
g = build_model([[0.0, 0.0, 10.0], [5.0, 0.0, 10.0], [-5.0, 0.0, 10.0]], [-1, -1, -1])
setup_training(g)
set_candidate(g, 0, 2, 3)
n = g._xyz.shape[0]
g._scaling.data = torch.full((n, 3), np.log(1e-4), device="cuda")
grads = torch.zeros((n, 1), device="cuda"); grads[0] = 1.0
g.densify_and_clone(grads, 0.0002, 1.0)
check("T19 clone 父候选保留", int(g._semantic_candidate_id.cpu().numpy()[0]) == 2
      and int(g._semantic_candidate_stable_count.cpu().numpy()[0]) == 3)
check("T19 clone 子候选清零", int(g._semantic_candidate_id.cpu().numpy()[n]) == -1
      and int(g._semantic_candidate_stable_count.cpu().numpy()[n]) == 0)
g._scaling.data = torch.full((g._xyz.shape[0], 3), np.log(1e-4), device="cuda")
g._scaling.data[1] = 0.0
n0 = g._xyz.shape[0]
set_candidate(g, 1, 1, 4)
g.densify_and_split(torch.ones((n0, 1), device="cuda"), 0.0002,
                    torch.zeros((n0, 1), device="cuda"), 0.0012,
                    1.0, torch.zeros((n0,), device="cuda"), N=2)
check("T19 split 后子候选清零", int(g._semantic_candidate_id.cpu().numpy()[n0]) == -1
      and int(g._semantic_candidate_stable_count.cpu().numpy()[n0]) == 0)

# =========================================================================== #
print("=" * 78)
print("T20  candidate状态 prune 时同步")
print("=" * 78)
g = build_model([[0.0, 0.0, 10.0], [5.0, 0.0, 10.0], [-5.0, 0.0, 10.0], [0.0, 5.0, 10.0]], [-1] * 4)
setup_training(g)
set_candidate(g, 0, 2, 3)
set_candidate(g, 3, 1, 7)
cid0 = g._semantic_candidate_id.clone()
cst0 = g._semantic_candidate_stable_count.clone()
rm = torch.zeros((4,), dtype=bool, device="cuda"); rm[1] = True
g.prune_points(rm)
keep = (~rm).cpu().numpy()
check("T20 prune 后 candidate_id 与keep同步", torch.equal(g._semantic_candidate_id, cid0[keep]))
check("T20 prune 后 candidate_stable 与keep同步", torch.equal(g._semantic_candidate_stable_count, cst0[keep]))
g._assert_semantic_alignment("T20")

# =========================================================================== #
print("=" * 78)
print("T21  checkpoint 与 sidecar 保存恢复正确")
print("=" * 78)
g = build_model([[0.0, 0.0, 10.0], [5.0, 0.0, 10.0]], [-1, 1])
opt_args = setup_training(g)
set_candidate(g, 0, 2, 3)
set_candidate(g, 1, 0, 1)
with tempfile.TemporaryDirectory() as td:
    ck = os.path.join(td, "ck.pth")
    torch.save(g.capture(), ck)
    g2 = GaussianModel(3)
    g2.restore(torch.load(ck), opt_args)
    check("T21 checkpoint candidate_id 逐行相等", torch.equal(g2._semantic_candidate_id, g._semantic_candidate_id))
    check("T21 checkpoint candidate_stable 逐行相等",
          torch.equal(g2._semantic_candidate_stable_count, g._semantic_candidate_stable_count))
    ply = os.path.join(td, "point_cloud.ply")
    g.save_ply(ply)
    g3 = GaussianModel(3)
    g3.load_ply(ply, load_semantics=True)
    check("T21 sidecar candidate_id 逐行相等", torch.equal(g3._semantic_candidate_id, g._semantic_candidate_id))
    check("T21 sidecar candidate_stable 逐行相等",
          torch.equal(g3._semantic_candidate_stable_count, g._semantic_candidate_stable_count))

# =========================================================================== #
print("=" * 78)
print("T22  两个开关关闭时 == 原路径（固定5x5、只处理unknown）")
print("=" * 78)
opt_off = make_opt(enable_dynamic_multiscale_semantic_patch=False,
                   enable_all_class_semantic_revision=False)
g = build_model([[0.0, 0.0, 10.0], [5.0, 0.0, 10.0]], [-1, 1])
cam = make_camera()
mask = torch.zeros((cam.image_height, cam.image_width), dtype=torch.uint8, device="cuda")
mask[100, 100] = 1; mask[103, 100] = 1
cam.object_mask = mask
rp = make_render_pkg(cam, 2, depth_value=10.0, radii_val=5.0)
run_semantic_evidence_accumulation(opt_off, 1, cam, g, rp)
oc = g._semantic_obs_count.cpu().numpy()
check("T22 开关关闭 VEA 只累计 unknown（plant obs==0）", int(oc[1]) == 0 and int(oc[0]) >= 0)
# known row never re-verified in legacy resolve even under strong opposition
g2 = build_model([[0.0, 0.0, 10.0]], [1])
set_window_all(g2, 0, [0.9, 0.03, 0.03, 0.04], 20)
g2.resolve_unknown_semantics(opt_off, iteration=1)
check("T22 开关关闭 已知plant不被反对修改", int(g2._object_id.cpu().numpy()[0]) == 1)
# accumulate on a known row is rejected in legacy mode
try:
    g3 = build_model([[0.0, 0.0, 10.0]], [1])
    g3.accumulate_semantic_evidence(torch.tensor([0], device="cuda"),
                                    torch.tensor([[0.5, 0.2, 0.2, 0.1]], device="cuda"))
    check("T22 关闭时 known 累计证据被拒绝", False)
except AssertionError:
    check("T22 关闭时 known 累计证据被拒绝", True)
check("T22 关闭时 candidate 状态保持默认", int(g._semantic_candidate_id.cpu().numpy()[1]) == -1
      and int(g._semantic_candidate_stable_count.cpu().numpy()[1]) == 0)

# =========================================================================== #
print("=" * 78)
print("T23  candidate Tensor 不在 optimizer（无污染）")
print("=" * 78)
g = build_model([[0.0, 0.0, 10.0], [5.0, 0.0, 10.0]], [-1, -1])
opt_args = setup_training(g)
ok = True
for opti in [g.optimizer, g.shoptimizer]:
    for grp in opti.param_groups:
        for p in grp["params"]:
            for k in ["_semantic_candidate_id", "_semantic_candidate_stable_count"]:
                ok &= getattr(g, k) is not p
                t = getattr(g, k)
                ok &= t.requires_grad is False and t.grad_fn is None and not isinstance(t, torch.nn.Parameter)
check("T23 candidate Tensor 非Parameter/无梯度/不在optimizer", ok)

# =========================================================================== #
print("=" * 78)
print("T24  连续运行无持续显存增长")
print("=" * 78)
g = build_model([[0.0, 0.0, 10.0]] * 8, [-1] * 8)
opt = make_opt()
cam = make_camera()
mask = torch.zeros((cam.image_height, cam.image_width), dtype=torch.uint8, device="cuda")
mask[100, 100] = 1
cam.object_mask = mask
rp = make_render_pkg(cam, 8, depth_value=10.0, radii_val=5.0)
for _ in range(3):
    run_semantic_evidence_accumulation(opt, 1, cam, g, rp)
    g.resolve_unknown_semantics(opt, iteration=2)
torch.cuda.empty_cache(); torch.cuda.synchronize()
mem0 = torch.cuda.memory_allocated()
for _ in range(8):
    run_semantic_evidence_accumulation(opt, 1, cam, g, rp)
    g.resolve_unknown_semantics(opt, iteration=3)
torch.cuda.synchronize()
mem1 = torch.cuda.memory_accumulated() if hasattr(torch.cuda, "memory_accumulated") else torch.cuda.memory_allocated()
check("T24 连续运行显存不持续增长", abs(mem1 - mem0) < 64 * 1024 * 1024,
      f"Δ={(mem1 - mem0) / 1e6:.1f}MB")

# =========================================================================== #
print("=" * 78)
print("T25  标签变化日志示例（semantic_resolution_debug）")
print("=" * 78)
buf = io.StringIO()
import contextlib
with contextlib.redirect_stdout(buf):
    g = build_model([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0]], [-1, 1])
    opt_dbg = make_opt(semantic_resolution_debug=True)
    set_window_all(g, 0, [0.1, 0.8, 0.05, 0.05], 20)   # unknown -> plant
    set_window_all(g, 1, [0.9, 0.03, 0.03, 0.04], 20)  # plant opposed by bg (stable=1, no revert yet)
    g.resolve_unknown_semantics(opt_dbg, iteration=1500)
log = buf.getvalue()
check("T25 debug 日志含 unknown_before / target_resolved / known_candidate_updated",
      "unknown_before=" in log and "target_resolved=" in log and "known_candidate_updated=" in log)
print("    " + log.strip().replace("\n", "\n    ")[:600])

# =========================================================================== #
print("=" * 78)
print("T26  已知类别满足保持条件时 candidate 重置")
print("=" * 78)
g = build_model([[0.0, 0.0, 10.0]], [1])
set_window_all(g, 0, [0.80, 0.10, 0.05, 0.05], 6)   # window1: bg conflict -> candidate=0 stable=1
g.resolve_unknown_semantics(make_opt(), iteration=1)
check("T26 冲突后 candidate_id=0 stable=1",
      int(g._semantic_candidate_id.cpu().numpy()[0]) == 0
      and int(g._semantic_candidate_stable_count.cpu().numpy()[0]) == 1)
set_window_all(g, 0, [0.05, 0.90, 0.03, 0.02], 8)   # window2: strong plant -> keep
g.resolve_unknown_semantics(make_opt(), iteration=2)
check("T26 保持条件满足 candidate 重置为 -1/0",
      int(g._semantic_candidate_id.cpu().numpy()[0]) == -1
      and int(g._semantic_candidate_stable_count.cpu().numpy()[0]) == 0)
check("T26 保持条件 object_id 不变(plant)", int(g._object_id.cpu().numpy()[0]) == 1)

# =========================================================================== #
print("=" * 78)
fails = [n for n, ok in RESULTS if not ok]
print(f"VEA+/VGER+ 测试总计 {len(RESULTS)} 项检查, 失败 {len(fails)} 项")
if fails:
    print("失败项:", fails[:20])
    raise SystemExit(1)
print("动态多尺度 + 全类别动态语义修正 全部通过 ✔")
print("=" * 78)
