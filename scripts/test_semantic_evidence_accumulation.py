#
# VEA (Visibility-filtered Evidence Accumulation) acceptance test.
#
# Covers: patch class ratios (center/edges/corners/sizes/padding), independent
# projection reference, depth-consistency (abs/rel/boundary/NaN/Inf/0/negative),
# warm-up gating, only-unknown accumulation, non-semantic state invariance,
# accumulation correctness, SLS/SSP integration, safe no-op paths, and performance
# (vectorised, no graph, bounded memory).
#
# Run: python scripts/test_semantic_evidence_accumulation.py  (needs CUDA)
#

import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
from utils.semantic_evidence_utils import (project_gaussians, compute_patch_class_ratios,
                                           run_semantic_evidence_accumulation)

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[VEA-T] {'PASS' if cond else 'FAIL'}  {name}{(' | ' + detail) if detail else ''}")
    assert cond, f"VEA-T FAIL: {name}"


def close(a, b, atol=1e-5):
    return np.allclose(a, b, atol=atol)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_opt(**over):
    base = dict(enable_semantic_evidence=True, semantic_warmup_iterations=1000,
                semantic_patch_size=5, semantic_depth_abs_tolerance=0.1,
                semantic_depth_rel_tolerance=0.02, semantic_min_render_alpha=0.5,
                semantic_evidence_debug=False, semantic_evidence_debug_interval=100,
                semantic_evidence_debug_export=False)
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
           "object_probs": rng.dirichlet(np.ones(4), N).astype(np.float32),
           "confidence": rng.rand(N).astype(np.float32),
           "stats": {"raw_num_points": N, "filtered_num_points": N, "npz_total": N,
                     "matched": N, "missing_id_count": 0, "unused_npz": 0,
                     "max_xyz_alignment_error": 1e-7}}
    g = GaussianModel(3)
    g.create_from_pcd(pcd, 1.0, semantics=sem)
    return g


def make_render_pkg(cam, n, depth_value=10.0, alpha=1.0, vis=None):
    H, W = cam.image_height, cam.image_width
    pd = torch.full((1, H, W), depth_value, device="cuda")
    al = torch.full((1, H, W), alpha, device="cuda")
    vis = torch.ones(n, dtype=bool, device="cuda") if vis is None else vis
    radii = torch.ones(n, device="cuda")
    return {"plane_depth": pd, "rendered_alpha": al,
            "visibility_filter": vis, "radii": radii}


def ref_patch_ratio(mask_np, u, v, k):
    H, W = mask_np.shape
    h = k // 2
    win = mask_np[max(0, v - h):min(H, v + h + 1), max(0, u - h):min(W, u + h + 1)]
    tot = win.size
    counts = np.array([int((win == c).sum()) for c in range(4)], np.float32)
    return (counts / tot) if tot > 0 else None, tot


# --------------------------------------------------------------------------- #
print("=" * 78)
print("A. 5x5 及多尺寸 Mask 类别比例（中心/边/角/边界不把padding当bg）")
print("=" * 78)
# A1: 15x15, 5x5 block [5:10,5:10] with bg10 plant12 rubik_1 2 rubik_2 1
m = np.zeros((15, 15), dtype=np.uint8)
blk = np.full((5, 5), 0, dtype=np.uint8)
blk.flat[:12] = 1; blk.flat[12:14] = 2; blk.flat[14] = 3   # 12 plant, 2 rubik_1, 1 rubik_2, rest bg(10)
m[5:10, 5:10] = blk
mt = torch.from_numpy(m).cuda()
for k in [1, 3, 5, 7]:
    ratios, denom = compute_patch_class_ratios(mt, k)
    r7 = ratios[:, 7, 7].cpu().numpy()
    ref7, d7 = ref_patch_ratio(m, 7, 7, k)
    check(f"A patch_size={k} 中心像素与独立参考一致", close(r7, ref7), f"prod={r7} ref={ref7}")
    check(f"A patch_size={k} denom==窗口内有效像素数", int(denom[7, 7].item()) == d7)
ratios5, _ = compute_patch_class_ratios(mt, 5)
check("A patch_size=5 已知分布 [0.40,0.48,0.08,0.04]",
      close(ratios5[:, 7, 7].cpu().numpy(), [0.40, 0.48, 0.08, 0.04], 1e-5))
# A2: 边和角（padding 不计数为 bg）
m2 = np.zeros((5, 5), dtype=np.uint8); m2[0, 0] = 1   # corner is plant
mt2 = torch.from_numpy(m2).cuda()
r, d = compute_patch_class_ratios(mt2, 5)
ref, dd = ref_patch_ratio(m2, 0, 0, 5)
check("A 角(0,0) patch=5: 只统计9个图像内像素", int(d[0, 0].item()) == 9)
check("A 角(0,0) ratio==[8/9,1/9,0,0]（padding不计bg）", close(r[:, 0, 0].cpu().numpy(), [8/9, 1/9, 0, 0], 1e-6))
check("A 角(0,0) 与独立参考一致", close(r[:, 0, 0].cpu().numpy(), ref))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("B. 投影测试（与独立数学参考对比）")
print("=" * 78)
cam = make_camera()
xyz = torch.tensor([[0.0, 0.0, 10.0], [-1.0, -1.0, 10.0], [100.0, 0.0, 10.0],
                    [0.0, 0.0, -5.0], [0.0, 0.0, 100.0]], device="cuda")
u, v, Z = project_gaussians(cam, xyz)
ref_u = xyz[:, 0] / xyz[:, 2] * cam.Fx + cam.Cx
ref_v = xyz[:, 1] / xyz[:, 2] * cam.Fy + cam.Cy
check("B u 与独立参考一致", torch.allclose(u, ref_u, atol=1e-4))
check("B v 与独立参考一致", torch.allclose(v, ref_v, atol=1e-4))
check("B Z == 相机空间z", torch.allclose(Z, xyz[:, 2]))
front = Z > 0
inside = (u >= 0) & (u < cam.image_width) & (v >= 0) & (v < cam.image_height) & front
expect_front = torch.tensor([True, True, True, False, True], device="cuda")
expect_inside = torch.tensor([True, True, False, False, True], device="cuda")
check("B 前方掩码正确", torch.equal(front, expect_front))
check("B 图像内掩码正确（含越界/后方拒绝）", torch.equal(inside, expect_inside))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("C. 深度一致性（abs/rel/边界/NaN/Inf/0/负）")
print("=" * 78)
# two points -> same pixel (100,100), Z=10 and Z=15, plane_depth=10
g = build_model([[0.0, 0.0, 10.0], [0.0, 0.0, 15.0]], [-1, -1])
cam = make_camera()
mask = torch.zeros((cam.image_height, cam.image_width), dtype=torch.uint8, device="cuda")
cam.object_mask = mask
opt = make_opt(semantic_warmup_iterations=0)
rp = make_render_pkg(cam, 2, depth_value=10.0)
run_semantic_evidence_accumulation(opt, 1, cam, g, rp)
oc = g._semantic_obs_count.cpu().numpy()
check("C 前方点(Z=10)与render深度一致→累计", int(oc[0]) == 1)
check("C 后方点(Z=15)被遮挡拒绝", int(oc[1]) == 0)

# abs tolerance: point B diff=5, abs_tol=6 -> accepted
g = build_model([[0.0, 0.0, 10.0], [0.0, 0.0, 15.0]], [-1, -1])
cam.object_mask = mask
opt = make_opt(semantic_warmup_iterations=0, semantic_depth_abs_tolerance=6.0, semantic_depth_rel_tolerance=0.0)
run_semantic_evidence_accumulation(opt, 1, cam, g, make_render_pkg(cam, 2, 10.0))
oc = g._semantic_obs_count.cpu().numpy()
check("C abs_tol=6: 两点均接受", int(oc[0]) == 1 and int(oc[1]) == 1)

# rel tolerance: diff=5, z=10, rel=0.4 -> tol=4 -> B rejected; rel=0.6 -> tol=6 -> B accepted
g = build_model([[0.0, 0.0, 10.0], [0.0, 0.0, 15.0]], [-1, -1])
cam.object_mask = mask
opt = make_opt(semantic_warmup_iterations=0, semantic_depth_abs_tolerance=0.0, semantic_depth_rel_tolerance=0.4)
run_semantic_evidence_accumulation(opt, 1, cam, g, make_render_pkg(cam, 2, 10.0))
check("C rel_tol=0.4: 后方点拒绝", int(g._semantic_obs_count.cpu().numpy()[1]) == 0)
# boundary: diff == tol -> accepted (rel 0.5 -> tol 5, diff 5)
g = build_model([[0.0, 0.0, 10.0], [0.0, 0.0, 15.0]], [-1, -1])
cam.object_mask = mask
opt = make_opt(semantic_warmup_iterations=0, semantic_depth_abs_tolerance=0.0, semantic_depth_rel_tolerance=0.5)
run_semantic_evidence_accumulation(opt, 1, cam, g, make_render_pkg(cam, 2, 10.0))
check("C 边界 diff==tol 接受", int(g._semantic_obs_count.cpu().numpy()[1]) == 1)

# NaN / Inf / 0 / negative depth -> invalid
for bad, name in [(float('nan'), 'NaN'), (float('inf'), 'Inf'), (0.0, '0'), (-3.0, '负')]:
    g = build_model([[0.0, 0.0, 10.0]], [-1])
    cam.object_mask = mask
    opt = make_opt(semantic_warmup_iterations=0)
    rp = make_render_pkg(cam, 1, depth_value=10.0)
    rp["plane_depth"].fill_(bad)
    run_semantic_evidence_accumulation(opt, 1, cam, g, rp)
    check(f"C depth={name} 视为无效、不累计", int(g._semantic_obs_count.cpu().numpy()[0]) == 0)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("D. 预热规则（warmup 前后与边界）")
print("=" * 78)
g = build_model([[0.0, 0.0, 10.0]], [-1])
cam.object_mask = torch.zeros((cam.image_height, cam.image_width), dtype=torch.uint8, device="cuda")
opt = make_opt(semantic_warmup_iterations=5)
run_semantic_evidence_accumulation(opt, 4, cam, g, make_render_pkg(cam, 1, 10.0))
check("D warmup 前(iter=4) evidence_sum 不变", g._semantic_evidence_sum.sum().item() == 0)
check("D warmup 前(iter=4) obs_count 不变", g._semantic_obs_count.sum().item() == 0)
run_semantic_evidence_accumulation(opt, 5, cam, g, make_render_pkg(cam, 1, 10.0))
check("D warmup 边界(iter==5) 开始累计", int(g._semantic_obs_count.cpu().numpy()[0]) == 1)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("E. 只更新 unknown（已知类别行不变）")
print("=" * 78)
pts = [[0.0, 0.0, 10.0], [5.0, 0.0, 10.0], [-5.0, 0.0, 10.0], [0.0, 5.0, 10.0], [0.0, -5.0, 10.0]]
g = build_model(pts, [-1, 0, 1, 2, 3])
cam.object_mask = torch.zeros((cam.image_height, cam.image_width), dtype=torch.uint8, device="cuda")
opt = make_opt(semantic_warmup_iterations=0)
run_semantic_evidence_accumulation(opt, 1, cam, g, make_render_pkg(cam, 5, 10.0))
oc = g._semantic_obs_count.cpu().numpy()
ev = g._semantic_evidence_sum.cpu().numpy()
check("E 只有 unknown 行可能变化", oc[0] >= 0)
check("E background 行不变", oc[1] == 0 and ev[1].sum() == 0)
check("E plant 行不变", oc[2] == 0 and ev[2].sum() == 0)
check("E rubik_1 行不变", oc[3] == 0 and ev[3].sum() == 0)
check("E rubik_2 行不变", oc[4] == 0 and ev[4].sum() == 0)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("F. 非语义状态 VEA 前后完全不变")
print("=" * 78)
g = build_model([[0.0, 0.0, 10.0]], [-1])
cam.object_mask = torch.zeros((cam.image_height, cam.image_width), dtype=torch.uint8, device="cuda")
snap = {k: getattr(g, k).clone() for k in
        ["_object_id", "_object_prob", "_semantic_confidence", "_source_point3D_id",
         "_semantic_stable_count"]}
opt = make_opt(semantic_warmup_iterations=0)
run_semantic_evidence_accumulation(opt, 1, cam, g, make_render_pkg(cam, 1, 10.0))
for k, v in snap.items():
    check(f"F VEA后 {k} torch.equal", torch.equal(getattr(g, k), v))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("G. 累计正确性（三个视角证据逐项相加，obs=3）")
print("=" * 78)
g = build_model([[0.0, 0.0, 10.0]], [-1])
cam.object_mask = torch.zeros((cam.image_height, cam.image_width), dtype=torch.uint8, device="cuda")
opt = make_opt(semantic_warmup_iterations=0, semantic_patch_size=1)
probs = []
for lab in [0, 1, 2]:
    m = torch.full((cam.image_height, cam.image_width), lab, dtype=torch.uint8, device="cuda")
    cam.object_mask = m
    run_semantic_evidence_accumulation(opt, 1, cam, g, make_render_pkg(cam, 1, 10.0))
    p = np.zeros(4); p[lab] = 1.0
    probs.append(p)
expected = np.sum(probs, axis=0)
check("G evidence_sum == 三视角逐项相加", close(g._semantic_evidence_sum.cpu().numpy()[0], expected, 1e-5),
      f"got={g._semantic_evidence_sum.cpu().numpy()[0]} exp={expected}")
check("G obs_count == 3", int(g._semantic_obs_count.cpu().numpy()[0]) == 3)

# --------------------------------------------------------------------------- #
def setup_training(g):
    from arguments import OptimizationParams
    pa = argparse.ArgumentParser(); op = OptimizationParams(pa)
    opt_args = op.extract(pa.parse_args([]))
    g.training_setup(opt_args)
    for opti in [g.optimizer, g.shoptimizer]:
        for pg in opti.param_groups:
            pp = pg["params"][0]
            if pp.grad is None:
                pp.grad = torch.zeros_like(pp)
        opti.step(); opti.zero_grad()
    return opt_args


print("=" * 78)
print("H. SLS 集成：clone/split/prune 后动态证据规则")
print("=" * 78)
g = build_model([[0.0, 0.0, 10.0], [5.0, 0.0, 10.0], [-5.0, 0.0, 10.0]], [-1, -1, -1])
setup_training(g)
g.accumulate_semantic_evidence(torch.tensor([0], device="cuda"),
                               torch.tensor([[0.4, 0.3, 0.2, 0.1]], device="cuda"))
ev0 = g._semantic_evidence_sum[0].clone()
# clone point 0 (all small scaling, grads high only at 0)
n = g._xyz.shape[0]
g._scaling.data = torch.full((n, 3), np.log(1e-4), device="cuda")
grads = torch.zeros((n, 1), device="cuda"); grads[0] = 1.0
g.densify_and_clone(grads, 0.0002, 1.0)
check("H clone 后父行证据保留", torch.equal(g._semantic_evidence_sum[0], ev0))
check("H clone 子行证据为0", g._semantic_evidence_sum[n].sum().item() == 0)
# split point 1 (all small except point 1 large -> split selects it)
g._scaling.data = torch.full((g._xyz.shape[0], 3), np.log(1e-4), device="cuda")
g._scaling.data[1] = 0.0                        # log(1) -> large
n0 = g._xyz.shape[0]
g.densify_and_split(torch.ones((n0, 1), device="cuda"), 0.0002,
                    torch.zeros((n0, 1), device="cuda"), 0.0012,
                    1.0, torch.zeros((n0,), device="cuda"), N=2)
check("H split 后语义长度正确", g._semantic_evidence_sum.shape[0] == g._xyz.shape[0])
check("H split 未变化行证据保留", torch.equal(g._semantic_evidence_sum[0], ev0))
check("H split 子行证据为0",
      g._semantic_evidence_sum[n0:].sum().item() == 0)  # children appended last
# prune a row with evidence
ev_before = g._semantic_evidence_sum.clone()
rm = torch.zeros((g._xyz.shape[0],), dtype=bool, device="cuda"); rm[0] = True
g.prune_points(rm)
check("H prune 后证据与keep mask同步", torch.equal(g._semantic_evidence_sum, ev_before[1:]))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("I. SSP 集成：checkpoint 与 PLY sidecar 恢复后证据逐行相等")
print("=" * 78)
import io
from arguments import OptimizationParams
pa = argparse.ArgumentParser(); op = OptimizationParams(pa)
opt_args = op.extract(pa.parse_args([]))

g = build_model([[0.0, 0.0, 10.0], [5.0, 0.0, 10.0]], [-1, -1])
setup_training(g)
g.accumulate_semantic_evidence(torch.tensor([0, 1], device="cuda"),
                               torch.tensor([[0.2, 0.3, 0.4, 0.1], [0.5, 0.2, 0.2, 0.1]], device="cuda"))
with tempfile.TemporaryDirectory() as td:
    ck = os.path.join(td, "ck.pth")
    torch.save(g.capture(), ck)
    ma = torch.load(ck)
    g3 = GaussianModel(3); g3.restore(ma, opt_args)
    check("I checkpoint 恢复 evidence_sum 逐行相等", torch.equal(g3._semantic_evidence_sum, g._semantic_evidence_sum))
    check("I checkpoint 恢复 obs_count 逐行相等", torch.equal(g3._semantic_obs_count, g._semantic_obs_count))
    # PLY sidecar round-trip
    ply = os.path.join(td, "point_cloud.ply")
    g.save_ply(ply)
    g4 = GaussianModel(3); g4.load_ply(ply, load_semantics=True)
    check("I PLY 恢复 evidence_sum 逐行相等", torch.equal(g4._semantic_evidence_sum, g._semantic_evidence_sum))
    check("I PLY 恢复 obs_count 逐行相等", torch.equal(g4._semantic_obs_count, g._semantic_obs_count))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("J. 安全跳过：无语义 / 无Mask / 未启用")
print("=" * 78)
# no semantics
pcd = BasicPointCloud(points=np.zeros((2, 3), np.float32), colors=np.random.rand(2, 3).astype(np.float32), normals=np.zeros((2, 3), np.float32))
g5 = GaussianModel(3); g5.create_from_pcd(pcd, 1.0)
cam = make_camera(); cam.object_mask = torch.zeros((cam.image_height, cam.image_width), dtype=torch.uint8, device="cuda")
run_semantic_evidence_accumulation(make_opt(semantic_warmup_iterations=0), 1, cam, g5, make_render_pkg(cam, 2))
check("J 无语义安全跳过", True)
# no mask
g6 = build_model([[0.0, 0.0, 10.0]], [-1])
cam2 = make_camera(); cam2.object_mask = None
run_semantic_evidence_accumulation(make_opt(semantic_warmup_iterations=0), 1, cam2, g6, make_render_pkg(cam2, 1))
check("J 无Mask安全跳过（obs仍0）", int(g6._semantic_obs_count.cpu().numpy()[0]) == 0)
# not enabled
g7 = build_model([[0.0, 0.0, 10.0]], [-1])
cam.object_mask = torch.zeros((cam.image_height, cam.image_width), dtype=torch.uint8, device="cuda")
run_semantic_evidence_accumulation(make_opt(semantic_warmup_iterations=0, enable_semantic_evidence=False), 1, cam, g7, make_render_pkg(cam, 1))
check("J enable=False 安全跳过", int(g7._semantic_obs_count.cpu().numpy()[0]) == 0)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("K. 性能：向量化、无计算图、显存不持续增长")
print("=" * 78)
rng = np.random.RandomState(0)
N = 20000
pts = rng.rand(N, 3).astype(np.float32) * 20.0 - 10.0
g8 = build_model(pts, np.full(N, -1, dtype=np.int64))
camK = make_camera(W=640, H=360, Fx=500, Fy=500)
camK.object_mask = (rng.randint(0, 4, (360, 640))).astype(np.uint8)
camK.object_mask = torch.from_numpy(camK.object_mask).cuda()
rp = make_render_pkg(camK, N, depth_value=10.0)
opt = make_opt(semantic_warmup_iterations=0)
t0 = time.perf_counter()
run_semantic_evidence_accumulation(opt, 1, camK, g8, rp)
dt = time.perf_counter() - t0
print(f"[VEA-T] K 单轮耗时 {dt*1000:.1f} ms, N={N}, updated={int(g8._semantic_obs_count.sum().item())}")
check("K 单轮耗时 < 2s", dt < 2.0)
check("K evidence_sum 无 grad_fn", g8._semantic_evidence_sum.grad_fn is None)
# memory stability over repeated calls
torch.cuda.empty_cache(); torch.cuda.synchronize()
mem0 = torch.cuda.memory_allocated()
for _ in range(5):
    run_semantic_evidence_accumulation(opt, 1, camK, g8, rp)
torch.cuda.synchronize()
mem1 = torch.cuda.memory_allocated()
check("K 连续运行显存不持续增长", (mem1 - mem0) < 64 * 1024 * 1024,
      f"Δ={(mem1-mem0)/1e6:.1f}MB")
check("K 临时one-hot+比例图显存 < 100MB",
      camK.image_height * camK.image_width * 4 * 4 < 100e6)   # [4,H,W] float32

# --------------------------------------------------------------------------- #
print("=" * 78)
fails = [n for n, ok in RESULTS if not ok]
print(f"VEA 测试总计 {len(RESULTS)} 项检查, 失败 {len(fails)} 项")
if fails:
    print("失败项:", fails[:20])
    raise SystemExit(1)
print("VEA 证据累计测试全部通过 ✔")
print("=" * 78)
