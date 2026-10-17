#
# VGER (Visibility-Guided Evidence Resolution) acceptance test.
#
# Evidence tensors are constructed as  ev = obs * window_prob  (self-consistent,
# exactly what VEA produces: each accumulation adds a row-sum-1 patch and +1 obs).
#
# Run: python scripts/test_semantic_resolution.py  (needs CUDA)
#

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[VGER-T] {'PASS' if cond else 'FAIL'}  {name}{(' | ' + detail) if detail else ''}")
    assert cond, f"VGER-T FAIL: {name}"


def close(a, b, atol=1e-5):
    return np.allclose(a, b, atol=atol)


def make_opt(**over):
    base = dict(semantic_prior_strength=5.0, semantic_target_min_observations=5,
                semantic_target_confidence=0.75, semantic_target_margin=0.20,
                semantic_background_min_observations=8, semantic_background_confidence=0.95,
                semantic_background_max_target_prob=0.05, semantic_background_stable_windows=2,
                semantic_resolution_debug=False)
    base.update(over)
    return argparse.Namespace(**base)


UNIFORM = np.tile([0.25, 0.25, 0.25, 0.25], (1, 1)).astype(np.float32)


def build_model(n):
    rng = np.random.RandomState(0)
    pcd = BasicPointCloud(points=(rng.rand(n, 3).astype(np.float32) + 5.0),
                          colors=rng.rand(n, 3).astype(np.float32),
                          normals=np.zeros((n, 3), np.float32))
    sem = {"point3D_ids": np.arange(1000, 1000 + n, dtype=np.int64),
           "object_id": np.full(n, -1, dtype=np.int64),
           "object_probs": np.tile([0.25, 0.25, 0.25, 0.25], (n, 1)).astype(np.float32),
           "confidence": np.full(n, 0.25, dtype=np.float32),
           "stats": {"raw_num_points": n, "filtered_num_points": n, "npz_total": n,
                     "matched": n, "missing_id_count": 0, "unused_npz": 0,
                     "max_xyz_alignment_error": 1e-7}}
    g = GaussianModel(3)
    g.create_from_pcd(pcd, 1.0, semantics=sem)
    return g


def set_state(g, object_id=None, probs=None, ev=None, obs=None, stable=None, conf=None):
    n = g._xyz.shape[0]
    if object_id is not None:
        g._object_id = torch.tensor(np.asarray(object_id, np.int64), dtype=torch.int64, device="cuda")
    if probs is not None:
        g._object_prob = torch.tensor(np.asarray(probs, np.float32), dtype=torch.float32, device="cuda")
    if conf is not None:
        g._semantic_confidence = torch.tensor(np.asarray(conf, np.float32), dtype=torch.float32, device="cuda")
    if ev is not None:
        g._semantic_evidence_sum = torch.tensor(np.asarray(ev, np.float32), dtype=torch.float32, device="cuda")
    if obs is not None:
        g._semantic_obs_count = torch.tensor(np.asarray(obs, np.int64), dtype=torch.int64, device="cuda")
    if stable is not None:
        g._semantic_stable_count = torch.tensor(np.asarray(stable, np.int64), dtype=torch.int64, device="cuda")


def set_window(g, row, window_prob, obs, old_prob=None):
    """Set ev = obs*window_prob and obs for one row only (preserves other rows)."""
    wp = np.asarray(window_prob, np.float32)
    n = g._xyz.shape[0]
    full_ev = g._semantic_evidence_sum.cpu().numpy().copy()
    full_obs = g._semantic_obs_count.cpu().numpy().copy()
    full_ev[row] = wp * obs
    full_obs[row] = obs
    if old_prob is not None:
        full_p = g._object_prob.cpu().numpy().copy()
        full_p[row] = np.asarray(old_prob, np.float32)
        set_state(g, probs=full_p, ev=full_ev, obs=full_obs)
    else:
        set_state(g, ev=full_ev, obs=full_obs)


def ref_fusion(old_prob, ev, obs, prior):
    return (prior * np.asarray(old_prob, np.float32) + np.asarray(ev, np.float32)) \
           / (prior + np.asarray(obs, np.float32)[:, None])


# --------------------------------------------------------------------------- #
print("=" * 78)
print("T1/T2 窗口概率 + prior 融合（独立 numpy 参考）")
print("=" * 78)
g = build_model(3)
old = np.array([[0.25, 0.25, 0.25, 0.25], [0.2, 0.3, 0.3, 0.2], [0.1, 0.2, 0.3, 0.4]], np.float32)
obs = np.array([5, 4, 4], np.int64)
wps = np.array([[0.1, 0.5, 0.3, 0.1], [0.4, 0.2, 0.2, 0.2], [0.5, 0.25, 0.25, 0.0]], np.float32)
ev = wps * obs[:, None]
set_state(g, probs=old, ev=ev, obs=obs, conf=np.array([0.25, 0.25, 0.25]))
g.resolve_unknown_semantics(make_opt(), iteration=1)
exp = ref_fusion(old, ev, obs, 5.0)
check("T2 new_prob 与独立参考一致", close(g._object_prob.cpu().numpy(), exp, 1e-5))
check("T2 confidence == max(new_prob)", close(g._semantic_confidence.cpu().numpy(), exp.max(-1), 1e-5))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T3 obs=0 概率保持不变")
print("=" * 78)
g = build_model(2)
set_state(g, probs=np.tile([0.25] * 4, (2, 1)), ev=np.zeros((2, 4), np.float32),
          obs=np.array([0, 0], np.int64))
g.resolve_unknown_semantics(make_opt(), iteration=1)
check("T3 obs=0 的 object_prob 不变", close(g._object_prob.cpu().numpy(), np.tile([0.25] * 4, (2, 1))))
check("T3 obs=0 仍 unknown", int((g._object_id == -1).sum()) == 2)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T4 观测不足保持 unknown")
print("=" * 78)
g = build_model(1)
set_window(g, 0, [0, 1, 0, 0], 2)                       # obs=2 < 5
g.resolve_unknown_semantics(make_opt(), iteration=1)
check("T4 obs=2 < 5 保持 unknown", int(g._object_id.cpu().numpy()[0]) == -1)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T5 target confidence 边界（默认 prior，纯plant窗口）")
print("=" * 78)
g = build_model(2)
# obs=10 -> fused plant prob = (1.25+10)/15 = 0.75  (boundary, converts)
# obs=9  -> fused plant prob = (1.25+9)/14 = 0.732  (< 0.75, stays)
set_window(g, 0, [0, 1, 0, 0], 10, old_prob=UNIFORM[0])
set_window(g, 1, [0, 1, 0, 0], 9, old_prob=UNIFORM[0])
g.resolve_unknown_semantics(make_opt(), iteration=1)
oid = g._object_id.cpu().numpy()
check("T5 top1_prob=0.75(边界) → plant", int(oid[0]) == 1)
check("T5 top1_prob=0.732 < 0.75 → 仍unknown", int(oid[1]) == -1)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T6 target margin 边界（target_confidence=0.5，prior≈0 直接控制 margin）")
print("=" * 78)
g = build_model(2)
set_window(g, 0, [0.10, 0.55, 0.35, 0.0], 5, old_prob=UNIFORM[0])  # margin≈0.20
set_window(g, 1, [0.09, 0.55, 0.36, 0.0], 5, old_prob=UNIFORM[0])  # margin≈0.19
opt6 = make_opt(semantic_target_confidence=0.5, semantic_prior_strength=0.0)
g.resolve_unknown_semantics(opt6, iteration=1)
oid = g._object_id.cpu().numpy()
check("T6 margin=0.2(边界) → 转换", int(oid[0]) == 1)
check("T6 margin=0.19 < 0.2 → 仍unknown", int(oid[1]) == -1)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T7 top1 为 plant / rubik_1 / rubik_2 分别正确转换")
print("=" * 78)
g = build_model(3)
set_window(g, 0, [0.05, 0.9, 0.05, 0.0], 20, old_prob=UNIFORM[0])   # plant
set_window(g, 1, [0.05, 0.05, 0.9, 0.0], 20, old_prob=UNIFORM[0])   # rubik_1
set_window(g, 2, [0.0, 0.05, 0.05, 0.9], 20, old_prob=UNIFORM[0])   # rubik_2
g.resolve_unknown_semantics(make_opt(), iteration=1)
oid = g._object_id.cpu().numpy()
check("T7 top1=plant → 1", int(oid[0]) == 1)
check("T7 top1=rubik_1 → 2", int(oid[1]) == 2)
check("T7 top1=rubik_2 → 3", int(oid[2]) == 3)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T8/T9/T10 背景：单窗口保持unknown / 连续两窗口转bg / 中间失败归零")
print("=" * 78)
BG_OLD = np.array([0.9, 0.04, 0.04, 0.02], np.float32)     # bg-heavy achievable prior


def bg_wp(prob0):
    return np.array([prob0, (1 - prob0) / 3] * 1 + [0.0], np.float32)[:4] if False else \
           np.array([prob0, (1 - prob0) * 0.1, (1 - prob0) * 0.1, (1 - prob0) * 0.8], np.float32)


# T8: one window meeting bg -> stable=1, still unknown
BG_WP = [0.99, 0.01, 0.0, 0.0]      # drives fused prob0 >= 0.95 with prior 5 / obs 8
g = build_model(1)
set_window(g, 0, BG_WP, 8, old_prob=BG_OLD)
g.resolve_unknown_semantics(make_opt(), iteration=1)
check("T8 单窗口背景 stable=1 保持unknown", int(g._object_id.cpu().numpy()[0]) == -1)
check("T8 stable_count == 1", int(g._semantic_stable_count.cpu().numpy()[0]) == 1)
# T9: second window -> convert to background
set_window(g, 0, BG_WP, 8, old_prob=g._object_prob.cpu().numpy()[0])
g.resolve_unknown_semantics(make_opt(), iteration=2)
check("T9 连续两窗口 → background", int(g._object_id.cpu().numpy()[0]) == 0)
check("T9 转换后 stable_count 归零", int(g._semantic_stable_count.cpu().numpy()[0]) == 0)
# T10: window1 bg (stable=1), window2 observed-but-fails-bg (reset), window3 bg (stable=1)
g = build_model(1)
set_window(g, 0, BG_WP, 8, old_prob=BG_OLD)
g.resolve_unknown_semantics(make_opt(), iteration=1)
check("T10 窗口1 bg stable=1", int(g._semantic_stable_count.cpu().numpy()[0]) == 1)
set_window(g, 0, [0.2, 0.6, 0.1, 0.1], 8, old_prob=g._object_prob.cpu().numpy()[0])  # plant-heavy -> not bg
g.resolve_unknown_semantics(make_opt(), iteration=2)
check("T10 中间窗口不满足 → stable 归零", int(g._semantic_stable_count.cpu().numpy()[0]) == 0)
set_window(g, 0, BG_WP, 8, old_prob=g._object_prob.cpu().numpy()[0])
g.resolve_unknown_semantics(make_opt(), iteration=3)
check("T10 再次满足 bg 但 stable=1 < 2 仍unknown", int(g._object_id.cpu().numpy()[0]) == -1)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T11 大量不满足条件的 unknown 保持 unknown")
print("=" * 78)
g = build_model(20)
set_state(g, ev=np.tile([0.25] * 4, (20, 1)).astype(np.float32), obs=np.array([1] * 20, np.int64))
g.resolve_unknown_semantics(make_opt(), iteration=1)
check("T11 20 个低观测 unknown 全部保持", int((g._object_id == -1).sum()) == 20)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T12 已确定类别七个语义状态完全不变")
print("=" * 78)
g = build_model(4)
set_state(g, object_id=[0, 1, 2, 3], probs=np.tile([0.25] * 4, (4, 1)),
          conf=[0.25] * 4, ev=np.tile([1, 1, 1, 1], (4, 1)).astype(np.float32),
          obs=[4, 4, 4, 4], stable=[1, 2, 3, 4])
snap = {k: getattr(g, k).clone() for k in
        ["_object_id", "_object_prob", "_semantic_confidence", "_semantic_stable_count",
         "_semantic_evidence_sum", "_semantic_obs_count"]}
g.resolve_unknown_semantics(make_opt(), iteration=1)
for k in ["_object_id", "_object_prob", "_semantic_confidence", "_semantic_stable_count"]:
    check(f"T12 已确定类别 {k} 不变", torch.equal(getattr(g, k), snap[k]))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T13/T14 证据清零 + object_prob 行和=1")
print("=" * 78)
g = build_model(3)
set_window(g, 0, [0.05, 0.9, 0.05, 0.0], 5, old_prob=UNIFORM[0])
set_window(g, 1, [0.05, 0.05, 0.9, 0.0], 5, old_prob=UNIFORM[0])
set_window(g, 2, [0.4, 0.2, 0.2, 0.2], 5, old_prob=UNIFORM[0])
g.resolve_unknown_semantics(make_opt(), iteration=1)
check("T13 evidence_sum 全部清零", g._semantic_evidence_sum.sum().item() == 0)
check("T13 obs_count 全部清零", g._semantic_obs_count.sum().item() == 0)
check("T14 object_prob 行和≈1", close(g._object_prob.cpu().numpy().sum(-1), np.ones(3), 1e-5))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T15 clone/split 继承刷新后的标签与概率")
print("=" * 78)
g = build_model(2)
set_window(g, 0, [0.05, 0.9, 0.05, 0.0], 20, old_prob=UNIFORM[0])   # -> plant
set_window(g, 1, [0.0, 0.05, 0.05, 0.9], 20, old_prob=UNIFORM[0])   # -> rubik_2
g.resolve_unknown_semantics(make_opt(), iteration=1)
pa = argparse.ArgumentParser(); op = OptimizationParams(pa); opt_args = op.extract(pa.parse_args([]))
g.training_setup(opt_args)
n = g._xyz.shape[0]
g._scaling.data = torch.full((n, 3), np.log(1e-4), device="cuda")
grads = torch.zeros((n, 1), device="cuda"); grads[0] = 1.0
g.densify_and_clone(grads, 0.0002, 1.0)
check("T15 clone 子继承已解析标签", int(g._object_id.cpu().numpy()[n]) == int(g._object_id.cpu().numpy()[0]))
check("T15 clone 子继承已解析概率", close(g._object_prob.cpu().numpy()[n], g._object_prob.cpu().numpy()[0]))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T16 checkpoint 恢复 stable_count 与概率不丢失")
print("=" * 78)
g = build_model(2)
set_window(g, 0, [0.05, 0.9, 0.05, 0.0], 20, old_prob=UNIFORM[0])   # -> plant
set_window(g, 1, [0.95, 0.02, 0.02, 0.01], 8, old_prob=BG_OLD)     # bg candidate stable=1
g.resolve_unknown_semantics(make_opt(), iteration=1)
pa = argparse.ArgumentParser(); op = OptimizationParams(pa); opt_args = op.extract(pa.parse_args([]))
g.training_setup(opt_args)
with tempfile.TemporaryDirectory() as td:
    ck = os.path.join(td, "ck.pth")
    torch.save(g.capture(), ck)
    g2 = GaussianModel(3); g2.restore(torch.load(ck), opt_args)
    check("T16 checkpoint 恢复 object_id", torch.equal(g2._object_id, g._object_id))
    check("T16 checkpoint 恢复 object_prob", torch.equal(g2._object_prob, g._object_prob))
    check("T16 checkpoint 恢复 stable_count", torch.equal(g2._semantic_stable_count, g._semantic_stable_count))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T17 刷新时机（warmup/interval 边界；未调用 resolve 状态不变）")
print("=" * 78)
def should_refresh(iteration, warmup, interval):
    return iteration > warmup and (iteration - warmup) % interval == 0
check("T17 iter=1499 不刷新", should_refresh(1499, 1000, 500) is False)
check("T17 iter=1500 首次刷新", should_refresh(1500, 1000, 500) is True)
check("T17 iter=2000 第二次刷新", should_refresh(2000, 1000, 500) is True)
check("T17 iter=1000(warmup) 不刷新", should_refresh(1000, 1000, 500) is False)
g = build_model(1)
set_window(g, 0, [0, 1, 0, 0], 5, old_prob=UNIFORM[0])
snap = g._object_id.clone()
check("T17 未调用 resolve 状态不变", torch.equal(g._object_id, snap)
      and g._semantic_obs_count.sum().item() == 5)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T18 正常返回统计 / 无语义安全跳过")
print("=" * 78)
g = build_model(1)
set_window(g, 0, [0, 1, 0, 0], 5, old_prob=UNIFORM[0])
ret = g.resolve_unknown_semantics(make_opt(), iteration=1500)
check("T18 正常 resolve 返回统计", ret is not None and "unknown_before" in ret)
pcd = BasicPointCloud(points=np.zeros((2, 3), np.float32), colors=np.zeros((2, 3), np.float32), normals=np.zeros((2, 3), np.float32))
g_ns = GaussianModel(3); g_ns.create_from_pcd(pcd, 1.0)
check("T18 无语义 resolve 返回 None", g_ns.resolve_unknown_semantics(make_opt(), iteration=1) is None)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T19 语义 Tensor 仍不在 optimizer")
print("=" * 78)
g = build_model(2)
set_window(g, 0, [0.05, 0.9, 0.05, 0.0], 20, old_prob=UNIFORM[0])
set_window(g, 1, [0.05, 0.05, 0.9, 0.0], 20, old_prob=UNIFORM[0])
g.resolve_unknown_semantics(make_opt(), iteration=1)
pa = argparse.ArgumentParser(); op = OptimizationParams(pa); opt_args = op.extract(pa.parse_args([]))
g.training_setup(opt_args)
ok = True
for opti in [g.optimizer, g.shoptimizer]:
    for grp in opti.param_groups:
        for p in grp["params"]:
            for k in ["_object_id", "_object_prob", "_semantic_confidence", "_semantic_evidence_sum",
                      "_semantic_obs_count", "_semantic_stable_count"]:
                ok &= getattr(g, k) is not p
check("T19 语义 Tensor 不在 optimizer", ok)

# --------------------------------------------------------------------------- #
print("=" * 78)
fails = [n for n, ok in RESULTS if not ok]
print(f"VGER 测试总计 {len(RESULTS)} 项检查, 失败 {len(fails)} 项")
if fails:
    print("失败项:", fails[:20])
    raise SystemExit(1)
print("VGER 语义消解测试全部通过 ✔")
print("=" * 78)
