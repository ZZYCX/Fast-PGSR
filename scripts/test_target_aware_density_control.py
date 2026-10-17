#
# TADC (Target-Aware Density Control) acceptance test.
#
# An independent Oracle computes the expected semantic grow mask (never calling the
# production get_semantic_grow_mask) and the expected gated densify candidates, then
# verifies the production output row-by-row on all four densify paths.
#
# Run: python scripts/test_target_aware_density_control.py  (needs CUDA)
#

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[TADC-T] {'PASS' if cond else 'FAIL'}  {name}{(' | ' + detail) if detail else ''}")
    assert cond, f"TADC-T FAIL: {name}"


# ---- independent Oracle --------------------------------------------------- #
def ref_grow(object_id, object_prob, threshold):
    oid = np.asarray(object_id)
    prob = np.asarray(object_prob, np.float32)
    known = (oid == 1) | (oid == 2) | (oid == 3)
    unknown = (oid == -1)
    p_target = prob[:, 1] + prob[:, 2] + prob[:, 3]
    return known | (unknown & (p_target >= float(threshold)))


def tadc_opt(enabled=True, threshold=0.4, debug=False):
    return argparse.Namespace(enable_semantic_density_control=enabled,
                              semantic_unknown_grow_threshold=threshold,
                              semantic_density_debug=debug,
                              semantic_density_debug_interval=100)


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
    pa = argparse.ArgumentParser(); op = OptimizationParams(pa); opt = op.extract(pa.parse_args([]))
    g.training_setup(opt)
    for o in [g.optimizer, g.shoptimizer]:
        for pg in o.param_groups:
            p = pg["params"][0]
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        o.step(); o.zero_grad()
    return g, opt


def set_sem(g, object_id, probs):
    n = g._xyz.shape[0]
    g._object_id = torch.tensor(np.asarray(object_id, np.int64), dtype=torch.int64, device="cuda")
    g._object_prob = torch.tensor(np.asarray(probs, np.float32), dtype=torch.float32, device="cuda")


# probe configuration: idx0 unknown high p_target, idx1 bg, idx2/3/4 targets, idx5 unknown low
OID = np.array([-1, 0, 1, 2, 3, -1], np.int64)
PROB = np.array([[0.3, 0.5, 0.1, 0.1],      # unknown p_target 0.7 (grow)
                 [0.95, 0.02, 0.02, 0.01],  # background (blocked)
                 [0.1, 0.8, 0.05, 0.05],    # plant
                 [0.1, 0.05, 0.8, 0.05],    # rubik_1
                 [0.1, 0.05, 0.05, 0.8],    # rubik_2
                 [0.9, 0.04, 0.03, 0.03]], np.float32)   # unknown p_target 0.1 (blocked)
GROW_04 = ref_grow(OID, PROB, 0.4)          # {0,2,3,4}


# --------------------------------------------------------------------------- #
print("=" * 78)
print("T1  grow mask 基本规则（plant/rubik=True, bg=False, unknown按p_target）")
print("=" * 78)
g, _ = build_model(6)
set_sem(g, OID, PROB)
got = g.get_semantic_grow_mask(0.4).cpu().numpy()
check("T1 生产 grow mask == 独立参考", np.array_equal(got, GROW_04),
      f"got={got.tolist()} ref={GROW_04.tolist()}")
check("T1 background 永远 False", not got[1])
check("T1 目标全部 True", got[2] and got[3] and got[4])
check("T1 unknown 高 p_target True", got[0])
check("T1 unknown 低 p_target False", not got[5])

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T2  阈值边界（精确0.40 / <0.40 / threshold=0 / threshold=1）")
print("=" * 78)
g, _ = build_model(3)
set_sem(g, [-1, -1, -1],
        [[0.6, 0.2, 0.1, 0.1],      # p_target = 0.4 exactly, sum = 1
         [0.3, 0.5, 0.1, 0.1],      # p_target = 0.7
         [0.65, 0.20, 0.10, 0.05]])  # p_target = 0.35
m = g.get_semantic_grow_mask(0.4).cpu().numpy()
check("T2 p_target==0.40 边界允许", bool(m[0]))
check("T2 p_target=0.70 允许", bool(m[1]))
check("T2 p_target=0.35 < 0.40 拒绝", not m[2])
check("T2 threshold=0: 所有unknown允许", g.get_semantic_grow_mask(0.0).cpu().numpy().all())
g2, _ = build_model(2)
set_sem(g2, [-1, 1], [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]])  # p_target 0.75 / known target
m2 = g2.get_semantic_grow_mask(1.0).cpu().numpy()
check("T2 threshold=1: unknown p_target=0.75 拒绝", not m2[0])
check("T2 threshold=1: 已知目标仍允许", bool(m2[1]))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T3  final_mask == original_mask & semantic_grow_mask")
print("=" * 78)
g, _ = build_model(6)
set_sem(g, OID, PROB)
orig = torch.tensor([True, True, False, True, False, True], dtype=torch.bool, device="cuda")
final = g._apply_semantic_grow_gate(orig, tadc_opt(), "clone")
check("T3 final == orig & grow", torch.equal(final, orig & torch.from_numpy(GROW_04).cuda()))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T4/T5/T6/T7  原始条件优先；bg被拒；unknown按p_target")
print("=" * 78)
# T4: target that fails original grad condition must NOT be selected
g, _ = build_model(6)
set_sem(g, OID, PROB)
n = g._xyz.shape[0]
g._scaling.data = torch.full((n, 3), np.log(1e-4), device="cuda")       # all clone-eligible
grads = torch.zeros((n, 1), device="cuda"); grads[[0, 1, 3, 5]] = 1.0   # plant idx2 grad=0 -> NOT original
g.densify_and_clone(grads, 0.0002, 1.0, tadc_opt())
orig = np.array([0, 1, 3, 5])                    # grad-selected (scaling ok everywhere)
final = orig[GROW_04[orig]]                      # {0,3}
check("T4 仅 final 候选被clone", g._xyz.shape[0] == n + len(final),
      f"got={g._xyz.shape[0] - n} exp={len(final)}")
children = g._source_point3D_id[n:].cpu().numpy()
check("T4 不满足原条件的plant(idx2)未被选中", 1002 not in children.tolist())
check("T4 满足条件的target(idx3 rubik_1)被选中", 1003 in children.tolist())

# T5: background blocked despite original condition (bg idx1 grad-high, scaling small)
g, _ = build_model(6)
set_sem(g, OID, PROB)
n = g._xyz.shape[0]
g._scaling.data = torch.full((n, 3), np.log(1e-4), device="cuda")
grads = torch.ones((n, 1), device="cuda")      # ALL original candidates
g.densify_and_clone(grads, 0.0002, 1.0, tadc_opt())
children = g._source_point3D_id[n:].cpu().numpy()
allowed = np.nonzero(GROW_04)[0]
check("T5 background 被拒（子代只含允许父）",
      set(children.tolist()) == set((1000 + allowed).tolist()))
check("T5 background 未被删除（原bg仍存在）", int((g._object_id == 0).sum()) >= 1)

# T6: unknown high p_target + original -> allowed
check("T6 unknown高p_target允许", int((g._source_point3D_id[n:].cpu().numpy() == 1000).sum()) == 1)
# T7: unknown low p_target blocked
check("T7 unknown低p_target被拒", int((g._source_point3D_id[n:].cpu().numpy() == 1005).sum()) == 0)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T8/T9 vanilla clone / split 路径逐行正确")
print("=" * 78)
g, _ = build_model(6)
set_sem(g, OID, PROB)
n = g._xyz.shape[0]
g._scaling.data = torch.full((n, 3), np.log(1e-4), device="cuda")
g.densify_and_clone(torch.ones((n, 1), device="cuda"), 0.0002, 1.0, tadc_opt())
children = g._source_point3D_id[n:].cpu().numpy()
check("T8 vanilla clone 子代 == 门控父 {0,2,3,4}",
      set(children.tolist()) == set((1000 + np.nonzero(GROW_04)[0]).tolist()))

g, _ = build_model(6)
set_sem(g, OID, PROB)
n = g._xyz.shape[0]
g._scaling.data = torch.full((n, 3), 0.0, device="cuda")     # all large -> split-eligible
g.densify_and_split(torch.ones((n, 1), device="cuda"), 0.0002,
                    torch.zeros((n, 1), device="cuda"), 0.0012, 1.0,
                    torch.zeros((n,), device="cuda"), N=2, args=tadc_opt())
allowed = np.nonzero(GROW_04)[0]
check("T9 vanilla split 净增==|grow|", g._xyz.shape[0] == n + len(allowed),
      f"got={g._xyz.shape[0]} exp={n + len(allowed)}")
# row-correct: children are the last 2*|grow| rows, block-repeated gated parents
child = g._source_point3D_id[g._xyz.shape[0] - 2 * len(allowed):].cpu().numpy()
check("T9 vanilla split 子代 source 与门控父逐行一致",
      np.array_equal(child, np.tile(1000 + allowed, 2)),
      f"got={child.tolist()} exp={np.tile(1000 + allowed, 2).tolist()}")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T10/T11 fastgs clone / split 路径逐行正确")
print("=" * 78)
g, _ = build_model(6)
set_sem(g, OID, PROB)
n = g._xyz.shape[0]
m_all = torch.ones((n,), dtype=torch.bool, device="cuda")
g.densify_and_clone_fastgs(m_all, m_all, tadc_opt())
children = g._source_point3D_id[n:].cpu().numpy()
check("T10 fastgs clone 子代 == 门控父", set(children.tolist()) == set((1000 + np.nonzero(GROW_04)[0]).tolist()))

g, _ = build_model(6)
set_sem(g, OID, PROB)
n = g._xyz.shape[0]
g.densify_and_split_fastgs(m_all, m_all, N=2, args=tadc_opt())
check("T11 fastgs split 净增==|grow|", g._xyz.shape[0] == n + len(allowed))
child = g._source_point3D_id[g._xyz.shape[0] - 2 * len(allowed):].cpu().numpy()
check("T11 fastgs split 子代 source 与门控父逐行一致",
      np.array_equal(child, np.tile(1000 + allowed, 2)))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T12  新子Gaussian几何父索引与SLS语义父索引一致（knn_f 见证）")
print("=" * 78)
g, _ = build_model(6)
set_sem(g, OID, PROB)
n = g._xyz.shape[0]
parent_knn = g._knn_f.detach().cpu().numpy()
g._scaling.data = torch.full((n, 3), np.log(1e-4), device="cuda")
g.densify_and_clone(torch.ones((n, 1), device="cuda"), 0.0002, 1.0, tadc_opt())
child_knn = g._knn_f[n:].detach().cpu().numpy()
allowed = np.nonzero(GROW_04)[0]
check("T12 几何knn_f顺序==语义父索引顺序",
      np.array_equal(child_knn, parent_knn[allowed]))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T13  VGER 更新前后 grow 状态（unknown高p→target→bg）")
print("=" * 78)
g, _ = build_model(1)
set_sem(g, [-1], [[0.2, 0.6, 0.1, 0.1]])     # unknown p_target 0.8
check("T13 unknown高p_target 可生长", bool(g.get_semantic_grow_mask(0.4).cpu().numpy()[0]))
set_sem(g, [1], [[0.1, 0.8, 0.05, 0.05]])    # VGER -> plant
check("T13 转plant 后仍可生长", bool(g.get_semantic_grow_mask(0.4).cpu().numpy()[0]))
set_sem(g, [0], [[0.95, 0.02, 0.02, 0.01]])  # VGER -> background
check("T13 转bg 后不可生长", not g.get_semantic_grow_mask(0.4).cpu().numpy()[0])
# actual VGER plant conversion then grow check
g, _ = build_model(1)
set_sem(g, [-1], [[0.25, 0.25, 0.25, 0.25]])
g._semantic_evidence_sum = torch.tensor([[0.0, 20.0, 0.0, 0.0]], dtype=torch.float32, device="cuda")
g._semantic_obs_count = torch.tensor([20], dtype=torch.int64, device="cuda")
vger = argparse.Namespace(semantic_prior_strength=5.0, semantic_target_min_observations=5,
                          semantic_target_confidence=0.75, semantic_target_margin=0.20,
                          semantic_background_min_observations=8, semantic_background_confidence=0.95,
                          semantic_background_max_target_prob=0.05, semantic_background_stable_windows=2,
                          semantic_resolution_debug=False)
g.resolve_unknown_semantics(vger, iteration=1)
check("T13 VGER 实际转为 plant", int(g._object_id.cpu().numpy()[0]) == 1)
check("T13 转plant后 grow 仍允许", bool(g.get_semantic_grow_mask(0.4).cpu().numpy()[0]))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T14/T15 无语义 / 未启用 → final == original")
print("=" * 78)
rng = np.random.RandomState(0)
pcd = BasicPointCloud(points=(rng.rand(6, 3).astype(np.float32) + 5.0),
                      colors=rng.rand(6, 3).astype(np.float32), normals=np.zeros((6, 3), np.float32))
g_ns = GaussianModel(3); g_ns.create_from_pcd(pcd, 1.0)          # no semantics
pa = argparse.ArgumentParser(); op = OptimizationParams(pa); opt_args = op.extract(pa.parse_args([]))
g_ns.training_setup(opt_args)
check("T14 无语义 grow mask 全True", g_ns.get_semantic_grow_mask(0.4).cpu().numpy().all())
nn = g_ns._xyz.shape[0]
g_ns._scaling.data = torch.full((nn, 3), np.log(1e-4), device="cuda")
g_ns.densify_and_clone(torch.ones((nn, 1), device="cuda"), 0.0002, 1.0, tadc_opt(enabled=True))
check("T14 无语义+TADC启用 → 全部clone（final==original）", g_ns._xyz.shape[0] == nn + nn,
      f"got={g_ns._xyz.shape[0]} exp={2 * nn}")
g, _ = build_model(6)
set_sem(g, OID, PROB)
n = g._xyz.shape[0]
g._scaling.data = torch.full((n, 3), np.log(1e-4), device="cuda")
g.densify_and_clone(torch.ones((n, 1), device="cuda"), 0.0002, 1.0, tadc_opt(enabled=False))
check("T15 未启用TADC → 全部clone（原行为）", g._xyz.shape[0] == n + 6)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T16  object_prob 异常必须报错")
print("=" * 78)
g, _ = build_model(1)
for bad, name in [([0.0, 0.5, 0.5, float('nan')], 'NaN'),
                  ([0.0, -0.2, 0.6, 0.6], '负值'),
                  ([0.3, 0.5, 0.5, 0.3], '和≠1')]:
    set_sem(g, [-1], [bad])
    try:
        g.get_semantic_grow_mask(0.4)
        check(f"T16 {name} 应报错", False)
    except AssertionError:
        check(f"T16 {name} 报错", True)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T17  语义Tensor不进入optimizer")
print("=" * 78)
g, _ = build_model(6)
set_sem(g, OID, PROB)
n = g._xyz.shape[0]
g._scaling.data = torch.full((n, 3), np.log(1e-4), device="cuda")
g.densify_and_clone(torch.ones((n, 1), device="cuda"), 0.0002, 1.0, tadc_opt())
ok = True
for o in [g.optimizer, g.shoptimizer]:
    for pg in o.param_groups:
        for p in pg["params"]:
            for k in ["_object_id", "_object_prob", "_semantic_confidence", "_semantic_evidence_sum",
                      "_semantic_obs_count", "_semantic_stable_count"]:
                ok &= getattr(g, k) is not p
check("T17 语义Tensor不在optimizer", ok)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T18  SLS / SSP 现有测试仍通过（TADC 改动共享 densify 路径）")
print("=" * 78)
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for suite in ["test_sls_sync.py", "test_semantic_persistence.py"]:
    r = subprocess.run([sys.executable, os.path.join(repo, "scripts", suite)],
                       cwd=repo, capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else f"rc={r.returncode}"
    check(f"T18 {suite} 通过", r.returncode == 0, tail[:80])

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T19/T20 TADC 不修改语义状态；background 不被删除")
print("=" * 78)
g, _ = build_model(6)
set_sem(g, OID, PROB)
snap = {k: getattr(g, k).clone() for k in
        ["_object_id", "_object_prob", "_semantic_confidence", "_semantic_evidence_sum",
         "_semantic_obs_count", "_semantic_stable_count"]}
_ = g.get_semantic_grow_mask(0.4)
check("T19 get_semantic_grow_mask 不修改语义状态",
      all(torch.equal(getattr(g, k), v) for k, v in snap.items()))
n = g._xyz.shape[0]
bg_before = int((g._object_id == 0).sum())
g._scaling.data = torch.full((n, 3), np.log(1e-4), device="cuda")
g.densify_and_clone(torch.ones((n, 1), device="cuda"), 0.0002, 1.0, tadc_opt())
# existing rows unchanged (SLS appends children only)
for k in ["_object_id", "_object_prob", "_semantic_confidence", "_semantic_evidence_sum",
          "_semantic_obs_count", "_semantic_stable_count"]:
    check(f"T19 densify 后已有行 {k} 不变", torch.equal(getattr(g, k)[:n], snap[k]))
check("T20 background Gaussian 仍存在且未因clone增长",
      int((g._object_id == 0).sum()) == bg_before)

# --------------------------------------------------------------------------- #
print("=" * 78)
fails = [n for n, ok in RESULTS if not ok]
print(f"TADC 测试总计 {len(RESULTS)} 项检查, 失败 {len(fails)} 项")
if fails:
    print("失败项:", fails[:20])
    raise SystemExit(1)
print("TADC 目标感知密度控制测试全部通过 ✔")
print("=" * 78)
