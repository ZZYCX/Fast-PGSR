#
# SLS (Semantic Lifecycle Synchronization) minimal acceptance test.
#
# Deterministically triggers at least one clone, one split and one prune on a
# synthetic GaussianModel and verifies:
#   - clone/split children inherit long-term semantics from their parent;
#   - children dynamic evidence state is all zero;
#   - after prune, remaining semantics match the keep mask row by row;
#   - optimizer param-group structure (names + count) is unchanged;
#   - without semantic_init_path the original Fast-PGSR path is unchanged.
#
# Run: python scripts/test_sls_sync.py
#

import argparse
import os
import sys

# Allow running as `python scripts/test_sls_sync.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud

CHECK = []


def check(name, cond):
    CHECK.append((name, bool(cond)))
    print(f"[TEST] {'PASS' if cond else 'FAIL'}  {name}")
    assert cond, f"FAIL: {name}"


def make_semantics(N, seed=0):
    rng = np.random.RandomState(seed)
    labels = np.tile([-1, 0, 1, 2, 3], N // 5 + 1)[:N]
    return {
        "point3D_ids": np.arange(1000, 1000 + N, dtype=np.int64),
        "object_id": labels.astype(np.int64),
        "object_probs": rng.dirichlet(np.ones(4), N).astype(np.float32),
        "confidence": rng.rand(N).astype(np.float32),
        "stats": {"raw_num_points": N, "filtered_num_points": N, "npz_total": N,
                  "matched": N, "missing_id_count": 0, "unused_npz": 0,
                  "max_xyz_alignment_error": 1.0e-7},
    }


def seed_optimizer_state(g):
    """Populate Adam state (exp_avg/exp_avg_sq) so replace_tensor_to_optimizer
    (called by densify_and_prune_fastgs) sees a non-None stored_state.
    Real training populates this naturally via optimizer.step() every iteration."""
    for opt in [g.optimizer, g.shoptimizer]:
        for pg in opt.param_groups:
            p = pg["params"][0]
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        opt.step()
        opt.zero_grad()


def build_model(N, with_semantics):
    rng = np.random.RandomState(1)
    pcd = BasicPointCloud(points=rng.rand(N, 3).astype(np.float32),
                          colors=rng.rand(N, 3).astype(np.float32),
                          normals=np.zeros((N, 3), dtype=np.float32))
    g = GaussianModel(3)
    g.create_from_pcd(pcd, 1.0,
                      semantics=make_semantics(N) if with_semantics else None)
    p = argparse.ArgumentParser()
    opt = OptimizationParams(p)
    g.training_setup(opt.extract(p.parse_args([])))
    seed_optimizer_state(g)
    return g, opt


def make_scaling_small_large(g):
    """First half small (clone-eligible), second half large (split-eligible)."""
    N = g._xyz.shape[0]
    log_scale = torch.zeros((N, 3), device="cuda")          # log(1.0) -> scale 1.0
    log_scale[: N // 2] = torch.log(torch.tensor(0.0001)).item()
    g._scaling.data = log_scale


def snap(g):
    return {k: getattr(g, k).detach().cpu().numpy() for k in
            ["_object_id", "_object_prob", "_semantic_confidence",
             "_source_point3D_id", "_semantic_evidence_sum",
             "_semantic_obs_count", "_semantic_stable_count"]}


def assert_children(g, before, child_start, parent_idx_np):
    n_after = g._xyz.shape[0]
    child = np.arange(child_start, n_after)
    pidx = np.asarray(parent_idx_np)
    for field in ["_object_id", "_object_prob", "_semantic_confidence", "_source_point3D_id"]:
        got = getattr(g, field).detach().cpu().numpy()[child]
        exp = before[field][pidx]
        check(f"clone/split 子Gaussian继承 {field} == 父Gaussian",
              np.array_equal(got, exp))
    check("子Gaussian _semantic_evidence_sum 全为0",
          np.allclose(g._semantic_evidence_sum.detach().cpu().numpy()[child], 0.0))
    check("子Gaussian _semantic_obs_count 全为0",
          np.all(g._semantic_obs_count.detach().cpu().numpy()[child] == 0))
    check("子Gaussian _semantic_stable_count 全为0",
          np.all(g._semantic_stable_count.detach().cpu().numpy()[child] == 0))
    g._assert_semantic_alignment("test")
    check("操作后语义长度与Gaussian一致", True)


def opt_structure(g):
    return ([grp["name"] for grp in g.optimizer.param_groups],
            [len(grp["params"]) for grp in g.optimizer.param_groups],
            [grp["name"] for grp in g.shoptimizer.param_groups],
            [len(grp["params"]) for grp in g.shoptimizer.param_groups])


# --------------------------------------------------------------------------- #
print("=" * 70)
print("Test A: 语义模型 — clone / split / fastgs / prune / final_prune")
print("=" * 70)
g, opt = build_model(40, with_semantics=True)
make_scaling_small_large(g)
N0 = g._xyz.shape[0]
s0 = snap(g)
print(f"初始 N={N0}, object_id 分布 {g._semantic_class_counts()}")

# ---- A1: vanilla clone (first 20 points are small-scale -> clone) ----
before = snap(g)
n_before = g._xyz.shape[0]
parent_idx = np.arange(0, 20)
g.densify_and_clone(torch.ones((n_before, 1), device="cuda"), 0.0002, 1.0)
n_after = g._xyz.shape[0]
check("A1 clone 实际新增点数==20", n_after == n_before + 20)
assert_children(g, before, n_before, parent_idx)

# ---- A2: vanilla split (second half large-scale -> split, N=2) ----
before = snap(g)
n_before = g._xyz.shape[0]
parent_idx = np.arange(20, 40)                      # the 20 large-scale points
g.densify_and_split(torch.ones((n_before, 1), device="cuda"), 0.0002,
                    torch.zeros((n_before, 1), device="cuda"), 0.0012,
                    1.0, torch.zeros((n_before,), device="cuda"), N=2)
check("A2 split 净增点数==20 (40个孩子 - 20个父点)",
      g._xyz.shape[0] == n_before + 20)
# split children are the last 2*len(parents) rows (parents pruned before them).
# Order is block-repeat: [parent_block, parent_block], matching `[selected].repeat(N,1)`.
child_start = g._xyz.shape[0] - 2 * len(parent_idx)
pidx_children = np.tile(parent_idx, 2)
assert_children(g, before, child_start, pidx_children)

# ---- A3: fastgs clone + split (direct) ----
before = snap(g)
n_before = g._xyz.shape[0]
metric_mask = torch.ones((n_before,), dtype=bool, device="cuda")
scaling = g.get_scaling.detach()
is_small = scaling.max(dim=-1).values <= 0.001
is_large = scaling.max(dim=-1).values > 0.001
g.densify_and_clone_fastgs(metric_mask, is_small)
n_clone = g._xyz.shape[0]
check("A3 fastgs clone 增加点数==小scale点数",
      n_clone - n_before == int(is_small.sum()))
child_s = n_before
assert_children(g, before, child_s, is_small.nonzero().flatten().cpu().numpy())

metric_mask = torch.ones((g._xyz.shape[0],), dtype=bool, device="cuda")   # refresh length
before = snap(g)
n_before = g._xyz.shape[0]
is_large = g.get_scaling.detach().max(dim=-1).values > 0.001
g.densify_and_split_fastgs(metric_mask, is_large, N=2)
check("A3 fastgs split 净增点数==大scale点数",
      g._xyz.shape[0] == n_before + int(is_large.sum()))
child_start = g._xyz.shape[0] - 2 * int(is_large.sum())
pidx_children = np.tile(is_large.nonzero().flatten().cpu().numpy(), 2)
assert_children(g, before, child_start, pidx_children)

# ---- A4: densify_and_prune_fastgs composed (clone + split + prune) ----
cur_n = g._xyz.shape[0]
g.xyz_gradient_accum = torch.ones((cur_n, 1), device="cuda")
g.xyz_gradient_accum_abs = torch.ones((cur_n, 1), device="cuda")
g.denom = torch.ones((cur_n, 1), device="cuda")
g.denom_abs = torch.ones((cur_n, 1), device="cuda")
g._opacity.data[torch.arange(0, cur_n, 7, device="cuda")] = -10.0   # low opacity -> prune
before = snap(g)
n_before = g._xyz.shape[0]
imp = torch.full((cur_n,), 10.0, device="cuda")
prun = torch.zeros((cur_n,), device="cuda")
g.densify_and_prune_fastgs(max_screen_size=0, min_opacity=0.005, extent=1.0,
                           radii=None, args=opt, importance_score=imp, pruning_score=prun)
n_after = g._xyz.shape[0]
check("A4 fastgs 组合操作后语义仍对齐", True)
g._assert_semantic_alignment("A4")

# ---- A5: prune_points — mask row-by-row alignment ----
before = snap(g)
n_before = g._xyz.shape[0]
mask = torch.zeros((n_before,), dtype=bool, device="cuda")
mask[::3] = True                                   # remove every 3rd point
keep = (~mask).cpu().numpy()
g.prune_points(mask)
check("A5 prune 实际删除点数==mask.sum()", n_before - g._xyz.shape[0] == int(mask.sum()))
for field in ["_object_id", "_object_prob", "_semantic_confidence",
              "_source_point3D_id", "_semantic_evidence_sum",
              "_semantic_obs_count", "_semantic_stable_count"]:
    got = getattr(g, field).detach().cpu().numpy()
    exp = before[field][keep]
    check(f"A5 prune 后 {field} 与保留mask逐行一致", np.array_equal(got, exp))

# ---- A6: final_prune_fastgs (score-based prune) ----
cur_n = g._xyz.shape[0]
prun = torch.zeros((cur_n,), device="cuda")
prun[::5] = 1.0
before_n = g._xyz.shape[0]
g.final_prune_fastgs(min_opacity=0.0, pruning_score=prun)
check("A6 final_prune 实际删除点数==score标记数",
      before_n - g._xyz.shape[0] == int((prun > 0.9).sum()))
g._assert_semantic_alignment("A6")

# ---- A7: optimizer structure unchanged ----
names0, sizes0, shnames0, shsizes0 = opt_structure(g)
check("A7 optimizer 参数组名称与数量不变",
      names0 == ["xyz", "knn_f", "f_dc", "opacity", "scaling", "rotation"]
      and sizes0 == [1] * 6 and shnames0 == ["f_rest"] and shsizes0 == [1])
check("A7 语义 Tensor 不在 optimizer 参数组", True)

# --------------------------------------------------------------------------- #
print("=" * 70)
print("Test B: 无语义路径 — 原 Fast-PGSR 行为不变")
print("=" * 70)
g2, opt2 = build_model(40, with_semantics=False)
make_scaling_small_large(g2)
names0 = opt_structure(g2)
g2.densify_and_clone(torch.ones((40, 1), device="cuda"), 0.0002, 1.0)
g2.densify_and_split(torch.ones((g2._xyz.shape[0], 1), device="cuda"), 0.0002,
                     torch.zeros((g2._xyz.shape[0], 1), device="cuda"), 0.0012,
                     1.0, torch.zeros((g2._xyz.shape[0],), device="cuda"), N=2)
b_mask = torch.zeros((g2._xyz.shape[0],), dtype=bool, device="cuda")
b_mask[::2] = True
g2.prune_points(b_mask)
check("B 未启用语义时 _semantics_attached==False", g2._semantics_attached is False)
check("B 语义张量保持空", g2._object_id.numel() == 0
      and g2._semantic_evidence_sum.numel() == 0
      and g2._semantic_obs_count.numel() == 0
      and g2._semantic_stable_count.numel() == 0)
check("B optimizer 参数组结构不变", opt_structure(g2) == names0)
check("B 无语义路径正常运行（未抛错）", True)

# --------------------------------------------------------------------------- #
print("=" * 70)
fails = [n for n, ok in CHECK if not ok]
print(f"总计 {len(CHECK)} 项检查, 失败 {len(fails)} 项")
if fails:
    print("失败项:", fails)
    raise SystemExit(1)
print("SLS 验收测试全部通过 ✔")
print("=" * 70)
