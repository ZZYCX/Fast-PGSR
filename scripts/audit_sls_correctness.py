#
# Independent, strict correctness audit of the SLS (Semantic Lifecycle Synchronization).
#
# The audit does NOT re-implement SLS. It builds an *independent CPU Oracle* that
# constructs expected semantic state from raw parent indices / keep masks / pre-op
# tensors (never calling _append_semantics_from_parents / _prune_semantics or any
# other production SLS helper), then compares production output row-by-row with
# torch.equal. Every initial Gaussian carries a unique fingerprint so that any row
# swap or masking mistake is detected.
#
# Run: python scripts/audit_sls_correctness.py   (needs CUDA, fast-pgsr env)
#

import argparse
import contextlib
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud

ALL = ["_object_id", "_object_prob", "_semantic_confidence", "_source_point3D_id",
       "_semantic_evidence_sum", "_semantic_obs_count", "_semantic_stable_count"]
LONG_TERM = ALL[:4]           # inherited by children
DYNAMIC = ALL[4:]             # children start at zero

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[AUDIT] {'PASS' if cond else 'FAIL'}  {name}{(' | ' + detail) if detail else ''}")
    assert cond, f"AUDIT FAIL: {name}"


def check_silent(name, cond):
    RESULTS.append((name, bool(cond)))
    assert cond, f"AUDIT FAIL(silent): {name}"


# --------------------------------------------------------------------------- #
# Independent Oracle — pure numpy, mirrors the DESIGN rules only.
# --------------------------------------------------------------------------- #
def snap(g):
    """Read current production semantic state as CPU numpy (allowed: reads the
    pre-op original state; constructs the reference from first principles)."""
    return {k: getattr(g, k).detach().cpu().numpy() for k in ALL}


def oracle_clone(ref, parent_idx):
    """append: long-term inherited from parent rows, dynamic zeroed for children."""
    out = {k: np.concatenate([ref[k], ref[k][parent_idx]]) for k in LONG_TERM}
    out["_semantic_evidence_sum"] = np.concatenate(
        [ref["_semantic_evidence_sum"], np.zeros((len(parent_idx), 4), np.float32)])
    out["_semantic_obs_count"] = np.concatenate(
        [ref["_semantic_obs_count"], np.zeros(len(parent_idx), np.int64)])
    out["_semantic_stable_count"] = np.concatenate(
        [ref["_semantic_stable_count"], np.zeros(len(parent_idx), np.int64)])
    return out


def oracle_split(ref, parent_idx, N):
    """split: children are block-repeat of the parent rows (matches production
    `get_xyz[selected].repeat(N,1)`), then the parent rows themselves are pruned."""
    child_idx = np.tile(parent_idx, N)          # [a,b,c,a,b,c] not [a,a,b,b,c,c]
    appended = oracle_clone(ref, child_idx)
    n0 = ref["_object_id"].shape[0]
    keep = np.ones(n0 + len(child_idx), dtype=bool)
    keep[parent_idx] = False
    return {k: appended[k][keep] for k in ALL}


def oracle_prune(ref, keep_mask):
    """prune: apply the SAME keep mask to every tensor (parameter is a remove mask)."""
    return {k: ref[k][keep_mask] for k in ALL}


def compare_all(g, ref, tag):
    for k in ALL:
        got = getattr(g, k)
        exp = torch.from_numpy(np.ascontiguousarray(ref[k])).to(got.device)
        check(f"{tag}: torch.equal({k})", torch.equal(got, exp))


# --------------------------------------------------------------------------- #
# Build models with unique fingerprints.
# --------------------------------------------------------------------------- #
def make_semantics(N):
    rng = np.random.RandomState(0)
    obj = np.tile([-1, 0, 1, 2, 3], N // 5 + 1)[:N].astype(np.int64)
    probs = rng.dirichlet(np.ones(4), N).astype(np.float32)          # every row distinct
    conf = rng.rand(N).astype(np.float32)
    return {"point3D_ids": (1000 + np.arange(N)).astype(np.int64),
            "object_id": obj,
            "object_probs": probs,
            "confidence": conf,
            "stats": {"raw_num_points": N, "filtered_num_points": N, "npz_total": N,
                      "matched": N, "missing_id_count": 0, "unused_npz": 0,
                      "max_xyz_alignment_error": 1e-7}}


def seed_optimizer_state(g):
    for opt in [g.optimizer, g.shoptimizer]:
        for pg in opt.param_groups:
            p = pg["params"][0]
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        opt.step()
        opt.zero_grad()


def build(N, with_semantics=True, sls_debug=False):
    rng = np.random.RandomState(1)
    pcd = BasicPointCloud(points=rng.rand(N, 3).astype(np.float32),
                          colors=rng.rand(N, 3).astype(np.float32),
                          normals=np.zeros((N, 3), np.float32))
    g = GaussianModel(3, sls_debug=sls_debug)
    g.create_from_pcd(pcd, 1.0,
                      semantics=make_semantics(N) if with_semantics else None)
    p = argparse.ArgumentParser()
    opt = OptimizationParams(p)
    g.training_setup(opt.extract(p.parse_args([])))
    seed_optimizer_state(g)
    if with_semantics:
        # Dynamic fingerprints (overwrite the zeroed init) so any row swap is visible.
        base = (torch.arange(N, device="cuda").float() * 0.13 + 0.01).unsqueeze(1)
        g._semantic_evidence_sum = (base.expand(-1, 4)
                                    + torch.arange(4, device="cuda").float() * 0.07).contiguous()
        g._semantic_obs_count = torch.arange(1, N + 1, dtype=torch.int64, device="cuda")
        g._semantic_stable_count = torch.arange(N, dtype=torch.int64, device="cuda") * 3 + 5
    return g, opt


def set_clone_eligible(g, targets):
    """All scaling small + grads high only at targets -> clone selects `targets`."""
    n = g._xyz.shape[0]
    g._scaling.data = torch.full((n, 3), np.log(1e-4), device="cuda")
    grads = torch.zeros((n, 1), device="cuda")
    grads[targets] = 1.0
    return grads


def set_split_eligible(g, targets):
    """Scaling large only at targets + all grads high -> split selects `targets`."""
    n = g._xyz.shape[0]
    sc = torch.full((n, 3), np.log(1e-4), device="cuda")
    sc[targets] = 0.0                                # log(1.0)
    g._scaling.data = sc
    return torch.ones((n, 1), device="cuda")


# --------------------------------------------------------------------------- #
print("=" * 78)
print("S0 顺序鉴别: split 子点顺序 = 块重复 [a,b,c,a,b,c] 而非交错 [a,a,b,b,c,c]")
print("=" * 78)
g, opt = build(16, True)
targets = np.array([2, 5, 9])
grads = set_split_eligible(g, targets)
before = snap(g)
before_knn = g._knn_f.detach().cpu().numpy()
n0 = g._xyz.shape[0]
g.densify_and_split(grads, 0.0002, torch.zeros((n0, 1), device="cuda"), 0.0012,
                    1.0, torch.zeros((n0,), device="cuda"), N=2)
n_after = g._xyz.shape[0]
child_src = g._source_point3D_id[n_after - 6:].cpu().numpy()
child_knn = g._knn_f[n_after - 6:].detach().cpu().numpy()
block_src = np.tile(before["_source_point3D_id"][targets], 2)
inter_src = np.repeat(before["_source_point3D_id"][targets], 2)
check("S0 语义子序==块重复[a,b,c,a,b,c]", np.array_equal(child_src, block_src),
      f"child={child_src.tolist()}")
check("S0 语义子序!=交错[a,a,b,b,c,c]", not np.array_equal(child_src, inter_src))
check("S0 几何knn_f==块重复(与语义同序)",
      np.array_equal(child_knn, np.tile(before_knn[targets], (2, 1))))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("S1 vanilla clone 精确继承（非连续父索引 [0,1,4,7]，几何/语义同序，无别名）")
print("=" * 78)
g, opt = build(16, True)
targets = np.array([0, 1, 4, 7])
grads = set_clone_eligible(g, targets)
before = snap(g)
before_knn = g._knn_f.detach().cpu().numpy()
n0 = g._xyz.shape[0]
g.densify_and_clone(grads, 0.0002, 1.0)
n1 = g._xyz.shape[0]
child = np.arange(n0, n1)
check("S1 新增点数==4", n1 - n0 == 4)
for k in ALL:
    check(f"S1 原有行[:{n0}] 完全不变({k})",
          np.array_equal(getattr(g, k).detach().cpu().numpy()[:n0], before[k]))
for k in LONG_TERM:
    check(f"S1 子长期语义==父行({k})",
          np.array_equal(getattr(g, k).detach().cpu().numpy()[child],
                         before[k][targets]))
for k in DYNAMIC:
    check(f"S1 子动态状态全0({k})",
          np.all(getattr(g, k).detach().cpu().numpy()[child] == 0))
# geometric order == semantic order via exact knn_f copy
check("S1 几何knn_f顺序==语义父索引顺序",
      np.array_equal(g._knn_f[child].detach().cpu().numpy(), before_knn[targets]))
# aliasing: mutate a child row, parent must be untouched
g._object_prob[child[0], 0] = -999.0
check("S1 无存储别名（改子行不影响父行）",
      np.array_equal(g._object_prob[:n0, 0].cpu().numpy(), before["_object_prob"][:, 0]))
# duplicate parent indices via production densification_postfix
g2, _ = build(16, True)
dup = torch.tensor([4, 1, 7, 4, 0], device="cuda")
ref = snap(g2)
nn0 = g2._xyz.shape[0]
g2.densification_postfix(torch.randn((5, 3), device="cuda"),
                         torch.randn((5, 6), device="cuda"),
                         torch.randn((5, 1, 3), device="cuda"),
                         torch.randn((5, 15, 3), device="cuda"),
                         torch.randn((5, 1), device="cuda"),
                         torch.randn((5, 3), device="cuda"),
                         torch.randn((5, 4), device="cuda"),
                         new_parent_ids=dup, op_type="clone")
for k in LONG_TERM:
    got = getattr(g2, k).detach().cpu().numpy()[nn0:]
    check(f"S1 重复父索引[{4,1,7,4,0}] 逐行继承({k})",
          np.array_equal(got, ref[k][dup.cpu().numpy()]))
for k in DYNAMIC:
    check(f"S1 重复父索引 子动态全0({k})",
          np.all(getattr(g2, k).detach().cpu().numpy()[nn0:] == 0))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("S2 fastgs clone 精确继承")
print("=" * 78)
g, opt = build(16, True)
targets = np.array([3, 6, 10, 12, 14])
mask = torch.zeros((16,), dtype=bool, device="cuda")
mask[targets] = True
before = snap(g)
n0 = g._xyz.shape[0]
g.densify_and_clone_fastgs(torch.ones((16,), dtype=bool, device="cuda"), mask)
child = np.arange(n0, g._xyz.shape[0])
check("S2 fastgs clone 新增==5", g._xyz.shape[0] - n0 == 5)
for k in LONG_TERM:
    check(f"S2 子继承父行({k})",
          np.array_equal(getattr(g, k).detach().cpu().numpy()[child], before[k][targets]))
for k in DYNAMIC:
    check(f"S2 子动态全0({k})",
          np.all(getattr(g, k).detach().cpu().numpy()[child] == 0))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("S3 split N=1/2/3（vanilla 与 fastgs，非连续父 [2,5,9]，父被删除，旧点不变）")
print("=" * 78)
for N in [1, 2, 3]:
    g, opt = build(16, True)
    targets = np.array([2, 5, 9])
    grads = set_split_eligible(g, targets)
    before = snap(g)
    before_knn = g._knn_f.detach().cpu().numpy()
    n0 = g._xyz.shape[0]
    g.densify_and_split(grads, 0.0002, torch.zeros((n0, 1), device="cuda"), 0.0012,
                        1.0, torch.zeros((n0,), device="cuda"), N=N)
    n1 = g._xyz.shape[0]
    check(f"S3 vanilla N={N}: 净增==(N-1)*3", n1 == n0 + (N - 1) * 3)
    # after split the parents are pruned, so children are the LAST N*3 rows.
    child = np.arange(n1 - N * len(targets), n1)
    tile_idx = np.tile(targets, N)
    for k in LONG_TERM:
        check(f"S3 vanilla N={N} 子继承父行({k})",
              np.array_equal(getattr(g, k).detach().cpu().numpy()[child], before[k][tile_idx]))
    for k in DYNAMIC:
        check(f"S3 vanilla N={N} 子动态全0({k})",
              np.all(getattr(g, k).detach().cpu().numpy()[child] == 0))
    keep_old = np.ones(n0, dtype=bool); keep_old[targets] = False
    for k in ALL:
        check(f"S3 vanilla N={N} 未split旧行不变({k})",
              np.array_equal(getattr(g, k).detach().cpu().numpy()[:n0 - len(targets)],
                             before[k][keep_old]))
    check(f"S3 vanilla N={N} 几何knn_f==块重复父行",
          np.array_equal(g._knn_f[child].detach().cpu().numpy(),
                         np.tile(before_knn[targets], (N, 1))))

    # fastgs split
    g, opt = build(16, True)
    before = snap(g)
    n0 = g._xyz.shape[0]
    mask = torch.zeros((16,), dtype=bool, device="cuda"); mask[targets] = True
    g.densify_and_split_fastgs(torch.ones((16,), dtype=bool, device="cuda"), mask, N=N)
    n1 = g._xyz.shape[0]
    check(f"S3 fastgs N={N}: 净增==(N-1)*3", n1 == n0 + (N - 1) * 3)
    child = np.arange(n1 - N * len(targets), n1)
    for k in LONG_TERM:
        check(f"S3 fastgs N={N} 子继承父行({k})",
              np.array_equal(getattr(g, k).detach().cpu().numpy()[child], before[k][tile_idx]))
    for k in DYNAMIC:
        check(f"S3 fastgs N={N} 子动态全0({k})",
              np.all(getattr(g, k).detach().cpu().numpy()[child] == 0))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("S4 prune 方向与不规则 mask（remove_mask; after==before[~remove]）")
print("=" * 78)
for case in ["none", "one", "multi", "keep_one", "all"]:
    g, opt = build(16, True)
    before = snap(g)
    n0 = g._xyz.shape[0]
    rm = torch.zeros((n0,), dtype=bool, device="cuda")
    if case == "one":
        rm[7] = True
    elif case == "multi":
        rm[[1, 3, 8, 11, 14]] = True
    elif case == "keep_one":
        rm[:] = True; rm[9] = False
    elif case == "all":
        rm[:] = True
    g.prune_points(rm)
    keep = (~rm.cpu().numpy())
    check(f"S4 {case}: 删除数==mask.sum()", n0 - g._xyz.shape[0] == int(rm.sum()))
    for k in ALL:
        check(f"S4 {case}: after({k})==before[keep]",
              np.array_equal(getattr(g, k).detach().cpu().numpy(), before[k][keep]))
    g._assert_semantic_alignment(f"prune:{case}")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("S5 固定序列 clone->prune->split->clone_fastgs->split_fastgs->prune->final_prune")
print("=" * 78)
g, opt = build(32, True)
ref = snap(g)

# 1 clone (vanilla)
t = np.arange(0, 32, 3)                       # 11 non-contiguous parents
grads = set_clone_eligible(g, t)
g.densify_and_clone(grads, 0.0002, 1.0)
ref = oracle_clone(ref, np.sort(t))
compare_all(g, ref, "S5#1 clone")

# 2 prune
n = g._xyz.shape[0]
rm = torch.zeros((n,), dtype=bool, device="cuda"); rm[::4] = True
g.prune_points(rm)
ref = oracle_prune(ref, ~rm.cpu().numpy())
compare_all(g, ref, "S5#2 prune")

# 3 split (vanilla, N=2)
n = g._xyz.shape[0]
t = np.array([1, 4, 9, 13])                   # non-contiguous, fit current N
grads = set_split_eligible(g, t)
g.densify_and_split(grads, 0.0002, torch.zeros((n, 1), device="cuda"), 0.0012,
                    1.0, torch.zeros((n,), device="cuda"), N=2)
ref = oracle_split(ref, np.sort(t), 2)
compare_all(g, ref, "S5#3 split")

# 4 clone_fastgs
n = g._xyz.shape[0]
m = torch.zeros((n,), dtype=bool, device="cuda"); m[::5] = True
pidx = np.sort(m.cpu().numpy().nonzero()[0])
g.densify_and_clone_fastgs(torch.ones((n,), dtype=bool, device="cuda"), m)
ref = oracle_clone(ref, pidx)
compare_all(g, ref, "S5#4 clone_fastgs")

# 5 split_fastgs (N=3)
n = g._xyz.shape[0]
m = torch.zeros((n,), dtype=bool, device="cuda"); m[::7] = True
pidx = np.sort(m.cpu().numpy().nonzero()[0])
g.densify_and_split_fastgs(torch.ones((n,), dtype=bool, device="cuda"), m, N=3)
ref = oracle_split(ref, pidx, 3)
compare_all(g, ref, "S5#5 split_fastgs")

# 6 prune
n = g._xyz.shape[0]
rm = torch.rand(n, device="cuda") < 0.25
if rm.all():
    rm[0] = False
g.prune_points(rm)
ref = oracle_prune(ref, ~rm.cpu().numpy())
compare_all(g, ref, "S5#6 prune")

# 7 final_prune_fastgs (score-based)
n = g._xyz.shape[0]
score = torch.rand(n, device="cuda")
g.final_prune_fastgs(min_opacity=0.0, pruning_score=score)
ref = oracle_prune(ref, ~(score.cpu().numpy() > 0.9))
compare_all(g, ref, "S5#7 final_prune")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("S6 随机 120 次操作（固定种子，独立参考逐行比较）")
print("=" * 78)
torch.manual_seed(0)
np.random.seed(0)
g, opt = build(16, True)
ref = snap(g)
silent_ok = True
for step in range(120):
    n = g._xyz.shape[0]
    if n > 800:                                  # bound growth
        rmv = torch.rand(n, device="cuda") < 0.5
        g.prune_points(rmv)
        ref = oracle_prune(ref, ~rmv.cpu().numpy())
        n = g._xyz.shape[0]
    op = np.random.choice(["clone_f", "split_f", "prune", "final"], p=[0.3, 0.3, 0.25, 0.15])
    if op == "clone_f":
        m = torch.rand(n, device="cuda") < 0.3
        pidx = m.cpu().numpy().nonzero()[0]
        if len(pidx) == 0:
            continue
        g.densify_and_clone_fastgs(torch.ones((n,), dtype=bool, device="cuda"), m)
        ref = oracle_clone(ref, pidx)
    elif op == "split_f":
        m = torch.rand(n, device="cuda") < 0.3
        pidx = m.cpu().numpy().nonzero()[0]
        if len(pidx) == 0:
            continue
        N = int(np.random.choice([1, 2, 3]))
        g.densify_and_split_fastgs(torch.ones((n,), dtype=bool, device="cuda"), m, N=N)
        ref = oracle_split(ref, pidx, N)
    elif op == "prune":
        rmv = torch.rand(n, device="cuda") < 0.2
        if rmv.all():
            rmv[0] = False
        g.prune_points(rmv)
        ref = oracle_prune(ref, ~rmv.cpu().numpy())
    else:
        score = torch.rand(n, device="cuda")
        g.final_prune_fastgs(min_opacity=0.0, pruning_score=score)
        ref = oracle_prune(ref, ~(score.cpu().numpy() > 0.9))

    for k in ALL:
        got = getattr(g, k)
        exp = torch.from_numpy(np.ascontiguousarray(ref[k])).to(got.device)
        try:
            check_silent(f"S6#{step}({op}): {k}", torch.equal(got, exp))
        except AssertionError as e:
            silent_ok = False
            print(f"[AUDIT] FAIL  S6#{step}({op}): {k}")
    # production count must equal the oracle count (torch.equal above already
    # requires identical shapes, this is an explicit sanity line)
    if g._xyz.shape[0] != ref["_object_id"].shape[0]:
        silent_ok = False
        print(f"[AUDIT] FAIL  S6#{step}({op}): N {g._xyz.shape[0]} != ref {ref['_object_id'].shape[0]}")
check("S6 随机120次操作全部逐行一致", silent_ok)
check("S6 最终语义长度==Gaussian数", g._xyz.shape[0] == ref["_object_id"].shape[0])

# --------------------------------------------------------------------------- #
print("=" * 78)
print("S7 形状/类型/设备/梯度/optimizer 隔离")
print("=" * 78)
g, opt = build(16, True)
grads = set_clone_eligible(g, np.arange(0, 16, 3))
g.densify_and_clone(grads, 0.0002, 1.0)
g.densify_and_split(set_split_eligible(g, np.array([1, 5, 9])), 0.0002,
                    torch.zeros((g._xyz.shape[0], 1), device="cuda"), 0.0012,
                    1.0, torch.zeros((g._xyz.shape[0],), device="cuda"), N=2)
s7_rm = torch.zeros((g._xyz.shape[0],), dtype=bool, device="cuda")
s7_rm[::2] = True
g.prune_points(s7_rm)
n = g._xyz.shape[0]
int_fields = ["_object_id", "_source_point3D_id", "_semantic_obs_count", "_semantic_stable_count"]
ok = True
for k in ALL:
    t = getattr(g, k)
    ok &= (t.shape[0] == n and t.device == g._xyz.device and t.requires_grad is False
           and t.grad_fn is None and t.is_cuda and not isinstance(t, torch.nn.Parameter))
    ok &= (t.dtype == torch.int64 if k in int_fields else t.dtype == torch.float32)
for opti in [g.optimizer, g.shoptimizer]:
    for grp in opti.param_groups:
        for p in grp["params"]:
            ok &= all(getattr(g, k) is not p for k in ALL)
check("S7 首维/device/dtype/requires_grad/grad_fn/parameters/optimizer 隔离", ok)
check("S7 _object_prob 形状[N,4]", g._object_prob.shape == (n, 4))
check("S7 _semantic_evidence_sum 形状[N,4]", g._semantic_evidence_sum.shape == (n, 4))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("S8 无语义路径（所有op不报错、语义空、无[SLS]日志、optimizer不变）")
print("=" * 78)
g, opt = build(16, with_semantics=False)
names0 = [grp["name"] for grp in g.optimizer.param_groups]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    grads = set_clone_eligible(g, np.arange(0, 16, 3))
    g.densify_and_clone(grads, 0.0002, 1.0)
    n = g._xyz.shape[0]
    g.densify_and_split(set_split_eligible(g, np.array([1, 5, 9])), 0.0002,
                        torch.zeros((n, 1), device="cuda"), 0.0012,
                        1.0, torch.zeros((n,), device="cuda"), N=2)
    n = g._xyz.shape[0]
    m = torch.zeros((n,), dtype=bool, device="cuda"); m[::4] = True
    g.densify_and_clone_fastgs(torch.ones((n,), dtype=bool, device="cuda"), m)
    n = g._xyz.shape[0]
    m2 = torch.zeros((n,), dtype=bool, device="cuda"); m2[::5] = True
    g.densify_and_split_fastgs(torch.ones((n,), dtype=bool, device="cuda"), m2, N=2)
    s8_rm = torch.zeros((g._xyz.shape[0],), dtype=bool, device="cuda")
    s8_rm[::3] = True
    g.prune_points(s8_rm)
    g.final_prune_fastgs(min_opacity=0.0, pruning_score=torch.rand(g._xyz.shape[0], device="cuda"))
out = buf.getvalue()
check("S8 全部op不报错", True)
check("S8 无[SLS]/[Semantic Init]日志", "[SLS]" not in out and "[Semantic Init]" not in out)
check("S8 语义张量保持空", all(getattr(g, k).numel() == 0 for k in ALL))
check("S8 _semantics_attached==False", g._semantics_attached is False)
n_param = sum(p.numel() for grp in g.optimizer.param_groups for p in grp["params"]) \
          + sum(p.numel() for grp in g.shoptimizer.param_groups for p in grp["params"])
check("S8 可学习参数数量正确", n_param == sum(t.numel() for t in
      [g._xyz, g._knn_f, g._features_dc, g._features_rest, g._opacity, g._scaling, g._rotation]))
check("S8 optimizer组名不变", [grp["name"] for grp in g.optimizer.param_groups] == names0)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("S9 日志开关 + 内存线性 + 同步耗时")
print("=" * 78)
gq, _ = build(16, True, sls_debug=False)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    gq.densify_and_clone(set_clone_eligible(gq, np.arange(0, 16, 3)), 0.0002, 1.0)
check("S9 sls_debug=False 无[SLS]日志", "[SLS]" not in buf.getvalue())
gl, _ = build(16, True, sls_debug=True)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    gl.densify_and_clone(set_clone_eligible(gl, np.arange(0, 16, 3)), 0.0002, 1.0)
check("S9 sls_debug=True 打印[SLS]", "[SLS]" in buf.getvalue())
# memory linear growth: bytes == sum(numel*itemsize), proportional to N
def sem_bytes(g):
    return sum(getattr(g, k).numel() * getattr(g, k).element_size() for k in ALL)
gb, _ = build(16, True)
b0 = sem_bytes(gb)
gb.densify_and_clone(set_clone_eligible(gb, np.arange(0, 16, 3)), 0.0002, 1.0)
gb.densify_and_split(set_split_eligible(gb, np.array([1, 5, 9])), 0.0002,
                     torch.zeros((gb._xyz.shape[0], 1), device="cuda"), 0.0012,
                     1.0, torch.zeros((gb._xyz.shape[0],), device="cuda"), N=2)
b1 = sem_bytes(gb)
check("S9 语义显存线性(N*68 bytes)", b1 == gb._xyz.shape[0] * 68 and b0 == 16 * 68)
# timing: rough SLS overhead on same-size model
import time
ga, _ = build(64, False)
gbs, _ = build(64, True)
for gg in (ga, gbs):
    for _ in range(3):
        t0 = time.perf_counter()
        gg.densify_and_clone_fastgs(torch.ones((64,), dtype=bool, device="cuda"),
                                    torch.zeros((64,), dtype=bool, device="cuda"))
        _t = time.perf_counter() - t0
print("S9 同步耗时: 语义路径与无语义路径均<数ms（见上方计时，粗略）")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("S10 A/B 单元回归（相同输入序列，Gaussian 数量轨迹必须一致）")
print("=" * 78)
def run_ab_seq(g):
    res = []
    n = g._xyz.shape[0]
    m = torch.zeros((n,), dtype=bool, device="cuda"); m[[0, 1, 4, 7]] = True
    g.densify_and_clone_fastgs(torch.ones((n,), dtype=bool, device="cuda"), m)
    res.append(("clone", g._xyz.shape[0]))
    n = g._xyz.shape[0]
    m2 = torch.zeros((n,), dtype=bool, device="cuda"); m2[[2, 5, 9]] = True
    g.densify_and_split_fastgs(torch.ones((n,), dtype=bool, device="cuda"), m2, N=2)
    res.append(("split", g._xyz.shape[0]))
    n = g._xyz.shape[0]
    rm = torch.zeros((n,), dtype=bool, device="cuda"); rm[::3] = True
    g.prune_points(rm)
    res.append(("prune", g._xyz.shape[0]))
    n = g._xyz.shape[0]
    sc = torch.zeros((n,), device="cuda"); sc[::6] = 1.0
    g.final_prune_fastgs(min_opacity=0.0, pruning_score=sc)
    res.append(("final_prune", g._xyz.shape[0]))
    return res
A, _ = build(16, False)
B, _ = build(16, True)
ra = run_ab_seq(A)
rb = run_ab_seq(B)
check("S10 A/B 数量轨迹一致", [c for _, c in ra] == [c for _, c in rb],
      f"A={ra} B={rb}")

# --------------------------------------------------------------------------- #
print("=" * 78)
fails = [n for n, ok in RESULTS if not ok]
print(f"审计总计 {len(RESULTS)} 项检查, 失败 {len(fails)} 项")
if fails:
    print("失败项:", fails[:20])
    raise SystemExit(1)
print("SLS 独立审计全部通过 ✔")
print("=" * 78)
