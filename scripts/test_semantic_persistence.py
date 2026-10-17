#
# SSP (Semantic State Persistence) acceptance test.
#
# Covers: checkpoint round-trip, non-zero dynamic fingerprints, post-densification
# round-trip, PLY+semantic_state.npz round-trip, xyz-order / gaussian-count /
# corrupted-probability / missing-sidecar failure modes, no-semantic compatibility,
# optimizer isolation after restore, and a continuous-vs-save-resume A/B comparison.
#
# Run: python scripts/test_semantic_persistence.py  (needs CUDA, fast-pgsr env)
#

import argparse
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel, SEMANTIC_STATE_VERSION
from utils.graphics_utils import BasicPointCloud

ALL = ["_object_id", "_object_prob", "_semantic_confidence", "_source_point3D_id",
       "_semantic_evidence_sum", "_semantic_obs_count", "_semantic_stable_count",
       "_semantic_candidate_id", "_semantic_candidate_stable_count"]
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[SSP-T] {'PASS' if cond else 'FAIL'}  {name}{(' | ' + detail) if detail else ''}")
    assert cond, f"SSP-T FAIL: {name}"


def expect_raises(name, fn, needle):
    try:
        fn()
        check(name, False, "expected RuntimeError")
    except RuntimeError as e:
        check(name, needle in str(e), str(e)[:100])


def make_semantics(N):
    rng = np.random.RandomState(0)
    obj = np.tile([-1, 0, 1, 2, 3], N // 5 + 1)[:N].astype(np.int64)
    return {"point3D_ids": (1000 + np.arange(N)).astype(np.int64),
            "object_id": obj,
            "object_probs": rng.dirichlet(np.ones(4), N).astype(np.float32),
            "confidence": rng.rand(N).astype(np.float32),
            "stats": {"raw_num_points": N, "filtered_num_points": N, "npz_total": N,
                      "matched": N, "missing_id_count": 0, "unused_npz": 0,
                      "max_xyz_alignment_error": 1e-7}}


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
    # populate Adam state (replace_tensor_to_optimizer needs a non-None stored_state)
    for opti in [g.optimizer, g.shoptimizer]:
        for pg in opti.param_groups:
            pp = pg["params"][0]
            if pp.grad is None:
                pp.grad = torch.zeros_like(pp)
        opti.step()
        opti.zero_grad()
    if with_semantics:
        # NON-ZERO unique dynamic fingerprints
        base = (torch.arange(N, device="cuda").float() * 0.13 + 0.01).unsqueeze(1)
        g._semantic_evidence_sum = (base.expand(-1, 4)
                                    + torch.arange(4, device="cuda").float() * 0.07).contiguous()
        g._semantic_obs_count = torch.arange(1, N + 1, dtype=torch.int64, device="cuda")
        g._semantic_stable_count = torch.arange(N, dtype=torch.int64, device="cuda") * 3 + 5
        # NON-NEUTRAL candidate state so persistence is verified too
        cand_labels = np.tile([-1, 0, 1, 2, 3], N // 5 + 1)[:N]
        g._semantic_candidate_id = torch.tensor(cand_labels, dtype=torch.int64, device="cuda")
        g._semantic_candidate_stable_count = torch.arange(N, dtype=torch.int64, device="cuda") % 5
    return g, opt


def ckpt_round_trip(g):
    model_args = g.capture()
    buf = io.BytesIO()
    torch.save(model_args, buf)
    buf.seek(0)
    return torch.load(buf)


def equal7(g1, g2, tag):
    for k in ALL:
        check(f"{tag}: torch.equal({k})", torch.equal(getattr(g1, k), getattr(g2, k)))


# --------------------------------------------------------------------------- #
print("=" * 78)
print("P1 checkpoint round-trip（含非零动态指纹）")
print("=" * 78)
g, opt = build(16, True)
for k in ["_semantic_evidence_sum", "_semantic_obs_count", "_semantic_stable_count"]:
    check(f"P1 动态指纹非零({k})", getattr(g, k).abs().sum().item() > 0)
g2 = GaussianModel(3, sls_debug=False)
g2.restore(ckpt_round_trip(g), opt)
equal7(g, g2, "P1")
check("P1 restore 后 _semantics_attached==True", g2._semantics_attached is True)
check("P1 restore 后语义与Gaussian对齐", g2._object_id.shape[0] == g2._xyz.shape[0])

# --------------------------------------------------------------------------- #
print("=" * 78)
print("P2 clone/split/prune 后 checkpoint round-trip")
print("=" * 78)
g, opt = build(32, True)
n = g._xyz.shape[0]
m = torch.zeros((n,), dtype=bool, device="cuda"); m[::4] = True
g.densify_and_clone_fastgs(torch.ones((n,), dtype=bool, device="cuda"), m)
n = g._xyz.shape[0]
m = torch.zeros((n,), dtype=bool, device="cuda"); m[::5] = True
g.densify_and_split_fastgs(torch.ones((n,), dtype=bool, device="cuda"), m, N=2)
rm = torch.zeros((g._xyz.shape[0],), dtype=bool, device="cuda"); rm[::6] = True
g.prune_points(rm)
g2 = GaussianModel(3, sls_debug=False)
g2.restore(ckpt_round_trip(g), opt)
equal7(g, g2, "P2")
check("P2 恢复后 Gaussian 数量一致", g._xyz.shape[0] == g2._xyz.shape[0])

# --------------------------------------------------------------------------- #
print("=" * 78)
print("P3 PLY + semantic_state.npz round-trip")
print("=" * 78)
g, opt = build(16, True)
with tempfile.TemporaryDirectory() as td:
    ply_path = os.path.join(td, "point_cloud.ply")
    g.save_ply(ply_path)
    check("P3 sidecar 已生成", os.path.exists(os.path.join(td, "semantic_state.npz")))
    g2 = GaussianModel(3)
    g2.load_ply(ply_path, load_semantics=True)
    # _knn_f is not persisted in the PLY (original Fast-PGSR behaviour).
    for field in ["_xyz", "_features_dc", "_features_rest",
                  "_opacity", "_scaling", "_rotation"]:
        check(f"P3 几何 torch.equal({field})", torch.equal(getattr(g2, field), getattr(g, field)))
    equal7(g, g2, "P3")
    check("P3 _semantics_attached==True", g2._semantics_attached is True)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("P4 顺序/计数/概率/缺失 校验")
print("=" * 78)
g, opt = build(16, True)
with tempfile.TemporaryDirectory() as td:
    ply_path = os.path.join(td, "point_cloud.ply")
    g.save_ply(ply_path)
    sidecar = os.path.join(td, "semantic_state.npz")
    data = np.load(sidecar)

    # (a) 打乱 PLY 顶点顺序 -> xyz_order_checksum 必须报错
    from plyfile import PlyData, PlyElement
    def permute_ply():
        pd = PlyData.read(ply_path)
        v = pd['vertex']
        perm = np.random.RandomState(3).permutation(v.count)
        new = np.empty_like(v.data)
        for name in v.data.dtype.names:
            new[name] = v.data[name][perm]
        PlyData([PlyElement.describe(new, 'vertex')]).write(ply_path)
    permute_ply()
    expect_raises("P4a PLY 顺序打乱被 xyz checksum 检测",
                  lambda: GaussianModel(3).load_ply(ply_path, load_semantics=True),
                  "xyz_order_checksum")

    # (b) 打乱 npz 行序 -> semantic_order_checksum 必须报错（保持 gaussian_count/xyz checksum 合法）
    g, opt = build(16, True)
    g.save_ply(ply_path)
    data = np.load(sidecar)
    arr_keys = ["object_id", "object_prob", "semantic_confidence", "source_point3D_id",
                "semantic_evidence_sum", "semantic_obs_count", "semantic_stable_count",
                "semantic_candidate_id", "semantic_candidate_stable_count"]
    perm = np.random.RandomState(5).permutation(16)
    rewrite = {k: np.asarray(data[k])[perm] for k in arr_keys}
    rewrite.update(semantics_attached=True, semantic_state_version=SEMANTIC_STATE_VERSION,
                   gaussian_count=16, class_names=data["class_names"],
                   xyz_order_checksum=str(data["xyz_order_checksum"]),
                   semantic_order_checksum=str(data["semantic_order_checksum"]))
    np.savez(sidecar, **rewrite)
    expect_raises("P4b npz 行序打乱被 semantic checksum 检测",
                  lambda: GaussianModel(3).load_ply(ply_path, load_semantics=True),
                  "semantic_order_checksum")

    # (c) 删除一行语义 -> 报错
    g, opt = build(16, True)
    g.save_ply(ply_path)
    data = np.load(sidecar)
    rewrite = {k: np.asarray(data[k])[:-1] for k in arr_keys}
    rewrite.update(semantics_attached=True, semantic_state_version=SEMANTIC_STATE_VERSION,
                   gaussian_count=15, class_names=data["class_names"],
                   xyz_order_checksum=str(data["xyz_order_checksum"]))
    np.savez(sidecar, **rewrite)
    expect_raises("P4c 删除一行语义被 gaussian_count 检测",
                  lambda: GaussianModel(3).load_ply(ply_path, load_semantics=True),
                  "gaussian_count")

    # (d) 概率 NaN / 负值 / 行和错误
    for bad, needle in [(np.nan, "NaN"), (-0.1, "负值"), (0.1, "概率和")]:
        g, opt = build(16, True)
        g.save_ply(ply_path)
        data = np.load(sidecar)
        prob = np.asarray(data["object_prob"]).copy()
        if bad == 0.1:      # make row 0 sum to 1.1
            prob[0, 0] += 0.1
        else:
            prob[0, 0] = bad
        rewrite = {k: np.asarray(data[k]) for k in arr_keys}
        rewrite["object_prob"] = prob
        rewrite.update(semantics_attached=True, semantic_state_version=SEMANTIC_STATE_VERSION,
                       gaussian_count=16, class_names=data["class_names"],
                       xyz_order_checksum=str(data["xyz_order_checksum"]))
        np.savez(sidecar, **rewrite)
        expect_raises(f"P4d 概率{bad}被检测",
                      lambda: GaussianModel(3).load_ply(ply_path, load_semantics=True), needle)

    # (e) 缺失 sidecar
    empty = os.path.join(td, "empty_dir")
    os.makedirs(empty, exist_ok=True)
    g, opt = build(16, True)
    g.save_ply(os.path.join(empty, "point_cloud.ply"))
    os.remove(os.path.join(empty, "semantic_state.npz"))
    expect_raises("P4e 语义模式缺失 sidecar 报错",
                  lambda: GaussianModel(3).load_ply(os.path.join(empty, "point_cloud.ply"),
                                                    load_semantics=True),
                  "缺失")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("P5 无语义路径（原 checkpoint/PLY 行为不变）")
print("=" * 78)
g, opt = build(16, with_semantics=False)
g2 = GaussianModel(3, sls_debug=False)
g2.restore(ckpt_round_trip(g), opt)
check("P5 无语义 checkpoint 可恢复", g2._xyz.shape[0] == 16)
check("P5 无语义 checkpoint 语义为空", all(getattr(g2, k).numel() == 0 for k in ALL))
check("P5 _semantics_attached==False", g2._semantics_attached is False)
with tempfile.TemporaryDirectory() as td:
    ply_path = os.path.join(td, "point_cloud.ply")
    g.save_ply(ply_path)
    check("P5 无语义不写 sidecar", not os.path.exists(os.path.join(td, "semantic_state.npz")))
    g3 = GaussianModel(3)
    g3.load_ply(ply_path, load_semantics=False)
    check("P5 无语义 PLY 可加载", g3._xyz.shape[0] == 16)
    check("P5 无语义 PLY 语义为空", all(getattr(g3, k).numel() == 0 for k in ALL))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("P6 恢复后 optimizer 隔离")
print("=" * 78)
g, opt = build(16, True)
g2 = GaussianModel(3, sls_debug=False)
g2.restore(ckpt_round_trip(g), opt)
ok = True
for k in ALL:
    t = getattr(g2, k)
    ok &= (t.requires_grad is False and t.grad_fn is None
           and not isinstance(t, torch.nn.Parameter)
           and t.device == g2._xyz.device)
for opti in [g2.optimizer, g2.shoptimizer]:
    for grp in opti.param_groups:
        for p in grp["params"]:
            ok &= all(getattr(g2, k) is not p for k in ALL)
check("P6 语义 Tensor 非Parameter/无梯度/不在optimizer/device正确", ok)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("P7 连续训练 vs 中断续训（checkpoint 恢复）")
print("=" * 78)
def run_ops(g, n_ops):
    for i in range(n_ops):
        n = g._xyz.shape[0]
        if i % 3 == 0:
            m = torch.zeros((n,), dtype=bool, device="cuda"); m[::4] = True
            g.densify_and_clone_fastgs(torch.ones((n,), dtype=bool, device="cuda"), m)
        elif i % 3 == 1:
            m = torch.zeros((n,), dtype=bool, device="cuda"); m[::5] = True
            g.densify_and_split_fastgs(torch.ones((n,), dtype=bool, device="cuda"), m, N=2)
        else:
            rm = torch.zeros((n,), dtype=bool, device="cuda"); rm[::6] = True
            g.prune_points(rm)

A, opt = build(32, True)
run_ops(A, 6)
B, _ = build(32, True)
run_ops(B, 3)
with tempfile.TemporaryDirectory() as td:
    ckpt = os.path.join(td, "chkpnt3.pth")
    torch.save((B.capture(), 3), ckpt)
    model_args, _ = torch.load(ckpt)
    B2 = GaussianModel(3, sls_debug=False)
    B2.restore(model_args, opt)
run_ops(B2, 3)
check("P7 最终 Gaussian 数量一致", A._xyz.shape[0] == B2._xyz.shape[0],
      f"A={A._xyz.shape[0]} B2={B2._xyz.shape[0]}")
check("P7 类别数量一致", A._semantic_class_counts() == B2._semantic_class_counts())
equal7(A, B2, "P7")

# --------------------------------------------------------------------------- #
print("=" * 78)
fails = [n for n, ok in RESULTS if not ok]
print(f"SSP 测试总计 {len(RESULTS)} 项检查, 失败 {len(fails)} 项")
if fails:
    print("失败项:", fails[:20])
    raise SystemExit(1)
print("SSP 持久化测试全部通过 ✔")
print("=" * 78)
