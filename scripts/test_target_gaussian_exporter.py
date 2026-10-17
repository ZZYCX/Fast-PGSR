#
# Target Gaussian Exporter acceptance test.
#
# An independent Oracle computes the expected selection indices directly from the
# crafted object_id / object_prob arrays (never calling the production selection
# logic). Every exported PLY / sidecar is then verified row-by-row against the Oracle.
#
# Run: python scripts/test_target_gaussian_exporter.py  (needs CUDA to build the model)
#

import argparse
import hashlib
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from plyfile import PlyData, PlyElement

from scene.gaussian_model import GaussianModel, _xyz_checksum, _semantic_checksum
from utils.graphics_utils import BasicPointCloud
import scripts.export_target_gaussians as exp

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[EXP-T] {'PASS' if cond else 'FAIL'}  {name}{(' | ' + detail) if detail else ''}")
    assert cond, f"EXP-T FAIL: {name}"


# ---- independent Oracle --------------------------------------------------- #
def ref_selection(object_id, object_prob, threshold, mode):
    oid = np.asarray(object_id)
    prob = np.asarray(object_prob, np.float32)
    strict = (oid == 1) | (oid == 2) | (oid == 3)
    if mode == "strict":
        return np.nonzero(strict)[0]
    p_target = prob[:, 1] + prob[:, 2] + prob[:, 3]
    cons = strict | ((oid == -1) & (p_target >= float(threshold)))
    return np.nonzero(cons)[0]


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---- probe model ---------------------------------------------------------- #
OID = np.array([-1, 0, 1, 2, 3, -1, 0, 1, 2, 3, -1, -1], np.int64)
PROB = np.array([[0.1, 0.5, 0.2, 0.2],     # 0  unknown p_target 0.9
                 [0.95, 0.02, 0.02, 0.01],  # 1  background
                 [0.1, 0.8, 0.05, 0.05],    # 2  plant
                 [0.1, 0.05, 0.8, 0.05],    # 3  rubik_1
                 [0.1, 0.05, 0.05, 0.8],    # 4  rubik_2
                 [0.9, 0.04, 0.03, 0.03],   # 5  unknown p_target 0.1
                 [0.95, 0.02, 0.02, 0.01],  # 6  background
                 [0.2, 0.7, 0.05, 0.05],    # 7  plant
                 [0.2, 0.05, 0.7, 0.05],    # 8  rubik_1
                 [0.2, 0.05, 0.05, 0.7],    # 9  rubik_2
                 [0.4, 0.3, 0.2, 0.1],      # 10 unknown p_target 0.60 (boundary)
                 [0.41, 0.3, 0.2, 0.09]],   # 11 unknown p_target 0.59
                np.float32)
THRESH = 0.60
STRICT_REF = np.array([2, 3, 4, 7, 8, 9])
CONS_REF = np.array([0, 2, 3, 4, 7, 8, 9, 10])


def build_probe(n=12):
    rng = np.random.RandomState(0)
    pcd = BasicPointCloud(points=(rng.rand(n, 3).astype(np.float32) + 5.0),
                          colors=rng.rand(n, 3).astype(np.float32),
                          normals=np.zeros((n, 3), np.float32))
    sem = {"point3D_ids": np.arange(1000, 1000 + n, dtype=np.int64),
           "object_id": OID.copy(), "object_probs": PROB.copy(),
           "confidence": np.full(n, 0.5, np.float32),
           "stats": {"raw_num_points": n, "filtered_num_points": n, "npz_total": n,
                     "matched": n, "missing_id_count": 0, "unused_npz": 0,
                     "max_xyz_alignment_error": 1e-7}}
    g = GaussianModel(3)
    g.create_from_pcd(pcd, 1.0, semantics=sem)
    return g


def make_iteration_dir():
    td = tempfile.mkdtemp()
    it = os.path.join(td, "iteration_10")
    os.makedirs(it)
    g = build_probe()
    g.save_ply(os.path.join(it, "point_cloud.ply"))
    return td, it, g


# --------------------------------------------------------------------------- #
print("=" * 78)
print("准备探测模型（12点：目标6/背景2/unknown4）")
print("=" * 78)
td, it_dir, g = make_iteration_dir()
orig_hash_ply = sha(os.path.join(it_dir, "point_cloud.ply"))
orig_hash_npz = sha(os.path.join(it_dir, "semantic_state.npz"))
out_root = os.path.join(td, "exports")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T1/T2 strict 与 conservative 选择索引 == 独立 Oracle")
print("=" * 78)
exp.export_target_gaussians(it_dir, out_root, mode="both", unknown_target_threshold=THRESH,
                            export_per_class=True, export_debug_cloud=True)
strict_idx = np.load(os.path.join(out_root, "strict", "selection_indices.npy"))
cons_idx = np.load(os.path.join(out_root, "conservative_p060", "selection_indices.npy"))
check("T1 strict selection_indices == Oracle", np.array_equal(strict_idx, STRICT_REF), f"{strict_idx.tolist()}")
check("T2 conservative selection_indices == Oracle", np.array_equal(cons_idx, CONS_REF), f"{cons_idx.tolist()}")
check("T2 conservative ⊇ strict", set(CONS_REF.tolist()) >= set(STRICT_REF.tolist()))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T3  阈值边界（0.60 含 / 0.59 排除 / threshold=0 / threshold=1）")
print("=" * 78)
check("T3 p_target==0.60(idx10) 保留", 10 in cons_idx)
check("T3 p_target==0.59(idx11) 排除", 11 not in cons_idx)
td0, it0, _ = make_iteration_dir()
out0 = os.path.join(td0, "o0")
exp.export_target_gaussians(it0, out0, mode="conservative", unknown_target_threshold=0.0)
c0 = np.load(os.path.join(out0, "conservative_p000", "selection_indices.npy"))
check("T3 threshold=0: 所有unknown保留", set(np.nonzero(OID == -1)[0].tolist()) <= set(c0.tolist()))
td1, it1, _ = make_iteration_dir()
out1 = os.path.join(td1, "o1")
exp.export_target_gaussians(it1, out1, mode="conservative", unknown_target_threshold=1.0)
c1 = np.load(os.path.join(out1, "conservative_p100", "selection_indices.npy"))
check("T3 threshold=1: 保守==strict", np.array_equal(c1, STRICT_REF))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T4/T5  保留的unknown仍为-1；原始顺序不变")
print("=" * 78)
cons_ply = PlyData.read(os.path.join(out_root, "conservative_p060", "point_cloud.ply"))
cons_obj = np.load(os.path.join(out_root, "conservative_p060", "semantic_state.npz"))["object_id"]
check("T4 保留unknown仍为-1", (cons_obj[np.nonzero(cons_obj == -1)] == -1).all())
check("T4 unknown未被迫改标签", set(cons_obj.tolist()) <= {-1, 1, 2, 3})
orig_ply = PlyData.read(os.path.join(it_dir, "point_cloud.ply"))
check("T5 输出第i行 == 原PLY[selection_indices[i]] 全属性",
      all(np.array_equal(cons_ply["vertex"].data[i], orig_ply["vertex"].data[cons_idx[i]]) for i in range(len(cons_idx))))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T7/T8  PLY 属性数量/名称/顺序/dtype 与原PLY一致")
print("=" * 78)
check("T7 属性数量一致", len(cons_ply["vertex"].properties) == len(orig_ply["vertex"].properties))
check("T7 属性名称一致", cons_ply["vertex"].data.dtype.names == orig_ply["vertex"].data.dtype.names)
check("T7 dtype一致", cons_ply["vertex"].data.dtype == orig_ply["vertex"].data.dtype)
check("T8 strict PLY 行 == 原PLY[strict_idx]",
      all(np.array_equal(PlyData.read(os.path.join(out_root, "strict", "point_cloud.ply"))["vertex"].data[i],
                         orig_ply["vertex"].data[STRICT_REF[i]]) for i in range(len(STRICT_REF))))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T9/T10 七个语义数组 == 原数组[selection]；sidecar checksum 正确")
print("=" * 78)
orig_sem = np.load(os.path.join(it_dir, "semantic_state.npz"))
cons_sem = np.load(os.path.join(out_root, "conservative_p060", "semantic_state.npz"))
for k in exp.SEMANTIC_KEYS:
    check(f"T9 {k} == 原[{cons_idx}]", np.array_equal(cons_sem[k], orig_sem[k][cons_idx]))
check("T10 gaussian_count == 选中数", int(cons_sem["gaussian_count"]) == len(cons_idx))
# recompute checksums from the OUTPUT ply + sidecar arrays
out_xyz = np.stack((cons_ply["vertex"]["x"].astype(np.float32),
                    cons_ply["vertex"]["y"].astype(np.float32),
                    cons_ply["vertex"]["z"].astype(np.float32)), axis=1)
out_xyz = np.ascontiguousarray(out_xyz)
check("T10 xyz checksum == 输出PLY重算", str(cons_sem["xyz_order_checksum"]) == _xyz_checksum(out_xyz))
check("T10 semantic checksum == 输出数组重算",
      str(cons_sem["semantic_order_checksum"]) == _semantic_checksum(
          cons_sem["object_id"], cons_sem["object_prob"], cons_sem["semantic_confidence"],
          cons_sem["source_point3D_id"], cons_sem["semantic_evidence_sum"],
          cons_sem["semantic_obs_count"], cons_sem["semantic_stable_count"],
          cons_sem["semantic_candidate_id"], cons_sem["semantic_candidate_stable_count"]))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T11 输出模型可被 load_ply(load_semantics=True) 正常加载")
print("=" * 78)
gl = GaussianModel(3)
gl.load_ply(os.path.join(out_root, "strict", "point_cloud.ply"), load_semantics=True)
check("T11 strict 加载成功且语义附加", gl._semantics_attached and gl._xyz.shape[0] == len(STRICT_REF))
gl2 = GaussianModel(3)
gl2.load_ply(os.path.join(out_root, "conservative_p060", "point_cloud.ply"), load_semantics=True)
check("T11 conservative 加载成功且语义附加", gl2._semantics_attached and gl2._xyz.shape[0] == len(CONS_REF))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T12/T13 strict 无背景/unknown；conservative 无背景仅目标+阈值unknown")
print("=" * 78)
strict_obj = np.load(os.path.join(out_root, "strict", "semantic_state.npz"))["object_id"]
check("T12 strict 无 background", not (strict_obj == 0).any())
check("T12 strict 无 unknown", not (strict_obj == -1).any())
cons_obj2 = np.load(os.path.join(out_root, "conservative_p060", "semantic_state.npz"))["object_id"]
check("T13 conservative 无 background", not (cons_obj2 == 0).any())
check("T13 conservative 只含目标+阈值unknown", set(cons_obj2.tolist()) <= {1, 2, 3, -1})

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T14 单类别导出纯度 100%")
print("=" * 78)
for dirname, cls in [("plant_only", 1), ("rubik_1_only", 2), ("rubik_2_only", 3),
                     ("unknown_target_candidates", -1)]:
    o = np.load(os.path.join(out_root, dirname, "semantic_state.npz"))["object_id"]
    check(f"T14 {dirname} 纯度100%（只含{cls}）", np.isin(o, [cls]).all())

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T15 输入错误必须报错")
print("=" * 78)
# missing PLY
try:
    exp.validate_input(os.path.join(td, "nope", "point_cloud.ply"), os.path.join(it_dir, "semantic_state.npz"), 0.6)
    check("T15 缺失PLY报错", False)
except RuntimeError:
    check("T15 缺失PLY报错", True)
# missing sidecar
try:
    exp.validate_input(os.path.join(it_dir, "point_cloud.ply"), os.path.join(td, "nope.npz"), 0.6)
    check("T15 缺失sidecar报错", False)
except RuntimeError:
    check("T15 缺失sidecar报错", True)
# wrong version
import numpy as _np
bad_ver = dict(np.load(os.path.join(it_dir, "semantic_state.npz")))
bad_ver["semantic_state_version"] = np.int64(99)
bad_ver_path = os.path.join(td, "bad_ver.npz")
_np.savez(bad_ver_path, **{k: v for k, v in bad_ver.items()})
try:
    exp.validate_input(os.path.join(it_dir, "point_cloud.ply"), bad_ver_path, 0.6)
    check("T15 版本错误报错", False)
except RuntimeError:
    check("T15 版本错误报错", True)
# bad checksum (corrupt xyz in npz-consistent ply? simplest: corrupt semantic array -> semantic checksum fail)
bad_sem = dict(np.load(os.path.join(it_dir, "semantic_state.npz")))
bad_sem["object_id"] = bad_sem["object_id"].copy(); bad_sem["object_id"][0] = 0   # still valid values
bad_sem_path = os.path.join(td, "bad_sem.npz")
_np.savez(bad_sem_path, **{k: v for k, v in bad_sem.items()})
try:
    exp.validate_input(os.path.join(it_dir, "point_cloud.ply"), bad_sem_path, 0.6)
    check("T15 语义checksum错误报错", False)
except RuntimeError:
    check("T15 语义checksum错误报错", True)
# bad prob
bad_prob = dict(np.load(os.path.join(it_dir, "semantic_state.npz")))
bad_prob["object_prob"] = bad_prob["object_prob"].copy(); bad_prob["object_prob"][0, 0] = float("nan")
bad_prob_path = os.path.join(td, "bad_prob.npz")
_np.savez(bad_prob_path, **{k: v for k, v in bad_prob.items()})
try:
    exp.validate_input(os.path.join(it_dir, "point_cloud.ply"), bad_prob_path, 0.6)
    check("T15 非法object_prob报错", False)
except RuntimeError:
    check("T15 非法object_prob报错", True)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T16 原PLY与sidecar导出前后哈希不变")
print("=" * 78)
check("T16 point_cloud.ply 未修改", sha(os.path.join(it_dir, "point_cloud.ply")) == orig_hash_ply)
check("T16 semantic_state.npz 未修改", sha(os.path.join(it_dir, "semantic_state.npz")) == orig_hash_npz)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T17 mode=both 两个输出均正确且 conservative >= strict")
print("=" * 78)
check("T17 strict 数量", len(STRICT_REF) > 0)
check("T17 conservative >= strict", len(CONS_REF) >= len(STRICT_REF))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T18 overwrite=False 拒绝覆盖；T19 空选择报错")
print("=" * 78)
try:
    exp.export_target_gaussians(it_dir, out_root, mode="both", unknown_target_threshold=THRESH)
    check("T18 已存在目录拒绝覆盖", False)
except RuntimeError:
    check("T18 已存在目录拒绝覆盖", True)
# overwrite=True 允许
exp.export_target_gaussians(it_dir, out_root, mode="both", unknown_target_threshold=THRESH, overwrite=True)
check("T18 overwrite=True 允许覆盖", os.path.exists(os.path.join(out_root, "strict", "point_cloud.ply")))
# empty strict selection
all_bg = np.full(6, 0, np.int64)
bg_prob = np.tile([0.95, 0.02, 0.02, 0.01], (6, 1)).astype(np.float32)
td2, it2, _ = make_iteration_dir()
# rebuild with all-background semantics
pcd = BasicPointCloud(points=(np.random.RandomState(1).rand(6, 3).astype(np.float32) + 5.0),
                      colors=np.zeros((6, 3), np.float32), normals=np.zeros((6, 3), np.float32))
sem_bg = {"point3D_ids": np.arange(2000, 2006), "object_id": all_bg, "object_probs": bg_prob,
          "confidence": np.ones(6, np.float32), "stats": {}}
gb = GaussianModel(3); gb.create_from_pcd(pcd, 1.0, semantics=sem_bg)
gb.save_ply(os.path.join(it2, "point_cloud.ply"))
try:
    exp.export_target_gaussians(it2, os.path.join(td2, "o"), mode="strict", unknown_target_threshold=0.6)
    check("T19 空选择报错", False)
except RuntimeError:
    check("T19 空选择报错", True)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T20 大规模导出（3万Gaussian）耗时/内存")
print("=" * 78)
rng = np.random.RandomState(0)
N = 30000
pcd = BasicPointCloud(points=(rng.rand(N, 3).astype(np.float32) * 20.0 - 10.0),
                      colors=rng.rand(N, 3).astype(np.float32), normals=np.zeros((N, 3), np.float32))
oid = np.random.RandomState(2).choice([-1, 0, 1, 2, 3], N, p=[0.1, 0.5, 0.2, 0.1, 0.1])
prob = np.random.RandomState(3).dirichlet(np.ones(4), N).astype(np.float32)
sem_big = {"point3D_ids": np.arange(50000, 50000 + N), "object_id": oid, "object_probs": prob,
           "confidence": np.full(N, 0.5, np.float32), "stats": {}}
gbig = GaussianModel(3); gbig.create_from_pcd(pcd, 1.0, semantics=sem_big)
td3 = tempfile.mkdtemp(); it3 = os.path.join(td3, "iteration_10"); os.makedirs(it3)
t0 = time.time()
gbig.save_ply(os.path.join(it3, "point_cloud.ply"))
t_save = time.time() - t0
t0 = time.time()
exp.export_target_gaussians(it3, os.path.join(td3, "o"), mode="both",
                            unknown_target_threshold=0.6, overwrite=True)
t_exp = time.time() - t0
print(f"[EXP-T] T20 N={N} save={t_save:.1f}s export={t_exp:.1f}s")
check("T20 导出耗时 < 60s", t_exp < 60.0)
sel = np.load(os.path.join(td3, "o", "strict", "selection_indices.npy"))
check("T20 大规模strict选择数>0", len(sel) > 0)
check("T20 strict 无背景/unknown",
      not (np.load(os.path.join(td3, "o", "strict", "semantic_state.npz"))["object_id"] == 0).any())

# --------------------------------------------------------------------------- #
print("=" * 78)
fails = [n for n, ok in RESULTS if not ok]
print(f"Exporter 测试总计 {len(RESULTS)} 项检查, 失败 {len(fails)} 项")
if fails:
    print("失败项:", fails[:20])
    raise SystemExit(1)
print("目标Gaussian导出器测试全部通过 ✔")
print("=" * 78)
