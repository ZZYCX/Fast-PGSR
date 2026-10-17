#!/usr/bin/env python
#
# Target Gaussian Exporter
# ------------------------
# Offline, read-only export of a trained point_cloud.ply + semantic_state.npz into
# target-focused Gaussian models (strict / conservative / per-class). No training
# logic is touched; Gaussians are never retrained or optimised here.
#
# Every exported model keeps ALL original PLY attributes (xyz, normals, SH DC/rest,
# opacity, scaling, rotation), the original row order, and a rebuilt semantic
# sidecar with freshly computed xyz_order_checksum / semantic_order_checksum.
#
# Run:
#   python scripts/export_target_gaussians.py \
#     --iteration_dir output/<scene>/point_cloud/iteration_30000 \
#     --output_dir output/<scene>/target_exports \
#     --mode both --unknown_target_threshold 0.60 \
#     --export_per_class --export_debug_cloud
#

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.semantic_io import SEMANTIC_STATE_VERSION, _xyz_checksum, _semantic_checksum

# order of the 9 semantic arrays (must match GaussianModel._save_semantic_sidecar).
# The last two (candidate state) only exist in semantic_state.npz version 2.
SEMANTIC_KEYS = ["object_id", "object_prob", "semantic_confidence", "source_point3D_id",
                 "semantic_evidence_sum", "semantic_obs_count", "semantic_stable_count",
                 "semantic_candidate_id", "semantic_candidate_stable_count"]
CORE_SEMANTIC_KEYS = SEMANTIC_KEYS[:7]
CANDIDATE_SEMANTIC_KEYS = SEMANTIC_KEYS[7:]
# legacy v1 sidecars never had candidate arrays
SUPPORTED_VERSIONS = (SEMANTIC_STATE_VERSION, 1)
_CAND_DEFAULT_ID = -1
_CAND_DEFAULT_STABLE = 0

DEBUG_COLORS = {0: (70, 130, 180), 1: (0, 200, 0), 2: (220, 20, 60), 3: (255, 215, 0)}


def _err(msg):
    raise RuntimeError(f"[Exporter] {msg}")


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #
def validate_input(ply_path, sidecar_path, unknown_target_threshold=0.60):
    """Load + validate the source PLY and semantic sidecar (all 12 checks)."""
    if not os.path.exists(ply_path):
        _err(f"point_cloud.ply 不存在: {ply_path}")
    if not os.path.exists(sidecar_path):
        _err(f"semantic_state.npz 不存在: {sidecar_path}")
    if not (0.0 <= float(unknown_target_threshold) <= 1.0):
        _err(f"unknown_target_threshold 必须在 [0,1]，当前 {unknown_target_threshold}")

    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]
    n = vertex.count
    xyz = np.stack((np.asarray(vertex["x"], dtype=np.float32),
                    np.asarray(vertex["y"], dtype=np.float32),
                    np.asarray(vertex["z"], dtype=np.float32)), axis=1)
    xyz = np.ascontiguousarray(xyz)

    data = np.load(sidecar_path)
    if int(data["semantic_state_version"]) not in SUPPORTED_VERSIONS:
        _err(f"语义状态版本 {int(data['semantic_state_version'])} 不在支持范围 {SUPPORTED_VERSIONS}")
    if not bool(data["semantics_attached"]):
        _err("semantics_attached=False，模型无语义状态")
    if int(data["gaussian_count"]) != n:
        _err(f"gaussian_count {int(data['gaussian_count'])} != PLY 点数 {n}")

    sem = {k: np.asarray(data[k]) for k in CORE_SEMANTIC_KEYS}
    # candidate arrays only exist in v2 sidecars; v1 -> neutral defaults
    for k in CANDIDATE_SEMANTIC_KEYS:
        if k in data:
            sem[k] = np.asarray(data[k])
        else:
            sem[k] = (np.full(n, _CAND_DEFAULT_ID, np.int64) if k == "semantic_candidate_id"
                      else np.zeros(n, np.int64))
    for k in SEMANTIC_KEYS:
        if sem[k].shape[0] != n:
            _err(f"语义数组 {k} 长度 {sem[k].shape[0]} != PLY 点数 {n}")
    oid = sem["object_id"]
    prob = sem["object_prob"]
    if not np.isin(oid, [-1, 0, 1, 2, 3]).all():
        _err("object_id 含非法值（仅允许 -1/0/1/2/3）")
    if prob.ndim != 2 or prob.shape[1] != 4:
        _err(f"object_prob 形状应为 [N,4]，实际 {prob.shape}")
    if not np.isfinite(prob).all() or (prob < 0).any():
        _err("object_prob 含 NaN/Inf 或负值")
    if not np.allclose(prob.sum(-1), 1.0, atol=1e-4):
        _err("object_prob 每行概率和不为 1")
    if sem["semantic_candidate_id"].ndim != 1 or sem["semantic_candidate_stable_count"].ndim != 1:
        _err("候选语义数组必须是一维")

    cand_id = sem["semantic_candidate_id"] if "semantic_candidate_id" in data else None
    cand_stable = sem["semantic_candidate_stable_count"] if "semantic_candidate_stable_count" in data else None
    if str(data["xyz_order_checksum"]) != _xyz_checksum(xyz):
        _err("xyz_order_checksum 校验失败：PLY 顺序与语义状态不一致")
    if str(data["semantic_order_checksum"]) != _semantic_checksum(
            sem["object_id"], sem["object_prob"], sem["semantic_confidence"],
            sem["source_point3D_id"], sem["semantic_evidence_sum"],
            sem["semantic_obs_count"], sem["semantic_stable_count"],
            cand_id, cand_stable):
        _err("semantic_order_checksum 校验失败：sidecar 内部行序被改动")

    return plydata, xyz, sem


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def compute_selection(object_id, object_prob, unknown_target_threshold, mode):
    """Return (strict_mask, conservative_mask, unknown_cand_mask, p_target)."""
    oid = np.asarray(object_id)
    prob = np.asarray(object_prob, np.float32)
    strict = (oid == 1) | (oid == 2) | (oid == 3)
    p_target = prob[:, 1] + prob[:, 2] + prob[:, 3]
    unknown_cand = (oid == -1) & (p_target >= float(unknown_target_threshold))
    if mode == "strict":
        conservative = strict.copy()
    else:
        conservative = strict | unknown_cand
    return strict, conservative, unknown_cand, p_target


def spatial_target_selection(xyz, object_id, radius):
    """Keep one dominant instance per target class, plus nearby detached foliage.

    Radius is in model coordinates (marker edge = 1 in the aligned pipeline).
    Nearby components are compared only to the dominant component, never grown
    recursively through other plants. Noise points cannot bridge two instances.
    """
    if not np.isfinite(radius) or radius <= 0:
        _err("spatial radius 必须为有限正数")
    import open3d as o3d
    from scipy.spatial import cKDTree
    keep = np.zeros(len(xyz), dtype=bool)
    report = {"radius": float(radius), "attachment_distance": float(2 * radius),
              "min_points": 5, "policy": "dominant_class_component_with_nearby_foliage", "classes": {}}
    for obj in (1, 2, 3):
        indices = np.flatnonzero(np.asarray(object_id) == obj)
        if not len(indices):
            continue
        points = xyz[indices]
        if not np.isfinite(points).all():
            _err("空间筛选输入坐标含 NaN/Inf")
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        labels = np.asarray(cloud.cluster_dbscan(eps=float(radius), min_points=5))
        counts = np.bincount(labels[labels >= 0])
        if not len(counts):
            _err("类别 {} 没有足够的空间支持，禁止自动导出".format(obj))
        order = np.argsort(-counts, kind="stable")
        main = int(order[0])
        if len(order) > 1 and counts[main] < 1.5 * counts[order[1]]:
            _err("类别 {} 无明显主实例，请复核掩码/半径，禁止猜测目标".format(obj))
        tree = cKDTree(points[labels == main])
        accepted = labels == main
        components = []
        for label in order:
            gap = float(tree.query(points[labels == label])[0].min())
            # Cubes stay a single component; plant leaves can be detached by occlusion.
            selected = int(label) == main or (obj == 1 and gap <= 2 * radius)
            if selected:
                accepted |= labels == label
            components.append({"count": int(counts[label]), "gap_to_main": gap, "kept": bool(selected)})
        keep[indices[accepted]] = True
        report["classes"][str(obj)] = {"input": int(len(indices)), "kept": int(accepted.sum()),
            "removed": int((~accepted).sum()), "noise": int((labels < 0).sum()), "components": components}
    return keep, report


# --------------------------------------------------------------------------- #
# per-export helpers
# --------------------------------------------------------------------------- #
def _filtered_xyz(vertex_data, mask):
    xyz = np.stack((vertex_data["x"][mask].astype(np.float32),
                    vertex_data["y"][mask].astype(np.float32),
                    vertex_data["z"][mask].astype(np.float32)), axis=1)
    return np.ascontiguousarray(xyz)


def rebuild_sidecar(out_dir, sem, mask, xyz):
    """Rebuild semantic_state.npz for the exported subset (all 9 arrays masked)."""
    filtered = {k: sem[k][mask] for k in SEMANTIC_KEYS}
    n = filtered["object_id"].shape[0]
    np.savez(os.path.join(out_dir, "semantic_state.npz"),
             object_id=filtered["object_id"],
             object_prob=filtered["object_prob"],
             semantic_confidence=filtered["semantic_confidence"],
             source_point3D_id=filtered["source_point3D_id"],
             semantic_evidence_sum=filtered["semantic_evidence_sum"],
             semantic_obs_count=filtered["semantic_obs_count"],
             semantic_stable_count=filtered["semantic_stable_count"],
             semantic_candidate_id=filtered["semantic_candidate_id"],
             semantic_candidate_stable_count=filtered["semantic_candidate_stable_count"],
             semantics_attached=True,
             semantic_state_version=SEMANTIC_STATE_VERSION,
             gaussian_count=int(n),
             class_names=np.array(["background", "plant", "rubik_1", "rubik_2"]),
             xyz_order_checksum=_xyz_checksum(xyz),
             semantic_order_checksum=_semantic_checksum(
                 filtered["object_id"], filtered["object_prob"], filtered["semantic_confidence"],
                 filtered["source_point3D_id"], filtered["semantic_evidence_sum"],
                 filtered["semantic_obs_count"], filtered["semantic_stable_count"],
                 filtered["semantic_candidate_id"], filtered["semantic_candidate_stable_count"]))
    return filtered, n


def write_selection_debug_ply(out_dir, vertex_data, mask, object_id):
    """Simple XYZ-RGB debug cloud (NOT a Gaussian model)."""
    oid = np.asarray(object_id)
    sel_oid = oid[mask]
    col = np.zeros((mask.sum(), 3), dtype=np.uint8)
    # unknown(-1, preserved) -> yellow; plant/rubik_1/rubik_2 -> green/red/blue
    col[sel_oid == -1] = (255, 255, 0)
    col[sel_oid == 1] = (0, 200, 0)
    col[sel_oid == 2] = (220, 20, 60)
    col[sel_oid == 3] = (255, 215, 0)
    xyz = _filtered_xyz(vertex_data, mask)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    el = np.empty(len(xyz), dtype=dtype)
    el["x"], el["y"], el["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    el["red"], el["green"], el["blue"] = col[:, 0], col[:, 1], col[:, 2]
    PlyData([PlyElement.describe(el, "vertex")]).write(os.path.join(out_dir, "selection_debug.ply"))


def write_manifest(out_dir, manifest):
    with open(os.path.join(out_dir, "export_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def export_one(out_dir, mode_label, mask, plydata, xyz_full, sem, p_target,
               source_ply, source_sidecar, unknown_target_threshold, overwrite,
               export_debug_cloud, spatial_cleanup=None):
    sel = np.nonzero(mask)[0].astype(np.int64)
    if sel.size == 0:
        _err(f"{mode_label}: 空选择结果，禁止生成无法渲染的空模型")
    if os.path.exists(out_dir):
        if not overwrite:
            _err(f"输出目录已存在且未设置 --overwrite: {out_dir}")
    else:
        os.makedirs(out_dir, exist_ok=True)

    vertex_data = plydata["vertex"].data
    out_xyz = _filtered_xyz(vertex_data, mask)

    # 1) filtered PLY (all attributes, order + dtype preserved)
    out_ply = os.path.join(out_dir, "point_cloud.ply")
    PlyData([PlyElement.describe(vertex_data[mask], "vertex")]).write(out_ply)
    # 2) rebuilt sidecar
    filtered, n = rebuild_sidecar(out_dir, sem, mask, out_xyz)
    # 3) selection indices
    np.save(os.path.join(out_dir, "selection_indices.npy"), sel)
    # 4) debug cloud
    if export_debug_cloud:
        write_selection_debug_ply(out_dir, vertex_data, mask, sem["object_id"])

    oid = filtered["object_id"]
    sel_p = p_target[mask]
    manifest = {
        "source_ply": source_ply,
        "source_semantic_state": source_sidecar,
        "export_mode": mode_label,
        "unknown_target_threshold": float(unknown_target_threshold),
        "original_gaussian_count": int(len(mask)),
        "selected_gaussian_count": int(n),
        "removed_gaussian_count": int(len(mask) - n),
        "selected_background_count": int((oid == 0).sum()),
        "selected_unknown_count": int((oid == -1).sum()),
        "selected_plant_count": int((oid == 1).sum()),
        "selected_rubik_1_count": int((oid == 2).sum()),
        "selected_rubik_2_count": int((oid == 3).sum()),
        "strict_target_count": int(((oid == 1) | (oid == 2) | (oid == 3)).sum()),
        "unknown_candidate_count": int((oid == -1).sum()),
        "p_target_min": float(sel_p.min()),
        "p_target_mean": float(sel_p.mean()),
        "p_target_max": float(sel_p.max()),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "semantic_state_version": SEMANTIC_STATE_VERSION,
        "xyz_order_checksum": str(np.load(os.path.join(out_dir, "semantic_state.npz"))["xyz_order_checksum"]),
        "semantic_order_checksum": str(np.load(os.path.join(out_dir, "semantic_state.npz"))["semantic_order_checksum"]),
    }
    if spatial_cleanup is not None:
        manifest["spatial_cleanup"] = spatial_cleanup
    write_manifest(out_dir, manifest)
    return sel.size, manifest


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def export_target_gaussians(iteration_dir, output_dir, mode="both",
                            unknown_target_threshold=0.60, export_per_class=False,
                            export_debug_cloud=False, overwrite=False, rubik_count=None,
                            spatial_radius=0.0):
    t0 = time.time()
    source_ply = os.path.join(iteration_dir, "point_cloud.ply")
    source_sidecar = os.path.join(iteration_dir, "semantic_state.npz")
    plydata, xyz, sem = validate_input(source_ply, source_sidecar, unknown_target_threshold)

    if rubik_count is not None:
        if rubik_count not in (1, 2):
            _err("rubik_count 必须为 1 或 2")
        if rubik_count == 1 and np.any(sem["object_id"] == 3):
            _err("单魔方模型中存在 rubik_2 标签，请检查掩膜和语义状态")
        for obj_id in range(1, rubik_count + 2):
            if not np.any(sem["object_id"] == obj_id):
                _err(f"目标类别 {obj_id} 为空，不能作为完整目标导出")

    strict_mask, cons_mask, unknown_cand_mask, p_target = compute_selection(
        sem["object_id"], sem["object_prob"], unknown_target_threshold, mode)
    spatial_cleanup = None
    if not np.isfinite(spatial_radius) or spatial_radius < 0:
        _err("spatial_radius 必须为有限非负数")
    spatial_mask = np.ones(len(xyz), dtype=bool)
    if spatial_radius > 0:
        if mode != "strict":
            _err("空间清理仅用于 strict；旧保守导出请设置 spatial_radius=0")
        spatial_mask, spatial_cleanup = spatial_target_selection(xyz, sem["object_id"], spatial_radius)
        strict_mask &= spatial_mask

    outputs = []
    if mode in ("strict", "both"):
        outputs.append(("strict", "strict", strict_mask))
    if mode in ("conservative", "both"):
        out_name = f"conservative_p{int(round(unknown_target_threshold * 100)):03d}"
        outputs.append((out_name, "conservative", cons_mask))

    results = {}
    for dir_name, mode_label, mask in outputs:
        out_dir = os.path.join(output_dir, dir_name)
        sel_n, manifest = export_one(out_dir, mode_label, mask, plydata, xyz, sem, p_target,
                                     source_ply, source_sidecar, unknown_target_threshold,
                                     overwrite, export_debug_cloud, spatial_cleanup)
        results[dir_name] = sel_n
        print(f"[Exporter] {mode_label}: selected {sel_n} / {len(mask)} -> {out_dir}")

    if export_per_class:
        per_class = [("plant_only", (sem["object_id"] == 1)),
                     ("rubik_1_only", (sem["object_id"] == 2)),
                     ("rubik_2_only", (sem["object_id"] == 3)),
                     ("unknown_target_candidates", unknown_cand_mask)]
        for dir_name, mask in per_class:
            if spatial_cleanup is not None:
                mask = mask & spatial_mask
            if int(mask.sum()) == 0:
                print(f"[Exporter] 跳过空类别导出: {dir_name}")
                continue
            out_dir = os.path.join(output_dir, dir_name)
            sel_n, _ = export_one(out_dir, dir_name, mask, plydata, xyz, sem, p_target,
                                  source_ply, source_sidecar, unknown_target_threshold,
                                  overwrite, export_debug_cloud, spatial_cleanup)
            results[dir_name] = sel_n
            print(f"[Exporter] {dir_name}: selected {sel_n} -> {out_dir}")

    print(f"[Exporter] 完成，耗时 {time.time() - t0:.2f}s")
    return results


def main():
    p = argparse.ArgumentParser(description="Export target Gaussian models from a trained PLY + semantic sidecar")
    p.add_argument("--iteration_dir", type=str, required=True,
                   help="输入目录，包含 point_cloud.ply 和 semantic_state.npz")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--mode", type=str, choices=["strict", "conservative", "both"], default="strict")
    p.add_argument("--unknown_target_threshold", type=float, default=0.60, help="[0,1]")
    p.add_argument("--rubik-count", type=int, choices=(1, 2), default=1)
    p.add_argument("--export_per_class", action="store_true")
    p.add_argument("--export_debug_cloud", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--spatial-radius", type=float, default=0.0,
                   help="单株空间清理半径；ArUco边长归一化坐标，0保留旧语义选择")
    args = p.parse_args()
    export_target_gaussians(args.iteration_dir, args.output_dir, args.mode,
                            args.unknown_target_threshold, args.export_per_class,
                            args.export_debug_cloud, args.overwrite, args.rubik_count, args.spatial_radius)


if __name__ == "__main__":
    main()
