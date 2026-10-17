#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
给 COLMAP 稀疏点云建立初始类别概率，并在运行结束后直接打开三维交互式可视化。

Mask 标签：
0 = background
1 = plant
2 = rubik_1
3 = rubik_2

默认目录：
<scene>/
├── images/
├── object_mask/
└── sparse/0/
    ├── images.bin
    └── points3D.bin

依赖：
pip install numpy opencv-python open3d


命令：
conda activate ObjectGS_GSAM2


python scripts/mask_generation/label_colmap_points_viewer.py \
    --scene data/mungbean13_0730 \
    --patch-size 5 \
    --target-dilation 5 \
    --candidate-min-hits 1 \
    --min-views 3 \
    --target-confidence 0.70 \
    --background-confidence 0.90 \
    --margin-threshold 0.20 \
    --point-size 4 \
    --visualize \
    --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


CLASS_NAMES = ["background", "plant", "rubik_1", "rubik_2"]
UNKNOWN = -1

# RGB：用于 PLY。OpenCV 绘图时转为 BGR。
CLASS_RGB = {
    0: (110, 110, 110),   # background: gray
    1: (30, 205, 60),     # plant: green
    2: (230, 50, 50),     # rubik_1: red
    3: (50, 105, 235),    # rubik_2: blue
    UNKNOWN: (255, 210, 20),  # unknown: yellow
}


@dataclass
class ImageRecord:
    image_id: int
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray


@dataclass
class PointRecord:
    point3D_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray
    point2D_idxs: np.ndarray


def read_exact(fid, n: int, fmt: str) -> Tuple:
    data = fid.read(n)
    if len(data) != n:
        raise EOFError(f"Unexpected EOF: need {n} bytes, got {len(data)}")
    return struct.unpack("<" + fmt, data)


def read_images_bin(path: Path) -> Dict[int, ImageRecord]:
    """读取 COLMAP images.bin。"""
    result: Dict[int, ImageRecord] = {}
    with path.open("rb") as f:
        num_images = read_exact(f, 8, "Q")[0]

        for _ in range(num_images):
            # image_id + qvec(4) + tvec(3) + camera_id
            p = read_exact(f, 64, "idddddddi")
            image_id = int(p[0])
            camera_id = int(p[8])

            name_buf = bytearray()
            while True:
                b = f.read(1)
                if not b:
                    raise EOFError("Unexpected EOF when reading image name")
                if b == b"\x00":
                    break
                name_buf.extend(b)
            name = name_buf.decode("utf-8")

            n2d = int(read_exact(f, 8, "Q")[0])
            dtype = np.dtype([
                ("x", "<f8"),
                ("y", "<f8"),
                ("point3D_id", "<i8"),
            ])
            raw = np.fromfile(f, dtype=dtype, count=n2d)
            if raw.size != n2d:
                raise EOFError(f"images.bin incomplete at image {image_id}")

            xys = np.column_stack([raw["x"], raw["y"]]).astype(np.float64, copy=False)
            point3D_ids = raw["point3D_id"].astype(np.int64, copy=False)

            result[image_id] = ImageRecord(
                image_id=image_id,
                camera_id=camera_id,
                name=name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return result


def read_points3D_bin(path: Path) -> Dict[int, PointRecord]:
    """读取 COLMAP points3D.bin。"""
    result: Dict[int, PointRecord] = {}
    with path.open("rb") as f:
        num_points = int(read_exact(f, 8, "Q")[0])

        for _ in range(num_points):
            p = read_exact(f, 43, "QdddBBBd")
            point_id = int(p[0])
            xyz = np.asarray(p[1:4], dtype=np.float64)
            rgb = np.asarray(p[4:7], dtype=np.uint8)
            error = float(p[7])

            track_len = int(read_exact(f, 8, "Q")[0])
            dtype = np.dtype([
                ("image_id", "<i4"),
                ("point2D_idx", "<i4"),
            ])
            track = np.fromfile(f, dtype=dtype, count=track_len)
            if track.size != track_len:
                raise EOFError(f"points3D.bin incomplete at point {point_id}")

            result[point_id] = PointRecord(
                point3D_id=point_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=track["image_id"].astype(np.int32, copy=False),
                point2D_idxs=track["point2D_idx"].astype(np.int32, copy=False),
            )
    return result


def image_path(images_dir: Path, image_name: str) -> Path:
    return images_dir / Path(image_name)


def mask_path(mask_dir: Path, image_name: str) -> Path:
    return mask_dir / Path(image_name).with_suffix(".png")


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if mask.ndim != 2:
        raise ValueError(f"Mask must be single-channel: {path}, shape={mask.shape}")
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    values = np.unique(mask)
    bad = values[(values < 0) | (values > 3)]
    if bad.size:
        raise ValueError(f"Mask contains invalid labels {bad.tolist()}: {path}")
    return mask


def load_image_size(path: Path) -> Tuple[int, int]:
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = im.shape
    return w, h


def ids_to_indices(
    ids: np.ndarray,
    sorted_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    POINT3D_ID 非连续，因此用 searchsorted 批量转连续数组索引。
    """
    ids_u64 = ids.astype(np.uint64, copy=False)
    idx = np.searchsorted(sorted_ids, ids_u64)
    found = idx < sorted_ids.size

    ok_pos = np.flatnonzero(found)
    checked = np.zeros(found.shape, dtype=bool)
    if ok_pos.size:
        checked[ok_pos] = sorted_ids[idx[ok_pos]] == ids_u64[ok_pos]
    return idx, checked


def map_xy_to_mask(
    xys: np.ndarray,
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int],
    allow_mismatch: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 images.bin 中的坐标映射到 Mask 坐标。
    """
    sw, sh = src_size
    dw, dh = dst_size

    if src_size != dst_size and not allow_mismatch:
        raise ValueError(
            f"Image/mask size mismatch: image={sw}x{sh}, mask={dw}x{dh}. "
            "Do not continue unless coordinate systems are confirmed identical. "
            "Then use --allow-size-mismatch."
        )

    sx = dw / float(sw)
    sy = dh / float(sh)
    x = np.rint(xys[:, 0] * sx).astype(np.int32)
    y = np.rint(xys[:, 1] * sy).astype(np.int32)
    inside = (x >= 0) & (x < dw) & (y >= 0) & (y < dh)
    return x, y, inside


def local_patch_probs(
    mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    patch_size: int,
    chunk_size: int,
) -> np.ndarray:
    """
    批量计算每个点周围 patch_size×patch_size 中 0/1/2/3 的占比。
    不使用逐点 Python 切片。
    """
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd number")

    h, w = mask.shape
    r = patch_size // 2
    dy, dx = np.meshgrid(
        np.arange(-r, r + 1, dtype=np.int32),
        np.arange(-r, r + 1, dtype=np.int32),
        indexing="ij",
    )
    dx = dx.reshape(1, -1)
    dy = dy.reshape(1, -1)

    out = np.zeros((x.size, 4), dtype=np.float32)

    for begin in range(0, x.size, chunk_size):
        end = min(begin + chunk_size, x.size)
        xx = x[begin:end, None] + dx
        yy = y[begin:end, None] + dy
        valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)

        sampled = np.full(xx.shape, 255, dtype=np.uint8)
        rr, cc = np.nonzero(valid)
        sampled[rr, cc] = mask[yy[rr, cc], xx[rr, cc]]

        denom = np.maximum(valid.sum(axis=1), 1).astype(np.float32)
        for c in range(4):
            out[begin:end, c] = (sampled == c).sum(axis=1) / denom

    return out


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    rgb = np.empty((labels.size, 3), dtype=np.uint8)
    for label, color in CLASS_RGB.items():
        rgb[labels == label] = np.asarray(color, dtype=np.uint8)
    return rgb


def write_binary_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    indices: Optional[np.ndarray] = None,
    object_id: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
    margin: Optional[np.ndarray] = None,
    view_count: Optional[np.ndarray] = None,
    target_hit_count: Optional[np.ndarray] = None,
    error: Optional[np.ndarray] = None,
    probs: Optional[np.ndarray] = None,
) -> None:
    """
    无需 plyfile，直接写 binary_little_endian PLY。
    """
    if indices is None:
        indices = np.arange(xyz.shape[0], dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)

    fields = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    props = [
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]

    optional = []
    if object_id is not None:
        fields.append(("object_id", "<i2"))
        props.append("property short object_id")
        optional.append(("object_id", object_id))
    if confidence is not None:
        fields.append(("semantic_confidence", "<f4"))
        props.append("property float semantic_confidence")
        optional.append(("semantic_confidence", confidence))
    if margin is not None:
        fields.append(("semantic_margin", "<f4"))
        props.append("property float semantic_margin")
        optional.append(("semantic_margin", margin))
    if view_count is not None:
        fields.append(("view_count", "<i4"))
        props.append("property int view_count")
        optional.append(("view_count", view_count))
    if target_hit_count is not None:
        fields.append(("target_hit_count", "<i4"))
        props.append("property int target_hit_count")
        optional.append(("target_hit_count", target_hit_count))
    if error is not None:
        fields.append(("reprojection_error", "<f4"))
        props.append("property float reprojection_error")
        optional.append(("reprojection_error", error))
    if probs is not None:
        for name in CLASS_NAMES:
            fields.append((f"prob_{name}", "<f4"))
            props.append(f"property float prob_{name}")

    data = np.empty(indices.size, dtype=np.dtype(fields, align=False))
    pts = xyz[indices].astype(np.float32, copy=False)
    cols = rgb[indices].astype(np.uint8, copy=False)
    data["x"], data["y"], data["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    data["red"], data["green"], data["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]

    for name, source in optional:
        data[name] = source[indices]

    if probs is not None:
        for c, name in enumerate(CLASS_NAMES):
            data[f"prob_{name}"] = probs[indices, c]

    header = "\n".join([
        "ply",
        "format binary_little_endian 1.0",
        "comment Generated by label_colmap_points.py",
        f"element vertex {indices.size}",
        *props,
        "end_header",
        "",
    ]).encode("ascii")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        data.tofile(f)




def save_semantic_npz(
    path: Path,
    point_ids: np.ndarray,
    xyz: np.ndarray,
    original_rgb: np.ndarray,
    probs: np.ndarray,
    argmax_label: np.ndarray,
    final_labels: np.ndarray,
    confidence: np.ndarray,
    margin: np.ndarray,
    view_count: np.ndarray,
    fine_view_count: np.ndarray,
    hit_count: np.ndarray,
    errors: np.ndarray,
) -> None:
    """把后续训练需要的全部数值压缩到一个 NPZ 文件中。"""
    np.savez_compressed(
        path,
        point3D_ids=point_ids,
        xyz=xyz.astype(np.float32),
        original_rgb=original_rgb.astype(np.uint8),
        object_probs=probs.astype(np.float32),
        argmax_label=argmax_label.astype(np.int16),
        object_id=final_labels.astype(np.int16),
        confidence=confidence.astype(np.float32),
        margin=margin.astype(np.float32),
        view_count=view_count.astype(np.int32),
        fine_view_count=fine_view_count.astype(np.int32),
        target_hit_count=hit_count.astype(np.int32),
        reprojection_error=errors.astype(np.float32),
        class_names=np.asarray(CLASS_NAMES),
        unknown_label=np.asarray([UNKNOWN], dtype=np.int16),
    )


def visualize_semantic_points(
    xyz: np.ndarray,
    labels: np.ndarray,
    point_size: float,
    window_width: int,
    window_height: int,
) -> bool:
    """
    使用 Open3D 直接显示语义稀疏点云。

    快捷键：
      A：显示全部
      T：只显示目标与 unknown
      B：切换背景
      P：切换植株
      1：切换 rubik_1
      2：切换 rubik_2
      U：切换 unknown
    """
    if sys.platform.startswith("linux"):
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not has_display:
            print(
                "[WARN] 当前 Linux 会话没有 DISPLAY/WAYLAND_DISPLAY，无法打开三维窗口。\n"
                "       结果已经保存，可在有桌面环境的机器上打开 points3D_semantic.ply，\n"
                "       或通过 X11 转发/VNC 后重新运行。"
            )
            return False

    try:
        import open3d as o3d
    except ImportError:
        print(
            "[WARN] 未安装 open3d，无法自动显示三维点云。\n"
            "       安装命令：pip install open3d\n"
            "       结果文件仍已正常保存。"
        )
        return False

    xyz64 = xyz.astype(np.float64, copy=False)
    semantic_rgb = labels_to_rgb(labels).astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    visible = {
        UNKNOWN: True,
        0: True,
        1: True,
        2: True,
        3: True,
    }

    def apply_visibility(vis, reset_view: bool = False) -> bool:
        active_labels = [label for label, enabled in visible.items() if enabled]
        if active_labels:
            keep = np.isin(labels, np.asarray(active_labels, dtype=labels.dtype))
        else:
            keep = np.zeros(labels.shape, dtype=bool)

        pcd.points = o3d.utility.Vector3dVector(xyz64[keep])
        pcd.colors = o3d.utility.Vector3dVector(semantic_rgb[keep])
        vis.update_geometry(pcd)
        if reset_view and np.any(keep):
            vis.reset_view_point(True)
        print(
            "[VIEW] visible: "
            f"background={visible[0]}, plant={visible[1]}, "
            f"rubik_1={visible[2]}, rubik_2={visible[3]}, unknown={visible[UNKNOWN]} "
            f"| points={int(keep.sum())}"
        )
        return False

    def toggle(label: int):
        def callback(vis):
            visible[label] = not visible[label]
            return apply_visibility(vis)
        return callback

    def show_all(vis):
        for label in visible:
            visible[label] = True
        return apply_visibility(vis, reset_view=True)

    def show_targets(vis):
        visible[0] = False
        visible[1] = True
        visible[2] = True
        visible[3] = True
        visible[UNKNOWN] = True
        return apply_visibility(vis, reset_view=True)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    created = vis.create_window(
        window_name="COLMAP semantic points | A all, T targets, B/P/1/2/U toggle",
        width=window_width,
        height=window_height,
        visible=True,
    )
    if not created:
        print("[WARN] Open3D 创建窗口失败；结果文件仍已正常保存。")
        return False

    # 先放入全部点，再注册交互。
    pcd.points = o3d.utility.Vector3dVector(xyz64)
    pcd.colors = o3d.utility.Vector3dVector(semantic_rgb)
    vis.add_geometry(pcd)

    vis.register_key_callback(ord("A"), show_all)
    vis.register_key_callback(ord("T"), show_targets)
    vis.register_key_callback(ord("B"), toggle(0))
    vis.register_key_callback(ord("P"), toggle(1))
    vis.register_key_callback(ord("1"), toggle(2))
    vis.register_key_callback(ord("2"), toggle(3))
    vis.register_key_callback(ord("U"), toggle(UNKNOWN))

    option = vis.get_render_option()
    option.background_color = np.asarray([0.03, 0.03, 0.03], dtype=np.float64)
    option.point_size = float(point_size)

    print("")
    print("[VIEW] 三维语义点云窗口已打开")
    print("       灰色=background，绿色=plant，红色=rubik_1，蓝色=rubik_2，黄色=unknown")
    print("       A=全部，T=目标，B/P/1/2/U=分别显示或隐藏，鼠标可旋转/缩放")

    vis.run()
    vis.destroy_window()
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Assign initial class probabilities to COLMAP sparse points and visualize them."
    )
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--images-dir", type=Path)
    p.add_argument("--mask-dir", type=Path)
    p.add_argument("--sparse-dir", type=Path)
    p.add_argument("--output-dir", type=Path)

    p.add_argument("--patch-size", type=int, default=5)
    p.add_argument("--target-dilation", type=int, default=5)
    p.add_argument("--candidate-min-hits", type=int, default=1)
    p.add_argument("--min-views", type=int, default=3)
    p.add_argument("--target-confidence", type=float, default=0.70)
    p.add_argument("--background-confidence", type=float, default=0.90)
    p.add_argument("--margin-threshold", type=float, default=0.20)
    p.add_argument("--background-probability", type=float, default=0.98)
    p.add_argument("--prob-eps", type=float, default=1e-4)
    p.add_argument(
        "--observation-weighting",
        choices=["equal", "purity"],
        default="equal",
    )
    p.add_argument("--max-reprojection-error", type=float, default=-1.0)
    p.add_argument("--chunk-size", type=int, default=100000)
    p.add_argument("--allow-size-mismatch", action="store_true")

    p.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open an interactive Open3D window after labeling (default: true).",
    )
    p.add_argument("--point-size", type=float, default=3.0)
    p.add_argument("--window-width", type=int, default=1280)
    p.add_argument("--window-height", type=int, default=900)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.patch_size <= 0 or args.patch_size % 2 == 0:
        raise ValueError("--patch-size must be a positive odd number")
    if args.min_views < 1:
        raise ValueError("--min-views must be >= 1")
    if args.target_dilation < 0:
        raise ValueError("--target-dilation must be >= 0")
    if args.candidate_min_hits < 1:
        raise ValueError("--candidate-min-hits must be >= 1")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    if args.point_size <= 0:
        raise ValueError("--point-size must be > 0")

    scene = args.scene.resolve()
    images_dir = (args.images_dir or scene / "images").resolve()
    masks_dir = (args.mask_dir or scene / "object_mask").resolve()
    sparse_dir = (args.sparse_dir or scene / "sparse" / "0").resolve()
    out_dir = (args.output_dir or scene / "semantic_initialization").resolve()

    images_bin = sparse_dir / "images.bin"
    points_bin = sparse_dir / "points3D.bin"
    for required in [images_dir, masks_dir, images_bin, points_bin]:
        if not required.exists():
            raise FileNotFoundError(required)

    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(f"{out_dir} exists; use --overwrite")
    out_dir.mkdir(parents=True)

    t_all = time.perf_counter()

    print("[1/5] Read COLMAP model")
    images = read_images_bin(images_bin)
    points = read_points3D_bin(points_bin)
    image_records = sorted(images.values(), key=lambda r: r.name)

    point_ids = np.asarray(sorted(points.keys()), dtype=np.uint64)
    point_records = [points[int(pid)] for pid in point_ids.tolist()]
    xyz = np.stack([p.xyz for p in point_records])
    original_rgb = np.stack([p.rgb for p in point_records])
    errors = np.asarray([p.error for p in point_records], dtype=np.float32)

    n = point_ids.size
    print(f"      images={len(image_records)}, points3D={n}")

    print("[2/5] Pass 1: target candidate screening")
    view_count = np.zeros(n, dtype=np.int32)
    hit_count = np.zeros(n, dtype=np.int32)
    processed_obs = 0
    missing_masks = []
    size_mismatch = []
    image_size_cache = {}

    kernel = None
    if args.target_dilation > 0:
        k = 2 * args.target_dilation + 1
        kernel = np.ones((k, k), dtype=np.uint8)

    for k_img, rec in enumerate(image_records, 1):
        ip = image_path(images_dir, rec.name)
        mp = mask_path(masks_dir, rec.name)
        if not mp.exists():
            missing_masks.append(rec.name)
            continue

        mask = load_mask(mp)
        src_size = load_image_size(ip)
        image_size_cache[rec.image_id] = src_size
        dst_size = (mask.shape[1], mask.shape[0])
        if src_size != dst_size:
            size_mismatch.append((rec.name, src_size, dst_size))

        valid = rec.point3D_ids != -1
        ids = rec.point3D_ids[valid]
        xys = rec.xys[valid]
        idx, found = ids_to_indices(ids, point_ids)
        idx, xys = idx[found], xys[found]
        if idx.size == 0:
            continue

        x, y, inside = map_xy_to_mask(
            xys, src_size, dst_size, args.allow_size_mismatch
        )
        idx, x, y = idx[inside], x[inside], y[inside]
        if idx.size == 0:
            continue

        target = (mask > 0).astype(np.uint8)
        if kernel is not None:
            target = cv2.dilate(target, kernel, iterations=1)
        hits = (target[y, x] > 0).astype(np.float32)

        view_count += np.bincount(idx, minlength=n).astype(np.int32)
        hit_count += np.bincount(idx, weights=hits, minlength=n).astype(np.int32)
        processed_obs += int(idx.size)

        if k_img % 20 == 0 or k_img == len(image_records):
            print(f"      {k_img}/{len(image_records)} images")

    if missing_masks:
        raise FileNotFoundError(
            f"Missing {len(missing_masks)} masks; first missing: {missing_masks[0]}"
        )
    if size_mismatch and not args.allow_size_mismatch:
        name, src_size, dst_size = size_mismatch[0]
        raise ValueError(
            f"Image/mask size mismatch in {name}: image={src_size}, mask={dst_size}"
        )

    candidate = hit_count >= args.candidate_min_hits
    background_candidate = (~candidate) & (view_count >= args.min_views)
    print(f"      processed observations={processed_obs}")
    print(f"      target candidates={int(candidate.sum())}")
    print(f"      background candidates={int(background_candidate.sum())}")

    print(f"[3/5] Pass 2: local {args.patch_size}x{args.patch_size} semantic evidence")
    votes = np.zeros((n, 4), dtype=np.float64)
    fine_views = np.zeros(n, dtype=np.int32)

    for k_img, rec in enumerate(image_records, 1):
        mp = mask_path(masks_dir, rec.name)

        valid = rec.point3D_ids != -1
        ids = rec.point3D_ids[valid]
        xys = rec.xys[valid]
        idx, found = ids_to_indices(ids, point_ids)
        idx, xys = idx[found], xys[found]
        if idx.size == 0:
            continue

        keep = candidate[idx]
        idx, xys = idx[keep], xys[keep]
        if idx.size == 0:
            continue

        mask = load_mask(mp)
        src_size = image_size_cache[rec.image_id]
        dst_size = (mask.shape[1], mask.shape[0])
        x, y, inside = map_xy_to_mask(
            xys, src_size, dst_size, args.allow_size_mismatch
        )
        idx, x, y = idx[inside], x[inside], y[inside]
        if idx.size == 0:
            continue

        local = local_patch_probs(mask, x, y, args.patch_size, args.chunk_size)
        weights = (
            local.max(axis=1)
            if args.observation_weighting == "purity"
            else np.ones(idx.size, dtype=np.float32)
        )

        for class_id in range(4):
            votes[:, class_id] += np.bincount(
                idx,
                weights=local[:, class_id] * weights,
                minlength=n,
            )
        fine_views += np.bincount(idx, minlength=n).astype(np.int32)

        if k_img % 20 == 0 or k_img == len(image_records):
            print(f"      {k_img}/{len(image_records)} images")

    print("[4/5] Compute probabilities and labels")
    probs = np.full((n, 4), 0.25, dtype=np.float32)
    has_vote = votes.sum(axis=1) > 0
    eps = float(args.prob_eps)

    if np.any(has_vote):
        smoothed = votes[has_vote] + eps
        smoothed /= smoothed.sum(axis=1, keepdims=True)
        probs[has_vote] = smoothed.astype(np.float32)

    bgp = float(args.background_probability)
    probs[background_candidate, 0] = bgp
    probs[background_candidate, 1:] = (1.0 - bgp) / 3.0
    probs = np.clip(probs, eps, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)

    argmax = probs.argmax(axis=1).astype(np.int16)
    ordered = np.sort(probs, axis=1)
    confidence = ordered[:, -1].astype(np.float32)
    margin = (ordered[:, -1] - ordered[:, -2]).astype(np.float32)

    quality_ok = np.ones(n, dtype=bool)
    if args.max_reprojection_error >= 0:
        quality_ok &= errors <= args.max_reprojection_error

    target_core = (
        quality_ok
        & (view_count >= args.min_views)
        & (argmax > 0)
        & (confidence >= args.target_confidence)
        & (margin >= args.margin_threshold)
    )
    background_core = (
        quality_ok
        & (view_count >= args.min_views)
        & (hit_count == 0)
        & (probs[:, 0] >= args.background_confidence)
    )

    final_labels = np.full(n, UNKNOWN, dtype=np.int16)
    final_labels[background_core] = 0
    final_labels[target_core] = argmax[target_core]
    semantic_rgb = labels_to_rgb(final_labels)

    counts = {
        "background": int((final_labels == 0).sum()),
        "plant": int((final_labels == 1).sum()),
        "rubik_1": int((final_labels == 2).sum()),
        "rubik_2": int((final_labels == 3).sum()),
        "unknown": int((final_labels == UNKNOWN).sum()),
    }
    print(f"      {counts}")

    print("[5/5] Save two result files")
    npz_path = out_dir / "point_semantics.npz"
    ply_path = out_dir / "points3D_semantic.ply"

    save_semantic_npz(
        npz_path,
        point_ids=point_ids,
        xyz=xyz,
        original_rgb=original_rgb,
        probs=probs,
        argmax_label=argmax,
        final_labels=final_labels,
        confidence=confidence,
        margin=margin,
        view_count=view_count,
        fine_view_count=fine_views,
        hit_count=hit_count,
        errors=errors,
    )
    write_binary_ply(
        ply_path,
        xyz=xyz,
        rgb=semantic_rgb,
        object_id=final_labels,
        confidence=confidence,
        margin=margin,
        view_count=view_count,
        target_hit_count=hit_count,
        error=errors,
        probs=probs,
    )

    elapsed = time.perf_counter() - t_all
    print("")
    print("[DONE]")
    print(f"       runtime={elapsed:.2f}s")
    print(f"       probabilities: {npz_path}")
    print(f"       colored point cloud: {ply_path}")
    print("       colors: gray=background, green=plant, red=rubik_1, blue=rubik_2, yellow=unknown")

    if args.visualize:
        visualize_semantic_points(
            xyz=xyz,
            labels=final_labels,
            point_size=args.point_size,
            window_width=args.window_width,
            window_height=args.window_height,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise