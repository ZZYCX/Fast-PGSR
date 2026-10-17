#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在 plant_images 上逐张图片独立识别叶片：Grounding DINO 检测叶片框，
SAM2 分割成 mask，输出实例级 label 图（每片叶子一个编号）。

不做视频传播 / 关键帧，每张图互不依赖。

输入：
    <scene>/plant_images       由 get_onlyPlant.py 抠出的植株原图

输出：
    <scene>/leave_mask/        每帧单通道二值图（255=叶片，0=背景），PNG；
                               加 --instance 改为实例编号图（0=背景，1..N=每片叶子）
    <scene>/leave_mask_check/  每帧彩色框选预览（同 init_leaves 风格）+ 报告

依赖：
    run_grounded_sam2_scene.py（只读导入其函数，不修改原脚本）

用法：
    python scripts/mask_generation/run_leaf_mask_scene.py \
        --scene mungbean12_0730 \
        --prompt "leaf" \
        --clean-background \
        --overwrite

说明：
    1. 每帧独立检测：DINO 检测所有叶片框 → SAM2 逐个分割 → 每片叶子赋编号 1..N；
    2. 无传播，因此叶片编号只在单帧内有意义，帧间不保证对应关系；
    3. --prompt 与 --leaf-key 需要匹配：提示词含哪个词，就按哪个词过滤检测框；
    4. DINO 有时会把整株植物也框成一个“叶片”，其 mask 面积远大于单叶，
       脚本按 mask 面积与中位数的倍数（--max-leaf-ratio）剔除这类整株大框；
    5. DINO 还可能在多尺度上把“互相重叠的几片叶子”整体框成一个聚合框，
       脚本按“完整包含 >=2 个其他叶片框”的规则剔除这类聚合框；
    6. 默认整图缩到 1024 后分割；加 --full-res 时改为对每片叶子在其
       全分辨率裁剪区域上分割（tiling），细结构（叶柄/枝干）精度更高，但更慢；
    7. leave_mask 默认输出二值图（255=叶片，0=背景），适合直接作为语义 mask；
       实例编号仍在内部计算（预览/报告使用），加 --instance 可改回编号图。
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "run_grounded_sam2_scene.py"
SCRIPT_VERSION = "2026-08-06-fastpgsr-leaf-mask-perimage-v2"
DEFAULT_PROMPT = "leaf"
LEAF_PALETTE = [
    (80, 210, 80),
    (60, 80, 235),
    (235, 150, 50),
    (200, 60, 180),
    (70, 200, 230),
    (240, 200, 70),
    (150, 90, 230),
    (60, 180, 130),
]


def neutralize_tf_availability() -> None:
    """禁用 transformers 对 tensorflow 的可用性检查。

    本机 ~/.local 下装了一个损坏的 tensorflow（缺少 google.protobuf），
    而 transformers 的 importlib.find_spec 会误判其“可用”，一旦 Grounding DINO
    检测到目标、解码标签短语时就会 `import tensorflow` 崩溃。这里把相关检查
    全部置为 False，避免触发崩溃，且不影响任何 torch 推理路径。
    """
    try:
        import transformers.utils.import_utils as import_utils
        import_utils.is_tf_available = lambda: False

        import transformers.utils.generic as generic
        generic.is_tf_available = lambda: False
        generic.is_tf_tensor = lambda x: False
        generic.is_tf_symbolic_tensor = lambda x: False
    except Exception:
        pass


def load_base():
    if not BASE_SCRIPT.is_file():
        raise FileNotFoundError(f"缺少基础传播脚本：{BASE_SCRIPT}")
    spec = importlib.util.spec_from_file_location("grounded_scene_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载：{BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = [
        "GSAM2_ROOT", "SAM2ImagePredictor", "build_sam2",
        "filter_label", "list_images", "load_grounding_model_offline", "load_image",
        "nms_preserve_order", "normalize_prompt", "predict_masks",
        "remove_generated_path", "run_grounding", "verify_bert",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "run_grounded_sam2_scene.py 缺少接口："
            + ", ".join(missing)
            + "，请保证两个脚本为配套版本。"
        )
    return module


def resolve_scene_dir(value: str) -> Path:
    """--scene 既支持直接路径（data/xxx），也支持场景名（xxx）。"""
    direct = Path(value)
    if direct.is_dir():
        return direct
    resolved = PROJECT_ROOT / "data" / value
    if not resolved.is_dir():
        raise FileNotFoundError(f"场景目录不存在：{resolved} 或 {direct}")
    return resolved


def filter_leaf(detections: list[dict[str, Any]], leaf_key: str) -> list[dict[str, Any]]:
    """按 leaf_key 过滤叶片检测框；默认兼容复数 'leaves'。"""
    keys = {leaf_key}
    if leaf_key == "leaf":
        keys.add("leaves")
    return [d for d in detections if d["label_key"] in keys]


def containing_box_keep_indices(leaves: list[dict[str, Any]]) -> list[int]:
    """剔除“聚合框”：完整包含 >=2 个其他叶片框的框。

    DINO 会在同一组叶片上同时给出单叶小框和一个更大范围的整体框
    （例如把互相重叠的几片叶子整体框住）。聚合框会完整包含多个单叶框，
    这里把这类框剔除；真正的单叶之间只是部分重叠，不会相互完整包含。
    """
    boxes = np.asarray([det["box"] for det in leaves], dtype=float)  # xyxy
    keep: list[int] = []
    for i, box_i in enumerate(boxes):
        contained = 0
        for j, box_j in enumerate(boxes):
            if i == j:
                continue
            # 1% 容差，避免边界像素导致误判。
            tol_x = 0.01 * (box_i[2] - box_i[0])
            tol_y = 0.01 * (box_i[3] - box_i[1])
            if (
                box_i[0] <= box_j[0] + tol_x
                and box_i[1] <= box_j[1] + tol_y
                and box_i[2] >= box_j[2] - tol_x
                and box_i[3] >= box_j[3] - tol_y
            ):
                contained += 1
        if contained < 2:
            keep.append(i)
    return keep


def serializable(det: dict[str, Any]) -> dict[str, Any]:
    return {
        "order": int(det["order"]),
        "label": str(det["label"]),
        "label_key": str(det["label_key"]),
        "score": float(det["score"]),
        "box": [float(x) for x in np.asarray(det["box"]).tolist()],
    }


def preview_leaf_instances(
    image_bgr: np.ndarray,
    leaves: list[dict[str, Any]],
    masks: np.ndarray,
) -> np.ndarray:
    """单帧预览：每片叶子不同颜色叠加 + 检测框 + 编号/置信度（同 init_leaves 风格）。"""
    result = image_bgr.copy()
    overlay = image_bgr.copy()
    for i, (det, mask) in enumerate(zip(leaves, masks)):
        color = LEAF_PALETTE[i % len(LEAF_PALETTE)]
        overlay[mask] = color
        x1, y1, x2, y2 = np.round(np.asarray(det["box"])).astype(int).tolist()
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            result,
            f"{i + 1}:{det['score']:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA,
        )
    return cv2.addWeighted(result, 0.55, overlay, 0.45, 0)


def predict_mask_on_crop(
    sam_predictor,
    crop_rgb: np.ndarray,
    box_crop: np.ndarray,
) -> tuple[np.ndarray, float]:
    """在单个裁剪区域上跑 SAM2，返回 (mask, score)。

    crop_rgb 为任意尺寸的全分辨率 RGB（SAM2 内部会统一缩放到 image_size），
    box_crop 为裁剪区域内的 xyxy 检测框坐标。
    """
    sam_predictor.set_image(np.array(crop_rgb, copy=True))
    cast = torch.autocast("cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), cast:
        masks, scores, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_crop.reshape(1, 4),
            multimask_output=False,
        )
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    return masks[0].astype(bool), float(np.asarray(scores).reshape(-1)[0])


def segment_leaves_fullres(
    sam_predictor,
    image_rgb: np.ndarray,
    leaves: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[float]]:
    """对每片叶子在其全分辨率裁剪区域上跑 SAM2，再把 mask 拼回原图坐标。

    tiling 思路：叶子原本只占整图的一小块，整图缩到 1024 时叶子被压得很小，
    细结构（叶柄/枝干）精度差；这里把每片叶子所在区域按原分辨率裁出来，
    让 SAM2 在更大的有效分辨率下分割，最后拼回原图。
    """
    h, w = image_rgb.shape[:2]
    masks: list[np.ndarray] = []
    scores: list[float] = []
    for det in leaves:
        x1, y1, x2, y2 = np.round(np.asarray(det["box"])).astype(int).tolist()
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        margin_x = int(round(args.crop_margin * bw))
        margin_y = int(round(args.crop_margin * bh))
        cx1, cy1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
        cx2, cy2 = min(w, x2 + margin_x), min(h, y2 + margin_y)
        # 限制裁剪边长，避免过大内存；若 box 本身已超限则保持原样交给 SAM2。
        cw, ch = cx2 - cx1, cy2 - cy1
        if (
            max(cw, ch) > args.max_crop_size
            and (x2 - x1) <= args.max_crop_size
            and (y2 - y1) <= args.max_crop_size
        ):
            scale = args.max_crop_size / max(cw, ch)
            nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
            center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            cx1 = int(round(center_x - nw / 2.0))
            cy1 = int(round(center_y - nh / 2.0))
            cx2, cy2 = cx1 + nw, cy1 + nh
            if cx1 < 0:
                cx2 -= cx1
                cx1 = 0
            if cy1 < 0:
                cy2 -= cy1
                cy1 = 0
            if cx2 > w:
                cx1 -= cx2 - w
                cx2 = w
            if cy2 > h:
                cy1 -= cy2 - h
                cy2 = h

        crop = image_rgb[cy1:cy2, cx1:cx2]
        box_crop = np.array([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], dtype=np.float32)
        mask_crop, score = predict_mask_on_crop(sam_predictor, crop, box_crop)

        mask_full = np.zeros((h, w), dtype=bool)
        mask_full[cy1:cy2, cx1:cx2] = mask_crop
        masks.append(mask_full)
        scores.append(score)
    return np.stack(masks), scores


def process_image(
    base,
    dino,
    sam_predictor,
    image_path: Path,
    prompt: str,
    leaf_key: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, list[dict[str, Any]], np.ndarray | None, np.ndarray]:
    """在单张图上检测并分割叶片，返回 (label图, 叶片框列表, 实例mask, 原图BGR)。

    label 图：uint8 单通道，0=背景，1..N=每片叶子编号；没有叶片时返回 label=None。
    """
    image_rgb, image_tensor = base.load_image(str(image_path))
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError("OpenCV 无法读取图片")
    h, w = image_rgb.shape[:2]

    detections = base.run_grounding(
        dino,
        image_tensor,
        w,
        h,
        prompt,
        args.box_threshold,
        args.text_threshold,
        "cuda",
    )
    leaves = base.nms_preserve_order(
        filter_leaf(detections, leaf_key),
        args.nms_threshold,
    )
    if not leaves:
        return None, [], None, image_bgr
    if len(leaves) > args.max_leaves:
        leaves = leaves[: args.max_leaves]

    if args.full_res:
        # 方案 B：每片叶子在全分辨率裁剪区域上分割（tiling）。
        masks, scores = segment_leaves_fullres(sam_predictor, image_rgb, leaves, args)
    else:
        boxes = np.stack([item["box"] for item in leaves]).astype(np.float32)
        masks, scores = base.predict_masks(sam_predictor, image_rgb, boxes, "cuda")

    area = h * w
    keep = [
        i
        for i, (mask, score) in enumerate(zip(masks, scores))
        if args.min_leaf_area * area <= mask.sum() <= args.max_leaf_area * area
    ]
    if not keep:
        return None, [], None, image_bgr
    leaves = [leaves[i] for i in keep]
    masks = masks[keep]

    # 剔除整株大框：DINO 可能把整株植物也框成一个“叶片”，其 mask 面积
    # 远大于单叶，按“面积 > 中位数 × max-leaf-ratio”过滤这类离群大块。
    if len(masks) >= 2:
        leaf_areas = np.asarray([mask.sum() for mask in masks], dtype=float)
        median_area = float(np.median(leaf_areas))
        if median_area > 0:
            keep = [
                i
                for i, leaf_area in enumerate(leaf_areas)
                if leaf_area <= args.max_leaf_ratio * median_area
            ]
            leaves = [leaves[i] for i in keep]
            masks = masks[keep]
    if not leaves:
        return None, [], None, image_bgr

    # 剔除聚合框：完整包含 >=2 个其他叶片框的大框（重叠叶子的整体框）。
    if len(leaves) >= 3:
        keep = containing_box_keep_indices(leaves)
        leaves = [leaves[i] for i in keep]
        masks = masks[keep]
    if not leaves:
        return None, [], None, image_bgr

    # 每片叶子按出现顺序编号 1..N。
    label = np.zeros((h, w), dtype=np.uint8)
    for obj_id, mask in enumerate(masks, start=1):
        label[mask] = obj_id

    # 可选：剔除黑底抠图中被并入的纯黑背景像素。
    if args.clean_background:
        dark = image_bgr.max(axis=2) < args.background_threshold
        label[dark] = 0

    return label, leaves, masks, image_bgr


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="逐图识别 plant_images 中的叶片并输出实例 label mask")
    p.add_argument("--scene", required=True,
                   help="场景名（如 mungbean12_0730）或直接路径（如 data/mungbean12_0730）")
    p.add_argument("--input-dir", default="plant_images",
                   help="输入图像目录名，默认 plant_images")
    p.add_argument("--output-dir", default="leave_mask",
                   help="叶片 label mask 输出目录名，默认 leave_mask")
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="Grounding DINO 文本提示词，默认 'leaf'")
    p.add_argument("--leaf-key", default="leaf",
                   help="过滤检测框用的标签关键字，需与提示词匹配，默认 'leaf'")
    p.add_argument("--box-threshold", type=float, default=0.10,
                   help="Grounding DINO 检测框置信度阈值")
    p.add_argument("--text-threshold", type=float, default=0.10,
                   help="检测框与提示词文本的匹配阈值")
    p.add_argument("--nms-threshold", type=float, default=0.72,
                   help="重复检测框去重 IoU 阈值")
    p.add_argument("--max-leaves", type=int, default=24,
                   help="单帧最多保留的叶片数（按置信度取前 N）")
    p.add_argument("--min-leaf-area", type=float, default=0.0001,
                   help="叶片 mask 最小面积占比，过滤过小碎片")
    p.add_argument("--max-leaf-area", type=float, default=0.60,
                   help="叶片 mask 最大面积占比，过滤异常大块")
    p.add_argument("--max-leaf-ratio", type=float, default=4.0,
                   help="剔除整株大框：某叶片 mask 面积超过全部叶片中位数的该倍数时，"
                        "视为整株误检并剔除，默认 4.0")
    p.add_argument("--full-res", action="store_true",
                   help="对每片叶子在其全分辨率裁剪区域上跑 SAM2（tiling），"
                        "避免整图缩到 1024 的细结构精度损失，更慢但更精确")
    p.add_argument("--crop-margin", type=float, default=0.25,
                   help="--full-res 时围绕检测框外扩的比例（相对框尺寸），"
                        "给 SAM2 更多上下文")
    p.add_argument("--max-crop-size", type=int, default=1024,
                   help="--full-res 时裁剪区域的最大边长，超限会收窄外扩")
    p.add_argument("--clean-background", action="store_true",
                   help="剔除输入图中接近纯黑的背景像素（黑底抠图时建议开启）")
    p.add_argument("--background-threshold", type=int, default=10,
                   help="--clean-background 的判定阈值：RGB 最大通道小于该值视为背景")
    p.add_argument("--overwrite", action="store_true",
                   help="输出目录已存在时删除重建")
    p.add_argument("--instance", action="store_true",
                   help="输出实例编号图（0=背景，1..N=每片叶子）而不是默认的"
                        "二值图（255=叶片，0=背景）")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.max_leaves < 1:
        raise ValueError("max-leaves 必须 >= 1")

    base = load_base()
    neutralize_tf_availability()
    prompt = base.normalize_prompt(args.prompt)
    leaf_key = args.leaf_key.strip().lower()

    scene_dir = resolve_scene_dir(args.scene)
    image_dir = scene_dir / args.input_dir
    if not image_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{image_dir}（请先运行 get_onlyPlant.py）")
    images = base.list_images(image_dir)

    output_dir = scene_dir / args.output_dir
    check_dir = scene_dir / f"{args.output_dir}_check"
    preview_dir = check_dir / "preview"
    existing = [
        path for path in (output_dir, check_dir)
        if path.exists() or path.is_symlink()
    ]
    if existing and not args.overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError("以下输出已存在，请增加 --overwrite：\n" + joined)
    if args.overwrite:
        for path in (output_dir, check_dir):
            if path.exists():
                base.remove_generated_path(path)
    for directory in (output_dir, check_dir, preview_dir):
        directory.mkdir(parents=True, exist_ok=True)

    bert_dir = base.GSAM2_ROOT / "bert-base-uncased"
    base.verify_bert(bert_dir)
    dino_cfg = (
        base.GSAM2_ROOT
        / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    dino_ckpt = base.GSAM2_ROOT / "gdino_checkpoints/groundingdino_swint_ogc.pth"
    sam_ckpt = base.GSAM2_ROOT / "checkpoints/sam2.1_hiera_base_plus.pt"
    sam_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到 CUDA GPU")

    tqdm.write("[1/4] 加载 Grounding DINO 和 SAM2")
    dino = base.load_grounding_model_offline(dino_cfg, dino_ckpt, bert_dir, "cuda")
    with redirect_stdout(io.StringIO()):
        sam_model = base.build_sam2(sam_cfg, str(sam_ckpt), device="cuda")
    sam_predictor = base.SAM2ImagePredictor(sam_model)

    tqdm.write("[2/4] 逐图检测与分割叶片")
    records = []
    for image_path in tqdm(images, desc="逐图叶片识别", unit="frame"):
        try:
            label, leaves, masks, image_bgr = process_image(
                base, dino, sam_predictor, image_path, prompt, leaf_key, args,
            )
        except Exception as exc:
            tqdm.write(f"[跳过] {image_path.name}：{type(exc).__name__}: {exc}")
            records.append(
                {
                    "frame_index": len(records),
                    "image_name": image_path.name,
                    "leaf_count": 0,
                    "quality_flags": "error",
                }
            )
            continue

        mask_path = output_dir / f"{image_path.stem}.png"
        preview_path = preview_dir / f"{image_path.stem}.png"

        if label is None or len(leaves) == 0:
            # 没有检测到叶片：写全 0 label，保留帧对齐。
            if not cv2.imwrite(str(mask_path), np.zeros(image_bgr.shape[:2], dtype=np.uint8)):
                raise RuntimeError(f"无法保存：{mask_path}")
            cv2.imwrite(str(preview_path), image_bgr)
            records.append(
                {
                    "frame_index": len(records),
                    "image_name": image_path.name,
                    "leaf_count": 0,
                    "quality_flags": "leaf_missing",
                }
            )
            continue

        if args.instance:
            out_mask = label
        else:
            # 默认输出二值图：255=叶片，0=背景。
            out_mask = (label > 0).astype(np.uint8) * 255
        if not cv2.imwrite(str(mask_path), out_mask):
            raise RuntimeError(f"无法保存：{mask_path}")
        cv2.imwrite(str(preview_path), preview_leaf_instances(image_bgr, leaves, masks))
        records.append(
            {
                "frame_index": len(records),
                "image_name": image_path.name,
                "leaf_count": int(label.max()),
                "quality_flags": "",
            }
        )

    del dino, sam_predictor, sam_model
    gc.collect()
    torch.cuda.empty_cache()

    tqdm.write("[3/4] 写汇总报告")
    report_path = check_dir / "report.csv"
    with report_path.open("w", encoding="utf-8-sig", newline="") as file:
        fields = ["frame_index", "image_name", "leaf_count", "quality_flags"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    leaf_counts = [int(r["leaf_count"]) for r in records if r["quality_flags"] != "error"]
    (check_dir / "summary.json").write_text(
        json.dumps(
            {
                "script_version": SCRIPT_VERSION,
                "scene": str(scene_dir),
                "input_dir": str(image_dir),
                "frame_count": len(images),
                "processed_frame_count": len(records),
                "frames_with_leaf": sum(1 for r in records if r["leaf_count"] > 0),
                "frames_without_leaf": sum(1 for r in records if r["leaf_count"] == 0),
                "avg_leaf_count_per_frame": (
                    round(sum(leaf_counts) / len(leaf_counts), 2) if leaf_counts else 0.0
                ),
                "prompt": prompt,
                "leaf_key": leaf_key,
                "full_res": args.full_res,
                "crop_margin": args.crop_margin,
                "max_crop_size": args.max_crop_size,
                "binary_output": not args.instance,
                "leaf_mask_dir": str(output_dir.resolve()),
                "check_dir": str(check_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n完成")
    print(f"叶片 label mask: {output_dir.resolve()}")
    print(f"检查结果: {check_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
