#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
指定关键帧，保留 1 个植株和指定数量（默认 1）的 rubik，并用 SAM2 双向传播。

正式输出保存到：
    <scene>/object_mask
    <scene>/color_mask

检查结果保存到：
    <scene>/grounded_sam2_scene
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import os
import re
import shutil
import sys
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert, nms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
GSAM2_ROOT = PROJECT_ROOT / "third_party" / "Grounded-SAM-2"
if str(GSAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(GSAM2_ROOT))

from grounding_dino.groundingdino.models import build_model
from grounding_dino.groundingdino.util.inference import load_image, predict
from grounding_dino.groundingdino.util.slconfig import SLConfig
from grounding_dino.groundingdino.util.utils import clean_state_dict
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

SCRIPT_VERSION = "2026-10-17-sfm-instance-v4"
DEFAULT_PROMPT = (
    "single green leafy plant. "
    "rubik's cube with attached ArUco marker. "
    "plot."
)
PLANT_KEY = "single_green_leafy_plant"
RUBIK_KEY = "rubik"
OBJECT_NAMES = {1: PLANT_KEY, 2: "rubik_1", 3: "rubik_2"}
PALETTE = {0: (0, 0, 0), 1: (80, 210, 80), 2: (60, 80, 235), 3: (235, 150, 50)}
SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def resolve_scene_dir(scene_arg: str) -> Path:
    """解析 --scene 指向的场景目录，按优先级：
    1) 绝对路径 -> 原样使用；
    2) data/ 下存在同名目录 -> 视为场景名，保持旧行为（如 mungbean10_0730 -> data/mungbean10_0730）；
    3) 相对项目根可解析的路径 -> 直接使用（如 data_test/soybean10_0730 -> data_test/soybean10_0730）；
    4) 兜底仍回退 data/<scene>（保持原报错信息）。
    两个脚本（base/auto）必须共用同一份逻辑，保证子进程传参后解析到同一目录。
    """
    p = Path(scene_arg).expanduser()
    if p.is_absolute():
        return p
    if (PROJECT_ROOT / "data" / scene_arg).is_dir():
        return PROJECT_ROOT / "data" / scene_arg
    if (PROJECT_ROOT / p).is_dir():
        return PROJECT_ROOT / p
    return PROJECT_ROOT / "data" / scene_arg


def natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"图像目录不存在：{image_dir}")
    images = sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix in SUFFIXES],
        key=natural_key,
    )
    if not images:
        raise FileNotFoundError(f"没有找到图片：{image_dir}")
    return images


def normalize_prompt(prompt: str) -> str:
    parts = [x.strip().lower() for x in prompt.split(".") if x.strip()]
    if not parts:
        raise ValueError("提示词为空")
    return ". ".join(parts) + "."


def label_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")


def verify_bert(bert_dir: Path) -> None:
    for name in ["config.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt"]:
        if not (bert_dir / name).is_file():
            raise FileNotFoundError(bert_dir / name)
    if not any((bert_dir / name).is_file() for name in ["model.safetensors", "pytorch_model.bin"]):
        raise FileNotFoundError(f"缺少 BERT 权重：{bert_dir}")


def load_grounding_model_offline(
    config: Path,
    checkpoint: Path,
    bert_dir: Path,
    device: str,
):
    args = SLConfig.fromfile(str(config))
    args.device = device
    args.text_encoder_type = str(bert_dir.resolve())
    with redirect_stdout(io.StringIO()):
        model = build_model(args)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    model.load_state_dict(clean_state_dict(state["model"]), strict=False)
    return model.eval().to(device)


def run_grounding(
    model,
    image_tensor,
    width: int,
    height: int,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> list[dict[str, Any]]:
    with torch.inference_mode():
        boxes, scores, labels = predict(
            model=model,
            image=image_tensor,
            caption=normalize_prompt(prompt),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )
    if len(boxes) == 0:
        return []
    scale = torch.tensor([width, height, width, height], dtype=boxes.dtype)
    xyxy = box_convert(boxes=boxes.cpu() * scale, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xyxy[:, (0, 2)] = np.clip(xyxy[:, (0, 2)], 0, width)
    xyxy[:, (1, 3)] = np.clip(xyxy[:, (1, 3)], 0, height)
    return [
        {
            "order": i,
            "label": str(label),
            "label_key": label_key(str(label)),
            "score": float(score),
            "box": np.asarray(box, dtype=np.float32),
        }
        for i, (box, score, label) in enumerate(zip(xyxy, scores.cpu().numpy(), labels))
    ]


def filter_label(detections: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [d for d in detections if d["label_key"] == key]


def nms_preserve_order(detections: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    if not detections:
        return []
    boxes = torch.tensor(np.stack([d["box"] for d in detections]), dtype=torch.float32)
    scores = torch.tensor([d["score"] for d in detections], dtype=torch.float32)
    keep = nms(boxes, scores, threshold).cpu().tolist()
    keep.sort(key=lambda i: detections[i]["order"])
    return [detections[i] for i in keep]


def active_ids(rubik_count=1):
    if rubik_count not in (1, 2):
        raise ValueError("rubik_count 必须为 1 或 2")
    return tuple(range(1, rubik_count + 2))


def choose_targets(detections: list[dict[str, Any]], nms_threshold: float,
                   rubik_count=1, plant_index=0):
    plants = nms_preserve_order(filter_label(detections, PLANT_KEY), nms_threshold)
    rubiks = nms_preserve_order(filter_label(detections, RUBIK_KEY), nms_threshold)
    if not plants:
        raise RuntimeError(f"没有检测到 {PLANT_KEY}")
    if len(rubiks) < rubik_count:
        raise RuntimeError(f"NMS 后只有 {len(rubiks)} 个 rubik，需要 {rubik_count} 个")
    active_ids(rubik_count)
    if not 0 <= plant_index < len(plants):
        raise ValueError(f"plant-index={plant_index} 超出 {len(plants)} 株候选范围")

    # 保持 NMS 后检测顺序，只选指定数量的魔方。
    plant = plants[plant_index]
    selected_rubiks = rubiks[:rubik_count]
    method = f"plant_{plant_index}_first_{rubik_count}_rubiks"
    return plant, selected_rubiks, plants, rubiks, method


def read_selection(path, scene_dir, init_frame, rubik_count, image_size):
    """Read the exact boxes scored by auto; reject stale or incompatible reports."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if (resolve_scene_dir(report["scene"]).resolve() != scene_dir.resolve()
            or report["selected_frame"] != init_frame
            or report["rubik_count"] != rubik_count
            or report["selected_frame_sha256"] != hashlib.sha256((scene_dir / "images" / init_frame).read_bytes()).hexdigest()):
        raise ValueError("selection-json 场景、关键帧或魔方数量不匹配")
    selected = report["selected_objects"]
    if len(selected) != len(active_ids(rubik_count)):
        raise ValueError("selection-json 目标数量不匹配")
    for det, key in zip(selected, [PLANT_KEY] + [RUBIK_KEY] * rubik_count):
        box = np.asarray(det["box"], dtype=float)
        width, height = image_size
        if (det["label_key"] != key or box.shape != (4,)
                or not np.isfinite(box).all() or not np.isfinite(det["score"])
                or not (0 <= box[0] < box[2] <= width)
                or not (0 <= box[1] < box[3] <= height)):
            raise ValueError("selection-json 包含非法目标框")
        det["box"] = box.astype(np.float32)
    return selected, report


def tracking_quality(labels, previous, object_ids):
    # Keep legacy area columns while only checking active instances.
    areas = {obj_id: int(np.count_nonzero(labels == obj_id)) for obj_id in OBJECT_NAMES}
    flags = []
    for obj_id in object_ids:
        name = OBJECT_NAMES[obj_id]
        if areas[obj_id] == 0:
            flags.append(f"{name}_missing")
        if previous and areas[obj_id] > 0 and previous[obj_id] > 0:
            ratio = max(areas[obj_id], previous[obj_id]) / min(areas[obj_id], previous[obj_id])
            if ratio > 2.5:
                flags.append(f"{name}_area_jump")
    return areas, flags


def predict_masks(predictor: SAM2ImagePredictor, image_rgb: np.ndarray, boxes: np.ndarray, device: str):
    predictor.set_image(np.array(image_rgb, copy=True))
    cast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    with torch.inference_mode(), cast:
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    return masks.astype(bool), [float(x) for x in np.asarray(scores).reshape(-1)]


def predict_mask_on_crop(
    sam_image,
    crop_rgb: np.ndarray,
    box_crop: np.ndarray,
) -> tuple[np.ndarray, float]:
    """在单个全分辨率裁剪区域上跑 SAM2，返回 (mask, score)。

    crop_rgb 为任意尺寸的全分辨率 RGB（SAM2 内部会统一缩放到 image_size），
    box_crop 为裁剪区域内的 xyxy 检测框坐标。
    """
    sam_image.set_image(np.array(crop_rgb, copy=True))
    cast = torch.autocast("cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), cast:
        masks, scores, _ = sam_image.predict(
            point_coords=None,
            point_labels=None,
            box=box_crop.reshape(1, 4),
            multimask_output=False,
        )
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    return masks[0].astype(bool), float(np.asarray(scores).reshape(-1)[0])


def crop_region(
    image_rgb: np.ndarray,
    box,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """以 box 为中心外扩 crop_margin，限制边长 ≤ max_crop_size。

    返回 (crop, box_in_crop, cy1, cx1, cy2, cx2)。
    """
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = np.round(np.asarray(box)).astype(int).tolist()
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    margin_x = int(round(args.crop_margin * bw))
    margin_y = int(round(args.crop_margin * bh))
    cx1, cy1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    cx2, cy2 = min(w, x2 + margin_x), min(h, y2 + margin_y)
    cw, ch = cx2 - cx1, cy2 - cy1
    if (
        max(cw, ch) > args.max_crop_size
        and bw <= args.max_crop_size
        and bh <= args.max_crop_size
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
    return crop, box_crop, cy1, cx1, cy2, cx2


def segment_targets_fullres(
    sam_image,
    image_rgb: np.ndarray,
    selected: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[float]]:
    """所有活动目标（植株/魔方）各自在全分辨率裁剪区域上分割（tiling）。

    整图缩到 1024 时目标（尤其植株）被压缩，细结构精度差；这里把每个目标
    所在区域按原分辨率裁出来分割，再拼回原图坐标。
    """
    h, w = image_rgb.shape[:2]
    masks: list[np.ndarray] = []
    scores: list[float] = []
    for det in selected:
        crop, box_crop, cy1, cx1, cy2, cx2 = crop_region(image_rgb, det["box"], args)
        mask_crop, score = predict_mask_on_crop(sam_image, crop, box_crop)
        mask_full = np.zeros((h, w), dtype=bool)
        mask_full[cy1:cy2, cx1:cx2] = mask_crop
        masks.append(mask_full)
        scores.append(score)
    return np.stack(masks), scores


def refine_labels_fullres(
    sam_image,
    image_rgb: np.ndarray,
    labels: np.ndarray,
    obj_ids: tuple[int, ...],
    args: argparse.Namespace,
) -> np.ndarray:
    """对传播得到的每帧 label，用目标 mask 的包围盒在全分辨率裁剪区域上重跑
    SAM2 细化边界，再写回 label。obj_ids 为要细化的目标编号。
    """
    new_labels = labels.copy()
    for obj_id in obj_ids:
        mask = labels == obj_id
        if mask.sum() == 0:
            continue
        ys, xs = np.where(mask)
        box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        crop, box_crop, cy1, cx1, cy2, cx2 = crop_region(image_rgb, box, args)
        mask_crop, _ = predict_mask_on_crop(sam_image, crop, box_crop)
        region = new_labels[cy1:cy2, cx1:cx2]
        new_labels[cy1:cy2, cx1:cx2] = np.where(mask_crop, obj_id, region)
    return new_labels


def prepare_frames(images: list[Path], output_dir: Path) -> Path:
    frame_dir = output_dir / "_sam2_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)
    for i, src in enumerate(images):
        dst = frame_dir / f"{i:05d}.jpg"
        if src.suffix.lower() in {".jpg", ".jpeg"}:
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)
        else:
            image = cv2.imread(str(src))
            if image is None or not cv2.imwrite(str(dst), image):
                raise RuntimeError(f"无法生成临时帧：{src}")
    return frame_dir


def logits_to_labels(obj_ids, mask_logits: torch.Tensor) -> np.ndarray:
    logits = mask_logits[:, 0].detach().float().cpu().numpy()
    ids = [int(x) for x in obj_ids]
    best_idx = np.argmax(logits, axis=0)
    best_score = np.max(logits, axis=0)
    labels = np.zeros(best_score.shape, dtype=np.uint8)
    for i, obj_id in enumerate(ids):
        labels[(best_idx == i) & (best_score > 0.0)] = obj_id
    return labels


def colorize(labels: np.ndarray) -> np.ndarray:
    result = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for obj_id, color in PALETTE.items():
        result[labels == obj_id] = color
    return result


def preview_image(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = image.copy()
    colors = colorize(labels)
    foreground = labels > 0
    blended = cv2.addWeighted(image, 0.58, colors, 0.42, 0)
    result[foreground] = blended[foreground]
    for obj_id, name in OBJECT_NAMES.items():
        binary = (labels == obj_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, PALETTE[obj_id], 2)
        if binary.sum() == 0:
            continue
        m = cv2.moments(binary)
        if m["m00"] > 0:
            x = int(m["m10"] / m["m00"])
            y = int(m["m01"] / m["m00"])
            cv2.putText(result, f"{obj_id}:{name}", (max(5, x - 30), max(25, y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, PALETTE[obj_id], 2, cv2.LINE_AA)
    return result


def init_preview(image: np.ndarray, selected: list[dict[str, Any]], masks: np.ndarray, object_ids) -> np.ndarray:
    result = image.copy()
    overlay = image.copy()
    for obj_id, detection, mask in zip(object_ids, selected, masks):
        overlay[mask] = PALETTE[obj_id]
        x1, y1, x2, y2 = np.round(np.asarray(detection["box"])).astype(int).tolist()
        cv2.rectangle(result, (x1, y1), (x2, y2), PALETTE[obj_id], 3)
        cv2.putText(result, f"{obj_id}:{OBJECT_NAMES[obj_id]} {detection['score']:.3f}",
                    (x1, max(28, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    PALETTE[obj_id], 2, cv2.LINE_AA)
    return cv2.addWeighted(result, 0.58, overlay, 0.42, 0)



def serializable(d: dict[str, Any]) -> dict[str, Any]:
    result = dict(d)
    result["box"] = [float(x) for x in np.asarray(d["box"]).tolist()]
    return result



def remove_generated_path(path: Path) -> None:
    """删除脚本生成的文件、目录或软链接。"""
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--rubik-count", type=int, choices=(1, 2), default=1)
    p.add_argument("--plant-index", type=int, default=0)
    p.add_argument("--selection-json", type=Path, help="复用自动关键帧评分时的目标框")
    p.add_argument("--init-frame", required=True)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--box-threshold", type=float, default=0.30)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--nms-threshold", type=float, default=0.72)
    p.add_argument("--direction", choices=["both", "forward"], default="both")
    p.add_argument("--offload-state-to-cpu", action="store_true")
    p.add_argument("--fill-hole-area", type=int, default=8,
                   help="SAM2 视频传播时填充 mask 内部背景小洞的最大面积（低分辨率像素，"
                        "默认 8，即原行为）。设为 0 可关闭补洞，避免叶柄/枝干周围把背景圈进 mask")
    p.add_argument("--full-res", action="store_true",
                   help="所有活动目标（植株/魔方）各自在全分辨率裁剪区域上分割，"
                        "并在视频传播后逐帧全分辨率细化，避免整图缩到 1024 的精度损失，更慢")
    p.add_argument("--crop-margin", type=float, default=0.25,
                   help="--full-res 时围绕目标框外扩的比例（相对框尺寸），给 SAM2 更多上下文")
    p.add_argument("--max-crop-size", type=int, default=1024,
                   help="--full-res 时裁剪区域的最大边长，超限会收窄外扩")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--disable-instance-guard", action="store_true",
                   help="仅供旧流程诊断：关闭单株SfM正负提示约束")
    return p


def main() -> int:
    args = parser().parse_args()
    object_ids = active_ids(args.rubik_count)

    # Fast-PGSR 场景目录：data/<scene>（支持绝对路径 / 项目根相对路径 / 场景名）
    scene_dir = resolve_scene_dir(args.scene)
    images = list_images(scene_dir / "images")
    name_to_idx = {p.name: i for i, p in enumerate(images)}
    if args.init_frame not in name_to_idx:
        raise FileNotFoundError(scene_dir / "images" / args.init_frame)
    init_idx = name_to_idx[args.init_frame]
    init_path = images[init_idx]

    output_dir = scene_dir / "grounded_sam2_scene"
    object_dir = scene_dir / "object_mask"
    color_dir = scene_dir / "color_mask"

    existing_outputs = [
        path
        for path in (output_dir, object_dir, color_dir)
        if path.exists() or path.is_symlink()
    ]
    if existing_outputs and not args.overwrite:
        joined = "\n".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            "以下输出已存在，请增加 --overwrite：\n" + joined
        )
    if args.overwrite:
        remove_generated_path(output_dir)
        remove_generated_path(object_dir)
        remove_generated_path(color_dir)

    init_dir = output_dir / "initialization"
    preview_dir = output_dir / "preview"
    for directory in (init_dir, preview_dir, object_dir, color_dir):
        directory.mkdir(parents=True, exist_ok=True)

    bert_dir = GSAM2_ROOT / "bert-base-uncased"
    verify_bert(bert_dir)
    dino_cfg = (
        GSAM2_ROOT
        / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    dino_ckpt = (
        GSAM2_ROOT
        / "gdino_checkpoints/groundingdino_swint_ogc.pth"
    )
    sam_ckpt = GSAM2_ROOT / "checkpoints/sam2.1_hiera_base_plus.pt"
    sam_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到 CUDA GPU")

    tqdm.write("[1/5] 加载 Grounding DINO 和 SAM2")
    dino = load_grounding_model_offline(
        dino_cfg,
        dino_ckpt,
        bert_dir,
        "cuda",
    )
    with redirect_stdout(io.StringIO()):
        sam_image_model = build_sam2(
            sam_cfg,
            str(sam_ckpt),
            device="cuda",
        )
    sam_image = SAM2ImagePredictor(sam_image_model)

    tqdm.write(f"[2/5] 从关键帧选择 1 株植株和 {args.rubik_count} 个 rubik")
    image_rgb, image_tensor = load_image(str(init_path))
    image_bgr = cv2.imread(str(init_path))
    if image_bgr is None:
        raise RuntimeError(f"无法读取关键帧：{init_path}")
    h, w = image_rgb.shape[:2]
    if args.selection_json:
        selected, selection_report = read_selection(
            args.selection_json, scene_dir, args.init_frame, args.rubik_count, (w, h))
        plants = [selected[0]]
        rubik_candidates = selected[1:]
        method = "auto_scored_boxes"
        detections = selected
    else:
        detections = run_grounding(
            dino,
            image_tensor,
            w,
            h,
            args.prompt,
            args.box_threshold,
            args.text_threshold,
            "cuda",
        )
        plant, rubiks, plants, rubik_candidates, method = choose_targets(
            detections,
            args.nms_threshold, args.rubik_count, args.plant_index,
        )
        selected = [plant, *rubiks]
    for obj_id, detection in zip(object_ids, selected):
        tqdm.write(
            f"  ID {obj_id}: {OBJECT_NAMES[obj_id]} "
            f"(score={detection['score']:.3f})"
        )

    tqdm.write(f"[3/5] 生成 {len(object_ids)} 个初始目标 Mask")
    boxes = np.stack([item["box"] for item in selected]).astype(np.float32)
    if args.full_res:
        # 方案 B：活动目标各自在全分辨率裁剪区域上分割（tiling）。
        masks, mask_scores = segment_targets_fullres(sam_image, image_rgb, selected, args)
    else:
        masks, mask_scores = predict_masks(
            sam_image,
            image_rgb,
            boxes,
            "cuda",
        )

    for obj_id, detection, mask, score in tqdm(
        zip(object_ids, selected, masks, mask_scores),
        total=len(object_ids),
        desc="初始目标 Mask",
        unit="object",
    ):
        detection["object_id"] = obj_id
        detection["object_name"] = OBJECT_NAMES[obj_id]
        detection["sam2_mask_score"] = score
        cv2.imwrite(
            str(init_dir / f"{OBJECT_NAMES[obj_id]}.png"),
            mask.astype(np.uint8) * 255,
        )

    cv2.imwrite(
        str(init_dir / "init_preview.jpg"),
        init_preview(image_bgr, selected, masks, object_ids),
    )
    ignored = [
        detection["label_key"]
        for detection in detections
        if detection["label_key"] not in {PLANT_KEY, RUBIK_KEY}
    ]
    (init_dir / "detections.json").write_text(
        json.dumps(
            {
                "script_version": SCRIPT_VERSION,
                "scene": args.scene,
                "init_frame": args.init_frame,
                "prompt": normalize_prompt(args.prompt),
                "plant_candidate_count": len(plants),
                "rubik_candidate_count_after_nms": len(rubik_candidates),
                "rubik_selection_method": method,
                "rubik_count": args.rubik_count,
                "selected_object_ids": list(object_ids),
                "ignored_labels": ignored,
                "selected_objects": [serializable(item) for item in selected],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    del dino
    if not args.full_res and (args.rubik_count != 1 or args.disable_instance_guard):
        # 非全分辨率模式：释放 SAM 图片预测器；full_res 下保留，供传播后细化用。
        del sam_image, sam_image_model
    gc.collect()
    torch.cuda.empty_cache()

    tqdm.write("[4/5] SAM2 双向传播")
    frame_dir = prepare_frames(images, output_dir)
    with redirect_stdout(io.StringIO()):
        predictor = build_sam2_video_predictor(
            config_file=sam_cfg,
            ckpt_path=str(sam_ckpt),
            device="cuda",
            apply_postprocessing=True,
            vos_optimized=False,
        )
        # 按 --fill-hole-area 覆盖补洞面积；设为 0 即关闭补洞，避免细结构周围鼓包。
        predictor.fill_hole_area = args.fill_hole_area
    state = predictor.init_state(
        video_path=str(frame_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=args.offload_state_to_cpu,
        async_loading_frames=False,
    )
    predictor.reset_state(state)

    frame_results: dict[int, np.ndarray] = {}
    with torch.inference_mode(), torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
    ):
        for obj_id, mask in zip(object_ids, masks):
            predictor.add_new_mask(state, init_idx, obj_id, mask)

        with tqdm(
            total=len(images) - init_idx,
            desc="SAM2 正向传播",
            unit="frame",
        ) as forward_bar:
            for frame_idx, obj_ids, logits in predictor.propagate_in_video(
                state,
                start_frame_idx=init_idx,
                reverse=False,
            ):
                frame_results[int(frame_idx)] = logits_to_labels(obj_ids, logits)
                forward_bar.update(1)

        if args.direction == "both" and init_idx > 0:
            with tqdm(
                total=init_idx + 1,
                desc="SAM2 反向传播",
                unit="frame",
            ) as reverse_bar:
                for frame_idx, obj_ids, logits in predictor.propagate_in_video(
                    state,
                    start_frame_idx=init_idx,
                    reverse=True,
                ):
                    frame_results[int(frame_idx)] = logits_to_labels(obj_ids, logits)
                    reverse_bar.update(1)

    expected = (
        list(range(len(images)))
        if args.direction == "both"
        else list(range(init_idx, len(images)))
    )
    missing = [index for index in expected if index not in frame_results]
    if missing:
        raise RuntimeError(f"传播缺少 {len(missing)} 帧：{missing[:10]}")

    if args.full_res:
        # 逐帧把传播得到的活动目标 mask 在全分辨率裁剪区域上细化。
        tqdm.write("[5/5] 全分辨率细化传播 Mask")
        for idx in tqdm(expected, desc="细化 Mask", unit="frame"):
            image_path = images[idx]
            frame_bgr = cv2.imread(str(image_path))
            if frame_bgr is None:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            labels = frame_results[idx].astype(np.uint8, copy=False)
            frame_results[idx] = refine_labels_fullres(
                sam_image, frame_rgb, labels, object_ids, args,
            )
    instance_report = None
    if args.rubik_count == 1 and not args.disable_instance_guard:
        # Refine actual training PNGs, before LABEL/TRAIN. Only RGB, masks and
        # COLMAP tracks are used; no trained Gaussian or export is consulted.
        del predictor, state
        gc.collect(); torch.cuda.empty_cache()
        from scripts.mask_generation.instance_mask_guard import refine_training_masks
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            instance_report = refine_training_masks(scene_dir, images, frame_results, init_idx, sam_image)
        (output_dir / "instance_guard.json").write_text(json.dumps(instance_report, indent=2), encoding="utf-8")
        final_masks = np.stack([frame_results[init_idx] == i for i in object_ids])
        for obj_id, mask in zip(object_ids, final_masks):
            cv2.imwrite(str(init_dir / f"{OBJECT_NAMES[obj_id]}.png"), mask.astype(np.uint8) * 255)
        cv2.imwrite(str(init_dir / "init_preview.jpg"), init_preview(image_bgr, selected, final_masks, object_ids))
    if args.full_res or instance_report is not None:
        del sam_image, sam_image_model
        gc.collect(); torch.cuda.empty_cache()

    tqdm.write("[5/5] 保存并检查 Mask")
    records = []
    previous = None
    for idx in tqdm(expected, desc="保存 Mask", unit="frame"):
        image_path = images[idx]
        labels = frame_results[idx].astype(np.uint8, copy=False)
        unique_values = set(int(value) for value in np.unique(labels))
        if not unique_values.issubset({0, *object_ids}):
            raise RuntimeError(
                f"{image_path.name} 出现非法 Mask 像素值："
                f"{sorted(unique_values)}"
            )

        object_path = object_dir / f"{image_path.stem}.png"
        color_path = color_dir / f"{image_path.stem}.png"
        preview_path = preview_dir / f"{image_path.stem}.jpg"

        if not cv2.imwrite(str(object_path), labels):
            raise RuntimeError(f"无法保存：{object_path}")
        if not cv2.imwrite(str(color_path), colorize(labels)):
            raise RuntimeError(f"无法保存：{color_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"无法读取：{image_path}")
        if not cv2.imwrite(str(preview_path), preview_image(image, labels)):
            raise RuntimeError(f"无法保存：{preview_path}")

        areas, flags = tracking_quality(labels, previous, object_ids)
        records.append(
            {
                "frame_index": idx,
                "image_name": image_path.name,
                "plant_area": areas[1],
                "rubik_1_area": areas[2],
                "rubik_2_area": areas[3],
                "quality_flags": ";".join(flags),
            }
        )
        previous = areas

    report_path = output_dir / "tracking_report.csv"
    with report_path.open("w", encoding="utf-8-sig", newline="") as file:
        fields = [
            "frame_index",
            "image_name",
            "plant_area",
            "rubik_1_area",
            "rubik_2_area",
            "quality_flags",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "script_version": SCRIPT_VERSION,
                "scene": args.scene,
                "frame_count": len(images),
                "saved_frame_count": len(expected),
                "image_sha256": {images[i].name: hashlib.sha256(images[i].read_bytes()).hexdigest() for i in expected},
                "plant_index": args.plant_index,
                "init_frame": args.init_frame,
                "init_frame_index": init_idx,
                "prompt": normalize_prompt(args.prompt),
                "direction": args.direction,
                "fill_hole_area": args.fill_hole_area,
                "full_res": args.full_res,
                "instance_guard": None if instance_report is None else {
                    k: v for k, v in instance_report.items() if k != "frames"},
                "mask_sha256": {} if instance_report is None else {
                    images[i].stem: hashlib.sha256((object_dir / f"{images[i].stem}.png").read_bytes()).hexdigest() for i in expected},
                "crop_margin": args.crop_margin,
                "max_crop_size": args.max_crop_size,
                "object_ids": {str(i): OBJECT_NAMES[i] for i in object_ids},
                "inactive_object_ids": [i for i in OBJECT_NAMES if i not in object_ids],
                "rubik_selection_method": method,
                "rubik_count": args.rubik_count,
                "selected_object_ids": list(object_ids),
                "object_mask_dir": str(object_dir.resolve()),
                "color_mask_dir": str(color_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n完成")
    print("关键帧:", init_path.name)
    print("object_mask:", object_dir.resolve())
    print("color_mask:", color_dir.resolve())
    print("初始化检查:", init_dir.resolve())
    print("逐帧预览:", preview_dir.resolve())
    print("质量报告:", report_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
