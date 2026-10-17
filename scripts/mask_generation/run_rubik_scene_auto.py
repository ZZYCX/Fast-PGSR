#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动选择 1 个植株 + 指定数量（默认 1）的 rubik 的最佳关键帧，并调用传播脚本。

正式标签 Mask 由基础脚本直接保存到：
    <scene>/object_mask


    python scripts/mask_generation/run_rubik_scene_auto.py \
    --scene mungbean10_0730 \
    --prompt "single green leafy plant. rubik's cube with attached ArUco marker. plot." \
    --scan-step 5 \
    --refine-top-k 5 \
    --box-threshold 0.30 \
    --text-threshold 0.25 \
    --nms-threshold 0.72 \
    --direction both \
    --overwrite


    参数	默认值	你当前使用值	作用
--scene	必填	mungbean10_0730	指定场景：可传场景名（读 data/<场景名>/images）、项目根相对路径（如 data_test/soybean10_0730）或绝对路径
--prompt	固定默认提示词	可省略	Grounding DINO的文本提示词
--scan-step	1	5	每隔多少帧扫描一次候选关键帧
--refine-top-k	5	5	对快速评分最高的多少张候选帧进行SAM2精细评价
--box-threshold	0.30	0.30	Grounding DINO检测框置信度阈值
--text-threshold	0.25	0.25	检测框与提示词中目标文本的匹配阈值
--nms-threshold	0.72	0.72	对重复检测框执行NMS时使用的IoU阈值
--direction	both	both	从关键帧向哪个方向传播
--offload-state-to-cpu	关闭	关闭	将SAM2视频状态转移到CPU，降低显存占用但降低速度
--overwrite	关闭	开启	删除上一次脚本生成的结果后重新运行
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "run_grounded_sam2_scene.py"
SCRIPT_VERSION = "2026-10-17-sfm-instance-v4"
LOCAL_DEFAULT_PROMPT = (
    "single green leafy plant. "
    "rubik's cube with attached ArUco marker. "
    "plot."
)


def load_base():
    if not BASE_SCRIPT.is_file():
        raise FileNotFoundError(f"缺少基础传播脚本：{BASE_SCRIPT}")
    spec = importlib.util.spec_from_file_location("grounded_scene_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载：{BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = [
        "GSAM2_ROOT", "PLANT_KEY", "RUBIK_KEY", "SAM2ImagePredictor", "build_sam2",
        "filter_label", "list_images", "load_grounding_model_offline", "load_image",
        "nms_preserve_order", "normalize_prompt", "predict_masks", "resolve_scene_dir",
        "run_grounding", "verify_bert", "choose_targets", "active_ids",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "两个脚本不是同一版本，run_grounded_sam2_scene.py 缺少接口："
            + ", ".join(missing)
            + "\n请同时部署同版本的两个入口和 instance_mask_guard.py。"
        )

    base_version = getattr(module, "SCRIPT_VERSION", None)
    if base_version != SCRIPT_VERSION:
        raise RuntimeError(
            "两个脚本版本不一致："
            f"auto={SCRIPT_VERSION}, base={base_version}。"
            "\n请同时部署同版本的两个入口和 instance_mask_guard.py。"
        )

    return module


def border_penalty(box: np.ndarray, width: int, height: int) -> float:
    x1, y1, x2, y2 = np.asarray(box, dtype=float)
    mx, my = width * 0.015, height * 0.015
    touches = int(x1 <= mx) + int(y1 <= my) + int(x2 >= width - mx) + int(y2 >= height - my)
    return 0.20 * touches


def sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    value = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return min(1.0, math.log1p(max(0.0, value)) / 8.0)


def green_fraction(image_rgb: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return 0.0
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    green = (
        (hsv[..., 0] >= 25) & (hsv[..., 0] <= 100)
        & (hsv[..., 1] >= 35) & (hsv[..., 2] >= 25)
    )
    return float(green[mask.astype(bool)].mean())


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return inter / union if union else 0.0


def fast_score(plant, rubiks, selected, image_bgr) -> float:
    h, w = image_bgr.shape[:2]
    count = len(selected)
    score = float(plant["score"]) + sum(float(x["score"]) for x in selected)
    score += 1.50 if len(rubiks) == count else -0.45 * (len(rubiks) - count)
    score += 0.12 * sharpness(image_bgr)
    score -= border_penalty(plant["box"], w, h)
    for item in selected:
        score -= border_penalty(item["box"], w, h)
    if count == 2:
        boxes = torch.tensor(np.stack([x["box"] for x in selected]), dtype=torch.float32)
        overlap = float(box_iou(boxes[:1], boxes[1:2])[0, 0])
        if overlap > 0.25:
            score -= overlap
    return score


def refine_score(base_score, image_rgb, masks, scores):
    if len(masks) not in (2, 3) or len(scores) != len(masks):
        raise ValueError("需要植株及 1/2 个魔方的 Mask 和评分")
    h, w = image_rgb.shape[:2]
    green = green_fraction(image_rgb, masks[0].astype(bool))
    rubiks = [m.astype(bool) for m in masks[1:]]
    areas = [float(m.sum() / (h * w)) for m in rubiks]
    overlap = mask_iou(*rubiks) if len(rubiks) == 2 else None
    ratio = max(areas) / max(min(areas), 1e-8) if len(rubiks) == 2 else None
    score = base_score + 0.50 * float(np.mean(scores)) + 0.40 * green
    score -= 0.8 * sum(a < 0.001 for a in areas)
    if overlap is not None:
        score -= 1.20 * overlap
        if ratio > 6.0:
            score -= min(1.2, 0.12 * (ratio - 6.0))
    return score, {
        "sam2_mean_score": float(np.mean(scores)),
        "plant_green_fraction": green,
        "rubik_mask_iou": overlap,
        "rubik_1_area_ratio": areas[0],
        "rubik_2_area_ratio": areas[1] if len(areas) == 2 else None,
        "rubik_area_ratio": ratio,
    }


def serializable(det):
    return {
        "order": int(det["order"]),
        "label": str(det["label"]),
        "label_key": str(det["label_key"]),
        "score": float(det["score"]),
        "box": [float(x) for x in np.asarray(det["box"]).tolist()],
    }


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--rubik-count", type=int, choices=(1, 2), default=1)
    p.add_argument("--plant-index", type=int, default=0, help="NMS 后的目标植株索引，默认第一株")
    p.add_argument("--prompt", default=None)
    p.add_argument("--scan-step", type=int, default=1)
    p.add_argument("--refine-top-k", type=int, default=5)
    p.add_argument("--box-threshold", type=float, default=0.30)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--nms-threshold", type=float, default=0.72)
    p.add_argument("--direction", choices=["both", "forward"], default="both")
    p.add_argument("--offload-state-to-cpu", action="store_true")
    p.add_argument("--fill-hole-area", type=int, default=8,
                   help="传给 run_grounded_sam2_scene.py 的补洞面积参数，设为 0 关闭补洞")
    p.add_argument("--full-res", action="store_true",
                   help="传给 run_grounded_sam2_scene.py 的全分辨率 tiling 分割/细化开关")
    p.add_argument("--crop-margin", type=float, default=0.25,
                   help="传给 run_grounded_sam2_scene.py 的裁剪外扩比例")
    p.add_argument("--max-crop-size", type=int, default=1024,
                   help="传给 run_grounded_sam2_scene.py 的裁剪最大边长")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--disable-instance-guard", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.scan_step < 1 or args.refine_top_k < 1:
        raise ValueError("scan-step 和 refine-top-k 必须 >= 1")

    base = load_base()
    prompt = args.prompt or LOCAL_DEFAULT_PROMPT
    scene_dir = base.resolve_scene_dir(args.scene)
    images = base.list_images(scene_dir / "images")

    bert_dir = base.GSAM2_ROOT / "bert-base-uncased"
    base.verify_bert(bert_dir)
    dino_cfg = (
        base.GSAM2_ROOT
        / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    dino_ckpt = (
        base.GSAM2_ROOT
        / "gdino_checkpoints/groundingdino_swint_ogc.pth"
    )
    sam_ckpt = (
        base.GSAM2_ROOT
        / "checkpoints/sam2.1_hiera_base_plus.pt"
    )
    sam_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到 CUDA GPU")

    tqdm.write("[1/4] 加载 Grounding DINO")
    dino = base.load_grounding_model_offline(
        dino_cfg,
        dino_ckpt,
        bert_dir,
        "cuda",
    )
    scan_indices = list(range(0, len(images), args.scan_step))
    if scan_indices[-1] != len(images) - 1:
        scan_indices.append(len(images) - 1)

    candidates = []
    scan_rejections = []
    tqdm.write("[2/4] 扫描候选关键帧")
    for idx in tqdm(
        scan_indices,
        desc="扫描候选关键帧",
        unit="frame",
    ):
        path = images[idx]
        try:
            image_rgb, image_tensor = base.load_image(str(path))
            image_bgr = cv2.imread(str(path))
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
            plants = base.nms_preserve_order(
                base.filter_label(detections, base.PLANT_KEY),
                args.nms_threshold,
            )
            rubiks = base.nms_preserve_order(
                base.filter_label(detections, base.RUBIK_KEY),
                args.nms_threshold,
            )
            if not plants or len(rubiks) < args.rubik_count:
                scan_rejections.append(
                    {
                        "frame_index": idx,
                        "image_name": path.name,
                        "reason": "insufficient_targets",
                        "plant_count": len(plants),
                        "rubik_count": len(rubiks),
                    }
                )
                continue

            # 保持 Grounding DINO 原始返回顺序。
            plant, selected, _, _, _ = base.choose_targets(
                detections, args.nms_threshold, args.rubik_count, args.plant_index)
            score = fast_score(plant, rubiks, selected, image_bgr)
            candidates.append(
                {
                    "frame_index": idx,
                    "image_name": path.name,
                    "fast_score": score,
                    "final_score": None,
                    "plant": plant,
                    "rubiks": rubiks,
                    "selected_rubiks": selected,
                }
            )
        except Exception as exc:
            scan_rejections.append(
                {
                    "frame_index": idx,
                    "image_name": path.name,
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )

    if not candidates:
        raise RuntimeError(f"没有找到包含目标植株和 {args.rubik_count} 个 rubik 的候选帧")

    exact = [record for record in candidates if len(record["rubiks"]) == args.rubik_count]
    pool = exact if exact else candidates
    pool_mode = (
        f"exact_{args.rubik_count}_rubik_only"
        if exact
        else f"fallback_at_least_{args.rubik_count}_rubik"
    )
    pool.sort(key=lambda record: record["fast_score"], reverse=True)
    refine = pool[: args.refine_top_k]

    tqdm.write("[3/4] 使用 SAM2 精细评价候选关键帧")
    sam_model = base.build_sam2(sam_cfg, str(sam_ckpt), device="cuda")
    sam_predictor = base.SAM2ImagePredictor(sam_model)
    for record in tqdm(
        refine,
        desc="关键帧精细评价",
        unit="frame",
    ):
        image_rgb, _ = base.load_image(str(images[record["frame_index"]]))
        selected = [record["plant"], *record["selected_rubiks"]]
        boxes = np.stack([item["box"] for item in selected]).astype(np.float32)
        masks, scores = base.predict_masks(
            sam_predictor,
            image_rgb,
            boxes,
            "cuda",
        )
        final, details = refine_score(
            record["fast_score"],
            image_rgb,
            masks,
            scores,
        )
        record["final_score"] = final
        record["refine_details"] = details

    del sam_predictor, sam_model, dino
    gc.collect()
    torch.cuda.empty_cache()

    refine.sort(key=lambda record: record["final_score"], reverse=True)
    best = refine[0]
    best_frame = best["image_name"]

    report = {
        "script_version": SCRIPT_VERSION,
        "scene": args.scene,
        "prompt": base.normalize_prompt(prompt),
        "selected_frame": best_frame,
        "selected_frame_sha256": hashlib.sha256(images[best["frame_index"]].read_bytes()).hexdigest(),
        "selected_frame_index": best["frame_index"],
        "selected_score": best["final_score"],
        "candidate_pool_mode": pool_mode,
        "rubik_count": args.rubik_count,
        "selected_object_ids": list(base.active_ids(args.rubik_count)),
        "selected_objects": [serializable(x) for x in [best["plant"], *best["selected_rubiks"]]],
        "selection_rule": {
            "keep": [base.PLANT_KEY, base.RUBIK_KEY],
            "plant_instances": 1,
            "rubik_instances": args.rubik_count,
            "ignore": ["cube", "plot", "other"],
            "plant_index": args.plant_index,
            "instance_order": "nms_detection_order",
        },
        "scan_rejections": scan_rejections,
        "ranking": [
            {
                "image_name": record["image_name"],
                "frame_index": record["frame_index"],
                "fast_score": record["fast_score"],
                "final_score": record["final_score"],
                "rubik_count_after_nms": len(record["rubiks"]),
                "plant": serializable(record["plant"]),
                "selected_rubiks": [
                    serializable(item)
                    for item in record["selected_rubiks"]
                ],
                "refine_details": record.get("refine_details"),
            }
            for record in candidates
        ],
    }
    report_path = scene_dir / "auto_rubik_keyframe.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    tqdm.write(f"已选择关键帧: {best_frame}")
    tqdm.write("[4/4] 提取目标 Mask 并执行 SAM2 双向传播")

    command = [
        sys.executable,
        str(BASE_SCRIPT),
        "--scene",
        args.scene,
        "--init-frame",
        best_frame,
        "--rubik-count", str(args.rubik_count),
        "--plant-index", str(args.plant_index),
        "--selection-json", str(report_path.resolve()),
        "--prompt",
        prompt,
        "--box-threshold",
        str(args.box_threshold),
        "--text-threshold",
        str(args.text_threshold),
        "--nms-threshold",
        str(args.nms_threshold),
        "--direction",
        args.direction,
        "--fill-hole-area",
        str(args.fill_hole_area),
    ]
    if args.full_res:
        command += [
            "--full-res",
            "--crop-margin",
            str(args.crop_margin),
            "--max-crop-size",
            str(args.max_crop_size),
        ]
    if args.offload_state_to_cpu:
        command.append("--offload-state-to-cpu")
    if args.overwrite:
        command.append("--overwrite")
    if args.disable_instance_guard:
        command.append("--disable-instance-guard")

    return_code = int(subprocess.run(command, check=False).returncode)
    if return_code == 0:
        print("关键帧报告:", report_path.resolve())
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
