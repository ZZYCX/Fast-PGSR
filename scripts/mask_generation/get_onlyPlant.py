#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 color_mask / object_mask 中抠出植株，生成与原图等尺寸的 RGB 原图。

输出保存到：
    <scene>/plant_images

用法：
    python scripts/mask_generation/get_onlyPlant.py --scene mungbean12_0730
    python scripts/mask_generation/get_onlyPlant.py --scene data/mungbean12_0730

说明：
    1. mask 优先使用 color_mask（植株为绿色），没有时回退到 object_mask；
    2. color_mask 中通过“绿色”像素判定植株（G 通道明显高于 R/B）；
    3. object_mask 中默认 label=1 为植株，可用 --plant-label 修改；
    4. 背景默认黑色，可用 --background 改为 white / transparent；
    5. 输出统一为 PNG（无损），避免原图为 JPG 时二次压缩损失植株像素、背景边缘不再纯黑；
    6. 输出目录已存在时需要 --overwrite；
    7. 加 --fill-holes 可填充植株 mask 内部的小空洞（叶片间露出的背景），
       只填被前景包围的空洞，不膨胀外边界（不会把叶柄/枝干周围背景圈进来）；
    8. 加 --green-filter 可按原图颜色只保留绿色植株像素：只要 RGB 里 G 通道
       占优（G ≥ R + margin 且 G ≥ B + margin）就保留，很宽松，沾一点绿都留；
       只去掉明显非绿（红/棕/灰/蓝）和黑色。余量用 --green-margin 调。
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"图像目录不存在：{image_dir}")
    images = sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix in IMAGE_SUFFIXES],
        key=natural_key,
    )
    if not images:
        raise FileNotFoundError(f"没有找到图片：{image_dir}")
    return images


def resolve_scene_dir(value: str) -> Path:
    """--scene 既支持直接路径（data/xxx），也支持场景名（xxx）。"""
    direct = Path(value)
    if direct.is_dir():
        return direct
    resolved = PROJECT_ROOT / "data" / value
    if not resolved.is_dir():
        raise FileNotFoundError(f"场景目录不存在：{resolved} 或 {direct}")
    return resolved


def choose_mask_dir(scene_dir: Path, mask_type: str) -> Path:
    color_dir = scene_dir / "color_mask"
    object_dir = scene_dir / "object_mask"

    def is_ready(directory: Path) -> bool:
        return directory.is_dir() and any(
            p.is_file() for p in directory.iterdir()
        )

    if mask_type == "color":
        if not is_ready(color_dir):
            raise FileNotFoundError(f"color_mask 目录不存在或为空：{color_dir}")
        return color_dir
    if mask_type == "object":
        if not is_ready(object_dir):
            raise FileNotFoundError(f"object_mask 目录不存在或为空：{object_dir}")
        return object_dir

    # auto：优先 color_mask（带颜色，可依据“绿色”识别植株），否则回退 object_mask。
    if is_ready(color_dir):
        return color_dir
    if is_ready(object_dir):
        return object_dir
    raise FileNotFoundError(f"{scene_dir} 下缺少 color_mask 或 object_mask")


def build_plant_mask(mask: np.ndarray, mask_type: str, plant_label: int, min_green: int) -> np.ndarray:
    """根据 mask 生成植株二值掩码（True 为植株像素）。"""
    if mask_type == "object":
        return mask == plant_label

    # color_mask：绿色像素判定，G 通道明显高于 R/B 且足够亮。
    b = mask[..., 0].astype(np.int16)
    g = mask[..., 1].astype(np.int16)
    r = mask[..., 2].astype(np.int16)
    return (g > r) & (g > b) & (g >= min_green)


def fill_enclosed_holes(plant_mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    """只填充植株 mask 内部的小空洞，不膨胀外边界。

    空洞 = 被前景（植株）包围、且不触图像边框的小背景连通域。叶片之间的
    空隙会被填上（避免抠图后露出地板背景），而叶柄/枝干周围的背景属于
    外边界，不会被填充，因此不会重新把背景圈进植株。
    """
    if max_hole_area <= 0:
        return plant_mask
    h, w = plant_mask.shape[:2]
    bg = (~plant_mask).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=8)
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        # 触到图像边框的连通域是外背景，不是空洞，跳过。
        if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h:
            continue
        if area <= max_hole_area:
            plant_mask[labels == i] = True
    return plant_mask


def refine_plant_by_green(
    image_bgr: np.ndarray,
    plant_mask: np.ndarray,
    args,
) -> np.ndarray:
    """按原图颜色后处理：只保留绿色植株像素，去掉粘连的非绿背景像素。

    判据非常宽松：只要 RGB 里绿色通道明显占优（G ≥ R + margin 且
    G ≥ B + margin）就保留，浅绿/深绿/带点绿都算，只有明显非绿
    （红/棕/灰/蓝）和黑色会被剔除，符合“沾了一点绿就保留”。
    """
    b = image_bgr[..., 0].astype(np.int16)
    g = image_bgr[..., 1].astype(np.int16)
    r = image_bgr[..., 2].astype(np.int16)
    green = (g >= r + args.green_margin) & (g >= b + args.green_margin)
    return plant_mask & green


def extract_plant(image_bgr: np.ndarray, plant_mask: np.ndarray, background: str) -> np.ndarray:
    """植株像素保留原图颜色，背景填成指定颜色；transparent 返回 RGBA。"""
    if background == "black":
        out = np.zeros_like(image_bgr)
        out[plant_mask] = image_bgr[plant_mask]
        return out
    if background == "white":
        out = np.full_like(image_bgr, 255)
        out[plant_mask] = image_bgr[plant_mask]
        return out

    # transparent：背景全 0，alpha=0；植株保留原色，alpha=255。
    out = np.zeros_like(image_bgr)
    out[plant_mask] = image_bgr[plant_mask]
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    alpha = plant_mask.astype(np.uint8) * 255
    return np.dstack([rgb, alpha])


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="从 mask 中抠出植株并生成 RGB 原图")
    p.add_argument("--scene", required=True, help="场景名（如 mungbean12_0730）或直接路径（如 data/mungbean12_0730）")
    p.add_argument("--mask-type", choices=["color", "object", "auto"], default="auto",
                   help="使用的 mask 类型，默认 auto（优先 color_mask）")
    p.add_argument("--plant-label", type=int, default=1,
                   help="object_mask 中植株的像素值，默认 1")
    p.add_argument("--min-green", type=int, default=60,
                   help="color_mask 中绿色判定的 G 通道下限，默认 60")
    p.add_argument("--background", choices=["black", "white", "transparent"], default="black",
                   help="抠图后的背景颜色，默认 black")
    p.add_argument("--fill-holes", action="store_true",
                   help="填充植株 mask 内部的小空洞（叶片间露出的背景），"
                        "只填被前景包围的空洞，不膨胀外边界")
    p.add_argument("--max-hole-area", type=int, default=5000,
                   help="--fill-holes 时填充的最大空洞面积（像素），默认 5000")
    p.add_argument("--green-filter", action="store_true",
                   help="按原图颜色只保留绿色植株像素（RGB 中 G 通道占优），"
                        "去掉粘连的非绿背景像素")
    p.add_argument("--green-margin", type=int, default=2,
                   help="--green-filter 的判定余量：G 需比 R 和 B 大至少该值才保留，"
                        "默认 2（很宽松，沾一点绿就留）；仍误删叶片可调成 0")
    p.add_argument("--overwrite", action="store_true",
                   help="输出目录已存在时删除重建")
    return p


def main() -> int:
    args = parser().parse_args()

    scene_dir = resolve_scene_dir(args.scene)
    images = list_images(scene_dir / "images")
    mask_dir = choose_mask_dir(scene_dir, args.mask_type)
    mask_type = "color" if mask_dir.name == "color_mask" else "object"
    tqdm.write(f"场景: {scene_dir}")
    tqdm.write(f"图片: {len(images)} 张，Mask: {mask_dir.name}")

    mask_by_stem = {p.stem: p for p in mask_dir.iterdir() if p.is_file()}

    output_dir = scene_dir / "plant_images"
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"输出目录已存在，请增加 --overwrite：{output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    processed = 0
    skipped_missing_mask = 0
    skipped_no_plant = 0

    for image_path in tqdm(images, desc="抠取植株", unit="frame"):
        mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is None:
            tqdm.write(f"[跳过] {image_path.name}：缺少对应 mask")
            skipped_missing_mask += 1
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            tqdm.write(f"[跳过] {image_path.name}：无法读取图片")
            skipped_missing_mask += 1
            continue

        if mask_type == "color":
            mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            tqdm.write(f"[跳过] {image_path.name}：无法读取 mask")
            skipped_missing_mask += 1
            continue

        plant_mask = build_plant_mask(mask, mask_type, args.plant_label, args.min_green)
        if args.fill_holes:
            plant_mask = fill_enclosed_holes(plant_mask, args.max_hole_area)
        if args.green_filter:
            plant_mask = refine_plant_by_green(image_bgr, plant_mask, args)
        if plant_mask.sum() == 0:
            tqdm.write(f"[跳过] {image_path.name}：未检测到植株")
            skipped_no_plant += 1
            continue

        extracted = extract_plant(image_bgr, plant_mask, args.background)
        # 统一输出为无损 PNG，避免 JPEG 二次压缩。
        out_path = output_dir / f"{image_path.stem}.png"
        if not cv2.imwrite(str(out_path), extracted):
            raise RuntimeError(f"无法保存：{out_path}")
        processed += 1

    print("\n完成")
    print(f"输出目录: {output_dir}")
    print(f"成功抠取: {processed} 张")
    print(f"缺少 mask 跳过: {skipped_missing_mask} 张")
    print(f"未检测到植株跳过: {skipped_no_plant} 张")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
