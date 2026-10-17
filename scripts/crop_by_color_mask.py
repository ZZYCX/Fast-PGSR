#!/usr/bin/env python3
"""按 color_mask 从原始 RGB 场景中抠出目标，背景置黑，目标保留原 RGB 颜色。

对 data/<scene>/images/*.jpg 与 data/<scene>/color_mask/*.png：
  - 依据文件名 stem 一一配对（如 frame_000001）；
  - 非黑色像素区域 = 目标 -> 保留原图该像素的 RGB；
  - 黑色像素区域 = 背景 -> 置为纯黑 (0,0,0)；
  - 输出到 data_test/<scene>/input/frame_000001.png。

用法示例：
  python scripts/crop_by_color_mask.py --scene mungbean10_0730
  python scripts/crop_by_color_mask.py --scene mungbean11_0730 --output data_test2
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image


def load_rgb(path):
    """加载图片为 HxWx3 uint8 RGB numpy 数组。"""
    with Image.open(path) as im:
        arr = np.array(im.convert("RGB"))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise RuntimeError(f"图片通道数异常: {path} shape={arr.shape}")
    return arr


def main():
    parser = argparse.ArgumentParser(description="按 color_mask 抠出目标，背景置黑")
    parser.add_argument("--scene", default="mungbean10_0730",
                        help="data 下的场景文件夹名")
    parser.add_argument("--source", default="data", help="场景数据根目录")
    parser.add_argument("--output", default="data_test", help="输出根目录")
    parser.add_argument("--image-dir", default="images", help="原图所在子目录")
    parser.add_argument("--mask-dir", default="color_mask", help="mask 所在子目录")
    parser.add_argument("--out-dir", default="input", help="输出子目录（置于 <output>/<scene>/ 下）")
    parser.add_argument("--image-ext", default=".jpg", help="原图扩展名")
    parser.add_argument("--mask-ext", default=".png", help="mask 扩展名")
    parser.add_argument("--out-ext", default=".png", help="输出扩展名")
    args = parser.parse_args()

    img_dir = os.path.join(args.source, args.scene, args.image_dir)
    mask_dir = os.path.join(args.source, args.scene, args.mask_dir)
    out_dir = os.path.join(args.output, args.scene, args.out_dir)

    if not os.path.isdir(img_dir):
        sys.exit(f"[错误] 找不到原图目录: {img_dir}")
    if not os.path.isdir(mask_dir):
        sys.exit(f"[错误] 找不到 mask 目录: {mask_dir}")

    images = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith(args.image_ext.lower())
    )
    if not images:
        sys.exit(f"[错误] {img_dir} 下没有 {args.image_ext} 图片")

    os.makedirs(out_dir, exist_ok=True)

    ok, missing, skipped = 0, [], []
    total_target_px = 0  # 累计目标像素数（用于统计）
    for name in images:
        stem = os.path.splitext(name)[0]
        mask_name = stem + args.mask_ext
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            missing.append(name)
            continue
        try:
            img = load_rgb(os.path.join(img_dir, name))
            mask = load_rgb(mask_path)
        except Exception as e:
            skipped.append((name, str(e)))
            continue

        if img.shape[:2] != mask.shape[:2]:
            print(f"  [警告] 尺寸不一致 {name}: image={img.shape[:2]} mask={mask.shape[:2]}，"
                  f"将 mask 调整为原图尺寸")
            mask = np.array(
                Image.fromarray(mask).resize((img.shape[1], img.shape[0]),
                                             Image.Resampling.NEAREST)
            )

        # 非黑色像素 = 目标；目标保留原 RGB，背景置黑
        target = (mask != 0).any(axis=2)
        out = np.zeros_like(img)
        out[target] = img[target]

        Image.fromarray(out).save(os.path.join(out_dir, stem + args.out_ext))
        ok += 1
        total_target_px += int(target.sum())

    # 汇总
    print(f"[完成] 场景 {args.scene}")
    print(f"  原图总数: {len(images)}")
    print(f"  成功抠图: {ok}")
    print(f"  目标平均占比: {total_target_px / ok / (img.shape[0] * img.shape[1]) * 100:.2f}% "
          f"(基于最后一张尺寸 {img.shape[1]}x{img.shape[0]})")
    if missing:
        print(f"  缺少对应 mask: {missing}")
    if skipped:
        print(f"  处理失败: {skipped}")
    print(f"  输出目录: {out_dir}")
    return 0 if (ok and not missing and not skipped) else 1


if __name__ == "__main__":
    sys.exit(main())
