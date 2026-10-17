#
# VMA (View-Mask Alignment) acceptance test.
#
# Covers: stem-based image<->mask matching, missing-mask error, duplicate-stem
# ambiguity error, nearest-neighbour rescale at multiple resolutions, label purity,
# exact image/mask size equality, debug overlay export, no-mask original behaviour,
# load-mask-but-unused A/B identity, and dtype/memory checks.
#
# Run: python scripts/test_vma_alignment.py  (needs CUDA, fast-pgsr env)
#

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from arguments import ModelParams
from scene.cameras import Camera
from scene.colmap_loader import Image as ColmapImage
from scene.colmap_loader import Camera as ColmapCamera
from scene.dataset_readers import CameraInfo, readColmapCameras
from utils.camera_utils import loadCam, export_mask_alignment_debug

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[VMA-T] {'PASS' if cond else 'FAIL'}  {name}{(' | ' + detail) if detail else ''}")
    assert cond, f"VMA-T FAIL: {name}"


def expect_raises(name, fn, needle):
    try:
        fn()
        check(name, False, "expected error")
    except (RuntimeError, FileNotFoundError) as e:
        check(name, needle in str(e), str(e)[:100])


def make_args(resolution=-1, preload_img=True):
    return argparse.Namespace(resolution=resolution, preload_img=preload_img,
                              ncc_scale=1.0, data_device="cuda")


def make_synthetic(td, n=12, img_h=64, img_w=80):
    img_dir = os.path.join(td, "images")
    mask_dir = os.path.join(td, "object_mask")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(n):
        Image.fromarray(rng.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)) \
            .save(os.path.join(img_dir, f"cam{i:03d}.jpg"))
        m = np.zeros((img_h, img_w), dtype=np.uint8)
        m[10:30, 10:30] = 1
        m[30:50, 40:60] = 2
        m[50:60, 5:15] = 3
        Image.fromarray(m, mode="L").save(os.path.join(mask_dir, f"cam{i:03d}.png"))
    return img_dir, mask_dir


def cam_info(img_dir, mask_dir, i, img_h=64, img_w=80):
    return CameraInfo(uid=i, global_id=i, R=np.eye(3), T=np.zeros(3), FovY=0.7, FovX=0.9,
                      image_path=os.path.join(img_dir, f"cam{i:03d}.jpg"),
                      image_name=f"cam{i:03d}", width=img_w, height=img_h,
                      fx=50, fy=50,
                      object_mask_path=os.path.join(mask_dir, f"cam{i:03d}.png"))


# --------------------------------------------------------------------------- #
print("=" * 78)
print("V1 正确图像—Mask 匹配（stem）+ dtype + 标签 + 尺寸")
print("=" * 78)
with tempfile.TemporaryDirectory() as td:
    img_dir, mask_dir = make_synthetic(td)
    cam = loadCam(make_args(), 0, cam_info(img_dir, mask_dir, 0), 1.0)
    check("V1 mask 已加载", cam.object_mask is not None)
    check("V1 dtype == uint8", cam.object_mask.dtype == torch.uint8)
    check("V1 mask 形状 == (H, W) == 图像空间", cam.object_mask.shape == (64, 80))
    check("V1 尺寸与图像完全一致", cam.object_mask.shape == cam.original_image.shape[1:])
    labels = set(torch.unique(cam.object_mask).tolist())
    check("V1 标签 ⊆ {0,1,2,3}", labels.issubset({0, 1, 2, 3}), f"labels={sorted(labels)}")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("V2 缺失 Mask 时报错")
print("=" * 78)
with tempfile.TemporaryDirectory() as td:
    img_dir, mask_dir = make_synthetic(td)
    os.remove(os.path.join(mask_dir, "cam005.png"))   # missing one mask
    ext = {i: ColmapImage(id=i, qvec=[1, 0, 0, 0], tvec=np.zeros(3), camera_id=1,
                          name=f"cam{i:03d}.jpg", xys=np.zeros((1, 2)), point3D_ids=np.zeros(1))
           for i in range(12)}
    intr = {1: ColmapCamera(id=1, model="PINHOLE", width=80, height=64, params=np.array([50, 50, 40, 32]))}
    expect_raises("V2 缺失 mask 报错",
                  lambda: readColmapCameras(ext, intr, img_dir, mask_dir), "缺少对应 object_mask")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("V3 重名 stem / 映射歧义时报错")
print("=" * 78)
with tempfile.TemporaryDirectory() as td:
    img_dir, mask_dir = make_synthetic(td, n=2)
    ext = {0: ColmapImage(id=0, qvec=[1, 0, 0, 0], tvec=np.zeros(3), camera_id=1,
                          name="cam000.jpg", xys=np.zeros((1, 2)), point3D_ids=np.zeros(1)),
           1: ColmapImage(id=1, qvec=[1, 0, 0, 0], tvec=np.zeros(3), camera_id=1,
                          name="cam000.JPG", xys=np.zeros((1, 2)), point3D_ids=np.zeros(1))}  # same stem
    intr = {1: ColmapCamera(id=1, model="PINHOLE", width=80, height=64, params=np.array([50, 50, 40, 32]))}
    expect_raises("V3 重名 stem 报错",
                  lambda: readColmapCameras(ext, intr, img_dir, mask_dir), "有歧义")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("V4/V5 不同分辨率下 nearest 缩放正确、标签保持 0/1/2/3")
print("=" * 78)
with tempfile.TemporaryDirectory() as td:
    img_dir, mask_dir = make_synthetic(td, n=2)
    for res in [1, 2, 4]:
        cam = loadCam(make_args(resolution=res), 0, cam_info(img_dir, mask_dir, 0), 1.0)
        W, H = cam.image_width, cam.image_height
        ref = torch.from_numpy(
            np.array(Image.open(os.path.join(mask_dir, "cam000.png"))
                     .resize((W, H), Image.NEAREST)))
        check(f"V4 resolution={res}: nearest 缩放与参考逐像素一致",
              torch.equal(cam.object_mask.cpu(), ref), f"train_size={(W, H)}")
        lab = set(torch.unique(cam.object_mask).tolist())
        check(f"V5 resolution={res}: 标签仍 ⊆ {{0,1,2,3}}", lab.issubset({0, 1, 2, 3}))
        check(f"V5 resolution={res}: 尺寸与图像一致", cam.object_mask.shape == cam.original_image.shape[1:])

# --------------------------------------------------------------------------- #
print("=" * 78)
print("V6 训练图像与 Mask 尺寸严格一致（含 >1600 自动缩放路径）")
print("=" * 78)
with tempfile.TemporaryDirectory() as td:
    # wide image (>1600) to exercise the auto-downscale path
    img_dir = os.path.join(td, "images"); mask_dir = os.path.join(td, "object_mask")
    os.makedirs(img_dir); os.makedirs(mask_dir)
    W0, H0 = 2000, 1000
    Image.fromarray(np.random.RandomState(1).randint(0, 255, (H0, W0, 3), dtype=np.uint8)) \
        .save(os.path.join(img_dir, "wide.jpg"))
    m = np.zeros((H0, W0), dtype=np.uint8); m[100:400, 100:900] = 2
    Image.fromarray(m, mode="L").save(os.path.join(mask_dir, "wide.png"))
    ci = CameraInfo(uid=0, global_id=0, R=np.eye(3), T=np.zeros(3), FovY=0.7, FovX=0.9,
                    image_path=os.path.join(img_dir, "wide.jpg"), image_name="wide",
                    width=W0, height=H0, fx=500, fy=500,
                    object_mask_path=os.path.join(mask_dir, "wide.png"))
    cam = loadCam(make_args(resolution=-1), 0, ci, 1.0)     # auto 1.6K downscale
    check("V6 >1600 自动缩放后 width<=1600", cam.image_width <= 1600, f"W={cam.image_width}")
    check("V6 缩放后 mask 尺寸 == 图像尺寸", cam.object_mask.shape == cam.original_image.shape[1:])
    check("V6 缩放后标签保持", set(torch.unique(cam.object_mask).tolist()).issubset({0, 1, 2, 3}))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("V7 调试叠加图导出 + 报告 JSON")
print("=" * 78)
with tempfile.TemporaryDirectory() as td:
    img_dir, mask_dir = make_synthetic(td, n=12)
    cams = [loadCam(make_args(), i, cam_info(img_dir, mask_dir, i), 1.0) for i in range(12)]
    out = os.path.join(td, "mask_alignment_debug")
    report = export_mask_alignment_debug(cams, out, n_sample=10, seed=7)
    files = sorted(os.listdir(out))
    overlays = [f for f in files if f.endswith("_overlay.jpg")]
    imgs = [f for f in files if f.endswith("_image.jpg")]
    masks = [f for f in files if f.endswith("_mask.png")]
    check("V7 导出 ≥10 组叠加图", len(overlays) >= 10, f"overlays={len(overlays)}")
    check("V7 导出 ≥10 组原图", len(imgs) >= 10)
    check("V7 导出 ≥10 组彩色mask", len(masks) >= 10)
    check("V7 报告 JSON 存在", "mask_alignment_report.json" in files)
    check("V7 报告含相机数/匹配数/缺失数",
          "camera_total" in report and report["camera_total"] == 12
          and report["matched"] == 12 and report["missing"] == 0)
    c0 = report["cameras"][0]
    check("V7 报告字段齐全（尺寸/类别/unique_labels）",
          "original_size" in c0 and "train_size" in c0 and "mask_original_size" in c0
          and "mask_train_size" in c0 and "class_pixels" in c0
          and "class_ratios" in c0 and "unique_labels" in c0)
    print(f"[VMA-T] 人工检查路径: {out}")
    print(f"[VMA-T] 示例: {os.path.join(out, overlays[0])}")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("V8 无 Mask 模式：原行为不变")
print("=" * 78)
with tempfile.TemporaryDirectory() as td:
    img_dir, _ = make_synthetic(td, n=2)
    ci_none = CameraInfo(uid=0, global_id=0, R=np.eye(3), T=np.zeros(3), FovY=0.7, FovX=0.9,
                         image_path=os.path.join(img_dir, "cam000.jpg"), image_name="cam000",
                         width=80, height=64, fx=50, fy=50, object_mask_path="")
    cam0 = loadCam(make_args(), 0, ci_none, 1.0)
    check("V8 无 mask 时 object_mask is None", cam0.object_mask is None)
    check("V8 图像正常加载", cam0.original_image is not None and cam0.original_image.shape == (3, 64, 80))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("V9 加载 Mask 但不使用 → 图像张量与 A/B 完全一致")
print("=" * 78)
with tempfile.TemporaryDirectory() as td:
    img_dir, mask_dir = make_synthetic(td, n=2)
    cam_a = loadCam(make_args(), 0, cam_info(img_dir, mask_dir, 0), 1.0)              # with mask
    ci_b = CameraInfo(uid=0, global_id=0, R=np.eye(3), T=np.zeros(3), FovY=0.7, FovX=0.9,
                      image_path=os.path.join(img_dir, "cam000.jpg"), image_name="cam000",
                      width=80, height=64, fx=50, fy=50, object_mask_path="")          # no mask
    cam_b = loadCam(make_args(), 0, ci_b, 1.0)
    check("V9 有/无 mask 的训练图像逐像素一致",
          torch.equal(cam_a.original_image, cam_b.original_image))
    check("V9 分辨率一致", (cam_a.image_width, cam_a.image_height) == (cam_b.image_width, cam_b.image_height))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("V10 Mask dtype 与显存占用")
print("=" * 78)
with tempfile.TemporaryDirectory() as td:
    img_dir, mask_dir = make_synthetic(td, n=2)
    cam = loadCam(make_args(), 0, cam_info(img_dir, mask_dir, 0), 1.0)
    n_bytes = cam.object_mask.numel() * cam.object_mask.element_size()
    check("V10 element_size == 1 (uint8)", cam.object_mask.element_size() == 1)
    check("V10 显存 == H*W bytes", n_bytes == 64 * 80, f"{n_bytes} bytes")
    check("V10 位于与图像相同 device", cam.object_mask.device == cam.original_image.device)

# --------------------------------------------------------------------------- #
print("=" * 78)
fails = [n for n, ok in RESULTS if not ok]
print(f"VMA 测试总计 {len(RESULTS)} 项检查, 失败 {len(fails)} 项")
if fails:
    print("失败项:", fails[:20])
    raise SystemExit(1)
print("VMA 对齐测试全部通过 ✔")
print("=" * 78)
