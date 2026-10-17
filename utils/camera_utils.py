#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
import sys

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global_down = orig_w / 1600
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    print(f"scale {float(global_down) * float(resolution_scale)}")
                    WARNED = True
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    sys.stdout.write('\r')
    sys.stdout.write("load camera {}".format(id))
    sys.stdout.flush()

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image_width=resolution[0], image_height=resolution[1],
                  image_path=cam_info.image_path,
                  image_name=cam_info.image_name, uid=cam_info.global_id,
                  preload_img=args.preload_img,
                  ncc_scale=args.ncc_scale,
                  data_device=args.data_device,
                  object_mask_path=getattr(cam_info, "object_mask_path", None))

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def export_mask_alignment_debug(cameras, out_dir, n_sample=10, seed=0):
    """Export >=n_sample (image, coloured mask, overlay) triplets and a
    mask_alignment_report.json for cameras that carry a loaded object_mask."""
    import json
    import os
    import random
    import cv2
    from PIL import Image as PILImage

    os.makedirs(out_dir, exist_ok=True)
    cams = [c for c in cameras if c.object_mask is not None]
    rng = random.Random(seed)
    sampled = rng.sample(cams, min(n_sample, len(cams))) if cams else []
    color_map = {0: (70, 130, 180), 1: (0, 200, 0), 2: (220, 20, 60), 3: (255, 215, 0)}
    report = {"camera_total": len(cams), "matched": len(cams), "missing": 0,
              "sampled": len(sampled), "cameras": []}

    for cam in sampled:
        img, _ = cam.get_image()
        img_np = img.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        H, W = img_np.shape[:2]
        mask_np = cam.object_mask.cpu().numpy()
        colored = np.zeros((H, W, 3), dtype=np.uint8)
        for k, c in color_map.items():
            colored[mask_np == k] = c
        overlay = img_np * 0.5 + (colored.astype(np.float32) / 255.0) * 0.5
        base = os.path.join(out_dir, cam.image_name)
        cv2.imwrite(base + "_image.jpg", (img_np * 255).astype(np.uint8)[:, :, ::-1])
        cv2.imwrite(base + "_mask.png", colored)
        cv2.imwrite(base + "_overlay.jpg", (overlay * 255).astype(np.uint8)[:, :, ::-1])
        orig = PILImage.open(cam.image_path)
        counts = {str(k): int((mask_np == k).sum()) for k in range(4)}
        total = mask_np.size
        report["cameras"].append({
            "image_name": cam.image_name,
            "original_size": [orig.size[0], orig.size[1]],
            "train_size": [cam.image_width, cam.image_height],
            "mask_original_size": list(cam.mask_orig_size),
            "mask_train_size": [cam.image_width, cam.image_height],
            "class_pixels": counts,
            "class_ratios": {k: round(v / total, 6) for k, v in counts.items()},
            "unique_labels": sorted(np.unique(mask_np).tolist()),
        })
    with open(os.path.join(out_dir, "mask_alignment_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[VMA] 已导出 mask 对齐调试: {out_dir} ({len(sampled)} 组)")
    return report


def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
