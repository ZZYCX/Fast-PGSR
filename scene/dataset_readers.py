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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    global_id: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fx: float
    fy: float
    object_mask_path: str = ""

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    semantics: object = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def load_poses(pose_path, num):
    poses = []
    with open(pose_path, "r") as f:
        lines = f.readlines()
    for i in range(num):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3,3] = c2w[:3,3] * 10.0
        w2c = np.linalg.inv(c2w)
        w2c = w2c
        poses.append(w2c)
    poses = np.stack(poses, axis=0)
    return poses

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, mask_dir=""):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        # VMA: match object_mask/<same_stem>.png by file stem (never by sort order).
        object_mask_path = os.path.join(mask_dir, image_name + ".png") if mask_dir else ""

        cam_info = CameraInfo(uid=uid, global_id=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                              image_path=image_path, image_name=image_name,
                              width=width, height=height, fx=focal_length_x, fy=focal_length_y,
                              object_mask_path=object_mask_path)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')

    if mask_dir:
        names = [ci.image_name for ci in cam_infos]
        if len(names) != len(set(names)):
            raise RuntimeError("[VMA] 图像文件 stem 存在重复，object_mask 映射有歧义")
        missing = [ci.image_name for ci in cam_infos if not os.path.exists(ci.object_mask_path)]
        if missing:
            raise RuntimeError(f"[VMA] {len(missing)} 个图像缺少对应 object_mask（如 {missing[:10]}），"
                               f"匹配目录: {mask_dir}")
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def load_semantics_from_npz(semantic_init_path, filtered_ids, filtered_xyz):
    """Load <scene>/semantic_initialization/point_semantics.npz and align the initial
    semantic labels to the Fast-PGSR-filtered COLMAP points strictly by POINT3D_ID.

    The returned arrays follow exactly the order of `filtered_ids` (= the order of the
    initial Gaussians). No row-index binding, no XYZ-nearest-neighbour matching and no
    dependence on any semantic PLY order is used.

    Acceptance checks performed here:
      - filtered POINT3D_ID and NPZ POINT3D_ID have no duplicates;
      - every filtered POINT3D_ID exists in the NPZ (missing -> hard error);
      - 20 randomly sampled IDs align on XYZ between Fast-PGSR and the NPZ
        (max abs error > 1e-5 -> hard error, likely different COLMAP model).
    """
    data = np.load(semantic_init_path)
    npz_ids = np.asarray(data["point3D_ids"])
    npz_obj = np.asarray(data["object_id"])
    npz_probs = np.asarray(data["object_probs"])
    npz_conf = np.asarray(data["confidence"])
    npz_xyz = np.asarray(data["xyz"])

    if not (len(npz_ids) == len(npz_obj) == len(npz_probs) == len(npz_conf) == len(npz_xyz)):
        raise RuntimeError(
            f"[Semantic Init] NPZ 数组长度不一致: point3D_ids={len(npz_ids)} "
            f"object_id={len(npz_obj)} object_probs={len(npz_probs)} "
            f"confidence={len(npz_conf)} xyz={len(npz_xyz)}")
    if npz_probs.ndim != 2 or npz_probs.shape[1] != 4:
        raise RuntimeError(f"[Semantic Init] object_probs 应为 [N,4]，实际 {npz_probs.shape}")
    if npz_xyz.ndim != 2 or npz_xyz.shape[1] != 3:
        raise RuntimeError(f"[Semantic Init] xyz 应为 [N,3]，实际 {npz_xyz.shape}")

    # ---- no duplicate IDs ----
    if len(np.unique(npz_ids)) != len(npz_ids):
        raise RuntimeError("[Semantic Init] NPZ POINT3D_ID 存在重复，无法按 ID 精确绑定")
    if len(np.unique(filtered_ids)) != len(filtered_ids):
        raise RuntimeError("[Semantic Init] Fast-PGSR 过滤后 POINT3D_ID 存在重复，绑定异常")

    # id -> row index lookup (strict ID matching, ids are unique in the npz)
    id_to_idx = {int(pid): i for i, pid in enumerate(npz_ids)}

    n = len(filtered_ids)
    obj_id = np.zeros(n, dtype=np.int64)
    probs = np.zeros((n, npz_probs.shape[1]), dtype=np.float32)
    conf = np.zeros(n, dtype=np.float32)
    missing = []
    for row, pid in enumerate(filtered_ids):
        j = id_to_idx.get(int(pid))
        if j is None:
            missing.append(int(pid))
            continue
        obj_id[row] = int(npz_obj[j])
        probs[row] = npz_probs[j]
        conf[row] = float(npz_conf[j])

    if missing:
        raise RuntimeError(
            f"[Semantic Init] Fast-PGSR 保留的 POINT3D_ID 在 NPZ 中缺失 {len(missing)} 个"
            f"（如 {missing[:10]}），禁止静默赋默认标签。请检查语义文件与 sparse/0 是否对应同一重建。")

    # ---- XYZ alignment spot check (20 random IDs) ----
    rng = np.random.default_rng(0)
    sample_k = min(20, n)
    sample_idx = rng.choice(n, size=sample_k, replace=False)
    max_xyz_err = 0.0
    for row in sample_idx:
        j = id_to_idx[int(filtered_ids[row])]
        err = float(np.abs(filtered_xyz[row] - npz_xyz[j]).max())
        if err > max_xyz_err:
            max_xyz_err = err
    if max_xyz_err > 1e-5:
        raise RuntimeError(
            f"[Semantic Init] XYZ 对齐最大误差 {max_xyz_err:.3e} > 1e-5，"
            f"NPZ 与 sparse/0 可能来自不同的 COLMAP 模型")
    print(f"[Semantic Init] 抽查 {sample_k} 个 POINT3D_ID，max_xyz_alignment_error = {max_xyz_err:.3e}")

    stats = {
        "npz_total": int(len(npz_ids)),
        "matched": int(n),
        "missing_id_count": int(len(missing)),
        "unused_npz": int(len(npz_ids) - n),
        "max_xyz_alignment_error": max_xyz_err,
    }
    return {
        "point3D_ids": np.asarray(filtered_ids, dtype=np.int64),
        "object_id": obj_id,
        "object_probs": probs,
        "confidence": conf,
        "stats": stats,
    }

def readColmapSceneInfo(path, images, eval, llffhold=8, semantic_init_path="", semantic_mask_dir=""):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir),
                                           mask_dir=semantic_mask_dir)
    # cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : int(x.image_name.split('_')[-1]))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    js_file = f"{path}/split.json"
    train_list = None
    test_list = None
    if os.path.exists(js_file):
        with open(js_file) as file:
            meta = json.load(file)
            train_list = meta["train"]
            test_list = meta["test"]
            print(f"train_list {len(train_list)}, test_list {len(test_list)}")

    if train_list is not None:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in train_list]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_list]
        print(f"train_cam_infos {len(train_cam_infos)}, test_cam_infos {len(test_cam_infos)}")
    elif eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    # ids / xyz / rgb are kept strictly in the same order after the original
    # error>2.0 & track_length<3 filtering, so semantic binding by ID is safe.
    filtered_ids = None
    raw_num_points = 0
    if not os.path.exists(ply_path) or True:
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            filtered_ids, xyz, rgb, _, raw_num_points = read_points3D_binary(bin_path)
            print(f"xyz {xyz.shape}")
        except:
            filtered_ids, xyz, rgb, _, raw_num_points = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    semantics = None
    if semantic_init_path:
        if not os.path.exists(semantic_init_path):
            raise FileNotFoundError(f"[Semantic Init] 语义文件不存在: {semantic_init_path}")
        print(f"[Semantic Init] 加载语义文件: {semantic_init_path}")
        semantics = load_semantics_from_npz(semantic_init_path, filtered_ids, xyz)
        semantics["stats"]["raw_num_points"] = int(raw_num_points)
        semantics["stats"]["filtered_num_points"] = int(len(filtered_ids))

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           semantics=semantics)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, global_id=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}