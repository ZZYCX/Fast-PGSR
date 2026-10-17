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

import torch
from scene import Scene
import os
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
from scene.app_model import AppModel
import copy
from collections import deque
import time
import shutil
import tempfile

def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0

def post_process_mesh(mesh, min_component_triangles=1):
    """
    Leaf-only mesh post-process.

    Different leaves are allowed to be disconnected components, so we do NOT
    keep only the largest N connected components. We only remove tiny mesh
    components whose triangle count is below min_component_triangles.
    """
    mesh_0 = copy.deepcopy(mesh)

    if len(mesh_0.triangles) == 0:
        print("[MESH] raw mesh has 0 triangles; skip component filtering")
        return mesh_0

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        triangle_clusters, cluster_n_triangles, cluster_area = mesh_0.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    if cluster_n_triangles.size == 0:
        print("[MESH] no connected components found")
        return mesh_0

    keep_mask = cluster_n_triangles >= min_component_triangles
    triangles_to_remove = ~keep_mask[triangle_clusters]

    print(f"[MESH] connected components : {len(cluster_n_triangles)}")
    print(f"[MESH] keep components      : {int(keep_mask.sum())}")
    print(f"[MESH] min triangles        : {min_component_triangles}")
    print(f"[MESH] raw vertices         : {len(mesh.vertices)}")
    print(f"[MESH] raw triangles        : {len(mesh.triangles)}")

    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()

    print(f"[MESH] post vertices        : {len(mesh_0.vertices)}")
    print(f"[MESH] post triangles       : {len(mesh_0.triangles)}")
    return mesh_0

def render_set(model_path, name, iteration, views, scene, gaussians, pipeline, background, args,
               app_model=None, max_depth=5.0, volume=None, use_depth_filter=False):

    # Leaf-only rendering must NOT overwrite the original PGSR render outputs.
    # Store temporary RGB/depth/normal renders under leaf_mesh/_render_cache.
    cache_root = os.path.join(
        model_path, "leaf_mesh", "_render_cache", name, f"ours_{iteration}"
    )
    gts_path = os.path.join(cache_root, "gt")
    render_path = os.path.join(cache_root, "renders")
    render_depth_path = os.path.join(cache_root, "renders_depth")
    render_normal_path = os.path.join(cache_root, "renders_normal")

    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(render_depth_path, exist_ok=True)
    os.makedirs(render_normal_path, exist_ok=True)

    total_time = 0.0
    depths_tsdf_fusion = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt, _ = view.get_image()
        start_time = time.time()
        out = render(view, gaussians, pipeline, background, args, app_model=app_model)
        end_time = time.time()
        total_time += (end_time - start_time)
        rendering = out["render"].clamp(0.0, 1.0)
        _, H, W = rendering.shape

        depth = out["plane_depth"].squeeze()
        depth_tsdf = depth.clone()
        depth_np = depth.detach().cpu().numpy()
        depth_i = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-20)
        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)

        normal = out["rendered_normal"].permute(1,2,0)
        normal = normal / (normal.norm(dim=-1, keepdim=True) + 1.0e-8)
        normal = normal.detach().cpu().numpy()
        normal = ((normal + 1) * 127.5).astype(np.uint8).clip(0, 255)

        if name == 'test':
            torchvision.utils.save_image(gt.clamp(0.0, 1.0),
                                         os.path.join(gts_path, view.image_name + ".png"))
            torchvision.utils.save_image(rendering,
                                         os.path.join(render_path, view.image_name + ".png"))
        else:
            rendering_np = (rendering.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]] * 255
                            ).detach().cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(render_path, view.image_name + ".jpg"), rendering_np)
        cv2.imwrite(os.path.join(render_depth_path, view.image_name + ".jpg"), depth_color)
        cv2.imwrite(os.path.join(render_normal_path, view.image_name + ".jpg"), normal)

        if use_depth_filter:
            view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
            depth_normal = out["depth_normal"].permute(1,2,0)
            depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
            dot = torch.sum(view_dir * depth_normal, dim=-1).abs()
            angle = torch.acos(dot)
            mask = angle > (80.0 / 180 * np.pi)
            depth_tsdf[mask] = 0
        depths_tsdf_fusion.append(depth_tsdf.squeeze().cpu())

    # ===== TSDF 融合并生成 mesh =====
    if volume is not None:
        if len(depths_tsdf_fusion) == 0:
            raise RuntimeError(f"No camera views available for split: {name}")

        depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
        for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):
            ref_depth = depths_tsdf_fusion[idx].cuda()

            # Remove invalid depth values before TSDF integration.
            ref_depth[~torch.isfinite(ref_depth)] = 0
            ref_depth[ref_depth <= 0] = 0

            if view.mask is not None:
                ref_depth[view.mask.squeeze() < 0.5] = 0

            ref_depth[ref_depth > max_depth] = 0
            ref_depth = ref_depth.detach().cpu().numpy()

            pose = np.identity(4)
            pose[:3, :3] = view.R.transpose(-1, -2)
            pose[:3, 3] = view.T
            # Use the current view's own depth resolution.
            H_view, W_view = ref_depth.shape

            color = o3d.io.read_image(os.path.join(render_path, view.image_name + ".jpg"))
            depth = o3d.geometry.Image((ref_depth * 1000).astype(np.uint16))

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)

            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                    W_view, H_view, view.Fx, view.Fy, view.Cx, view.Cy
                ),
                pose)
        """
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        mesh_path_color = os.path.join(mesh_path, "mesh_color.ply")
        o3d.io.write_triangle_mesh(mesh_path_color, mesh)

        mesh_no_color = o3d.geometry.TriangleMesh(mesh) 
        mesh_no_color.vertex_colors = o3d.utility.Vector3dVector(np.zeros((len(mesh_no_color.vertices), 3)))
        mesh_no_color.compute_vertex_normals()
        mesh_path_nocolor = os.path.join(mesh_path, "mesh_nocolor.ply")
        o3d.io.write_triangle_mesh(mesh_path_nocolor, mesh_no_color)
        """

        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()

        # 创建网格输出目录
        mesh_path = os.path.join(model_path, "leaf_mesh")
        os.makedirs(mesh_path, exist_ok=True)

        # 保存带颜色的网格
        mesh_path_color = os.path.join(mesh_path, "mesh_color.ply")
        o3d.io.write_triangle_mesh(
            mesh_path_color,
            mesh,
            write_triangle_uvs=True,
            write_vertex_colors=True,
            write_vertex_normals=True,
        )

        # 保存不带颜色的网格
        mesh_no_color = o3d.geometry.TriangleMesh(mesh)
        mesh_no_color.vertex_colors = o3d.utility.Vector3dVector(
            np.zeros((len(mesh_no_color.vertices), 3))
        )
        mesh_no_color.compute_vertex_normals()

        mesh_path_nocolor = os.path.join(mesh_path, "mesh_nocolor.ply")
        o3d.io.write_triangle_mesh(
            mesh_path_nocolor,
            mesh_no_color,
            write_triangle_uvs=True,
            write_vertex_colors=True,
            write_vertex_normals=True,
        )

        print(f"带颜色网格已保存至：{mesh_path_color}")
        print(f"无颜色网格已保存至：{mesh_path_nocolor}")
    num_frames = len(views)
    avg_time = total_time / num_frames if num_frames > 0 else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"[{name}] Rendered {num_frames} frames in {total_time:.2f}s. Average FPS: {fps:.2f}")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                 max_depth : float, voxel_size : float, min_component_triangles: int,
                 use_depth_filter : bool, leaf_ply: str, args):
    with torch.no_grad():
        # Step 1: use the ORIGINAL model only to reconstruct cameras/masks.
        scene_gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, scene_gaussians, load_iteration=iteration, shuffle=False)

        if leaf_ply is None or leaf_ply.strip() == "":
            leaf_ply = os.path.join(
                dataset.model_path,
                "point_cloud",
                f"iteration_{scene.loaded_iter}",
                "leaf_only.ply",
            )

        leaf_ply = os.path.abspath(leaf_ply)

        if not os.path.isfile(leaf_ply):
            raise FileNotFoundError(f"leaf_only Gaussian PLY not found: {leaf_ply}")

        print("=" * 80)
        print("[LEAF-ONLY RENDER]")
        print(f"model path : {dataset.model_path}")
        print(f"source path: {dataset.source_path}")
        print(f"iteration  : {scene.loaded_iter}")
        print(f"leaf ply   : {leaf_ply}")
        print("=" * 80)

        # Step 2: load leaf_only.ply into a FRESH GaussianModel.
        #
        # This Fast-PGSR fork automatically looks for semantic_state.npz next
        # to the PLY. The sidecar beside iteration_30000 belongs to the full
        # point_cloud.ply, not the leaf subset, so loading leaf_only.ply in
        # place would incorrectly attach the full-model semantic sidecar.
        #
        # Copy only leaf_only.ply to an isolated temporary directory where no
        # semantic_state.npz exists, then load it there.
        # Put the temporary copy under:
        #   /media/ubuntu2/A/LWL/code/Train/Fast-PGSR/tmp
        # i.e. the "tmp" folder under the project root where this script lives.
        project_root = os.path.dirname(os.path.abspath(__file__))
        temp_root = os.path.join(project_root, "tmp")
        os.makedirs(temp_root, exist_ok=True)

        isolated_dir = tempfile.mkdtemp(
            prefix="fast_pgsr_leaf_only_",
            dir=temp_root
        )
        isolated_leaf_ply = os.path.join(isolated_dir, "leaf_only.ply")

        try:
            shutil.copy2(leaf_ply, isolated_leaf_ply)

            gaussians = GaussianModel(dataset.sh_degree)

            # The leaf subset has already been selected. Mesh extraction only
            # needs geometry/rendering, not semantic state.
            if hasattr(args, "semantic_pipeline"):
                args.semantic_pipeline = False
            if hasattr(pipeline, "semantic_pipeline"):
                pipeline.semantic_pipeline = False

            print(f"[LEAF] isolated load path : {isolated_leaf_ply}")
            print("[LEAF] semantic sidecar   : not loaded for leaf subset")

            gaussians.load_ply(isolated_leaf_ply)

        finally:
            shutil.rmtree(isolated_dir, ignore_errors=True)

            # If project_root/tmp becomes empty after cleanup, remove it too.
            try:
                if os.path.isdir(temp_root) and not os.listdir(temp_root):
                    os.rmdir(temp_root)
            except OSError:
                pass

        leaf_count = int(gaussians.get_xyz.shape[0])
        if leaf_count <= 0:
            raise RuntimeError("leaf_only.ply contains 0 Gaussians")

        print(f"[LEAF] loaded Gaussians   : {leaf_count:,}")
        # app_model = AppModel()
        # app_model.load_weights(scene.model_path)
        # app_model.eval()
        # app_model.cuda()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0*voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), scene, gaussians, pipeline, background, args, 
                       max_depth=max_depth, volume=volume, use_depth_filter=use_depth_filter)
            print(f"extract_triangle_mesh")
            mesh = volume.extract_triangle_mesh()

            path = os.path.join(dataset.model_path, "leaf_mesh")
            os.makedirs(path, exist_ok=True)
            
            o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion.ply"), mesh, 
                                       write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
            
            mesh = post_process_mesh(mesh, min_component_triangles)
            o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion_post.ply"), mesh, 
                                       write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)

            # Mesh has been written successfully; remove temporary rendered frames.
            cache_dir = os.path.join(dataset.model_path, "leaf_mesh", "_render_cache")
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir)
                print(f"[CLEAN] removed temporary render cache: {cache_dir}")

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene, gaussians, pipeline, background, args)

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument(
        "--scene_override",
        default="",
        type=str,
        help="Force the current scene directory after cfg_args is loaded. "
             "This prevents stale absolute paths saved in cfg_args from being reused."
    )
    parser.add_argument(
        "--leaf_ply",
        default="",
        type=str,
        help="Path to leaf_only.ply. If omitted, use "
             "<model_path>/point_cloud/iteration_<loaded>/leaf_only.ply"
    )
    parser.add_argument(
        "--min_component_triangles",
        default=1,
        type=int,
        help="Keep every disconnected mesh component with at least this many triangles. Default=1 preserves all leaf components."
    )
    parser.add_argument("--mult", type=float, default=0.5)
    parser.add_argument("--use_depth_filter", action="store_true")

    args = get_combined_args(parser)

    # ------------------------------------------------------------------
    # IMPORTANT PATH FIX
    #
    # cfg_args may contain absolute paths from the machine/dataset layout
    # used during the original training. For this leaf-only re-render we
    # explicitly force all scene-dependent paths back to the CURRENT scene.
    # ------------------------------------------------------------------
    if args.scene_override:
        current_scene = os.path.abspath(args.scene_override)

        # Always force the current scene root.
        args.source_path = current_scene

        # ------------------------------------------------------------------
        # Repair stale scene-local absolute paths loaded from cfg_args.
        #
        # Your old cfg_args can contain paths such as:
        #   .../data_retry/2025.08.15_soybean9_0815/object_mask
        #   .../data_retry/2025.08.15_soybean9_0815/
        #       semantic_initialization/point_semantics.npz
        #
        # For leaf-only rendering, these must point to the CURRENT scene under
        # data/data_test/<scene>.  We repair by path suffix instead of relying
        # on one exact argument name, because different Fast-PGSR forks may
        # call the semantic-init argument differently.
        # ------------------------------------------------------------------

        def _scene_local_path(value):
            if not isinstance(value, str) or value == "":
                return value

            norm = value.replace("\\", "/")

            # Exact directory suffixes.
            dir_suffix_map = {
                "/object_mask": "object_mask",
                "/color_mask": "color_mask",
                "/semantic_initialization": "semantic_initialization",
                "/grounded_sam2_scene": "grounded_sam2_scene",
                "/images": "images",
                "/input": "input",
                "/stereo": "stereo",
            }

            for suffix, rel in dir_suffix_map.items():
                if norm.endswith(suffix):
                    return os.path.join(current_scene, rel)

            # Files stored below semantic_initialization/.
            marker = "/semantic_initialization/"
            if marker in norm:
                tail = norm.split(marker, 1)[1]
                return os.path.join(current_scene, "semantic_initialization", tail)

            # Files stored below object_mask/.
            marker = "/object_mask/"
            if marker in norm:
                tail = norm.split(marker, 1)[1]
                return os.path.join(current_scene, "object_mask", tail)

            # Files stored below color_mask/.
            marker = "/color_mask/"
            if marker in norm:
                tail = norm.split(marker, 1)[1]
                return os.path.join(current_scene, "color_mask", tail)

            return value

        for attr_name, old_value in list(vars(args).items()):
            new_value = _scene_local_path(old_value)
            if new_value != old_value:
                setattr(args, attr_name, new_value)
                print(f"[PATHFIX] {attr_name}")
                print(f"          old: {old_value}")
                print(f"          new: {new_value}")

        # Force the most important paths even if cfg_args did not expose them
        # with the same names on every Fast-PGSR version.
        args.source_path = current_scene
        args.semantic_mask_dir = os.path.join(current_scene, "object_mask")

        # Common semantic-init argument names used by customized forks.
        semantic_npz = os.path.join(
            current_scene, "semantic_initialization", "point_semantics.npz"
        )
        semantic_dir = os.path.join(current_scene, "semantic_initialization")

        for attr_name in (
            "semantic_init_path",
            "semantic_initialization_path",
            "semantic_file",
            "semantic_path",
        ):
            if hasattr(args, attr_name):
                setattr(args, attr_name, semantic_npz)

        for attr_name in (
            "semantic_init_dir",
            "semantic_initialization_dir",
        ):
            if hasattr(args, attr_name):
                setattr(args, attr_name, semantic_dir)

    print("=" * 80)
    print("[RENDER_LEAF VERSION] 2026-09-22-project-tmp-v5")
    print(f"[ARGS] model_path        : {args.model_path}")
    print(f"[ARGS] source_path       : {getattr(args, 'source_path', None)}")
    print(f"[ARGS] semantic_mask_dir : {getattr(args, 'semantic_mask_dir', None)}")
    print(f"[ARGS] leaf_ply          : {args.leaf_ply}")

    for _name, _value in sorted(vars(args).items()):
        if "semantic" in _name.lower():
            print(f"[ARGS][SEMANTIC] {_name} = {_value}")
    print("=" * 80)

    # Fail early with a clear message instead of letting Scene() fail later.
    if not os.path.isdir(args.source_path):
        raise FileNotFoundError(f"source_path does not exist: {args.source_path}")

    semantic_mask_dir = getattr(args, "semantic_mask_dir", None)
    if semantic_mask_dir and not os.path.isdir(semantic_mask_dir):
        raise FileNotFoundError(
            f"semantic_mask_dir does not exist: {semantic_mask_dir}"
        )

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.max_depth,
        args.voxel_size,
        args.min_component_triangles,
        args.use_depth_filter,
        args.leaf_ply,
        args,
    )
