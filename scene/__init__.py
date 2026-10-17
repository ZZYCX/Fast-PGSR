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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import numpy as np
import torch
from utils.training_cache import nearest_view_indices

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.source_path = args.source_path

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # VMA: mask mode is on when --semantic_mask_dir is given or <source>/object_mask exists.
        mask_dir = args.semantic_mask_dir if args.semantic_mask_dir else os.path.join(args.source_path, "object_mask")
        if not os.path.isdir(mask_dir):
            if args.semantic_mask_dir:
                raise FileNotFoundError(f"[VMA] semantic_mask_dir 不存在: {mask_dir}")
            mask_dir = ""   # default object_mask absent -> original behaviour

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval,
                                                          semantic_init_path=args.semantic_init_path,
                                                          semantic_mask_dir=mask_dir)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(f"cameras_extent {self.cameras_extent}")

        self.multi_view_num = args.multi_view_num
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            
            print("computing nearest_id")
            self.world_view_transforms = []
            camera_centers = []
            center_rays = []
            for id, cur_cam in enumerate(self.train_cameras[resolution_scale]):
                self.world_view_transforms.append(cur_cam.world_view_transform)
                camera_centers.append(cur_cam.camera_center)
                R = cur_cam.rotation_tensor
                center_ray = torch.tensor([0.0,0.0,1.0]).float().cuda()
                center_ray = center_ray@R.transpose(-1,-2)
                center_rays.append(center_ray)
            self.world_view_transforms = torch.stack(self.world_view_transforms)
            camera_centers = torch.stack(camera_centers, dim=0)
            center_rays = torch.stack(center_rays, dim=0)
            center_rays = torch.nn.functional.normalize(center_rays, dim=-1)
            nearest = nearest_view_indices(camera_centers, center_rays,
                args.multi_view_max_angle, args.multi_view_min_dis, args.multi_view_max_dis,
                self.multi_view_num)
            with open(os.path.join(self.model_path, "multi_view.json"), 'w') as file:
                for id, cur_cam in enumerate(self.train_cameras[resolution_scale]):
                    sorted_indices = nearest[id]
                    multi_view_num = len(sorted_indices)
                    json_d = {'ref_name' : cur_cam.image_name, 'nearest_name': []}
                    for index in sorted_indices[:multi_view_num]:
                        cur_cam.nearest_id.append(index)
                        cur_cam.nearest_names.append(self.train_cameras[resolution_scale][index].image_name)
                        json_d["nearest_name"].append(self.train_cameras[resolution_scale][index].image_name)
                    json_str = json.dumps(json_d, separators=(',', ':'))
                    file.write(json_str)
                    file.write('\n')
                    # print(f"frame {cur_cam.image_name}, neareast {cur_cam.nearest_names}, \
                    #       angle {angles[id][cur_cam.nearest_id]}, diss {diss[id][cur_cam.nearest_id]}")

        if args.export_mask_alignment_debug and mask_dir:
            from utils.camera_utils import export_mask_alignment_debug
            export_mask_alignment_debug(self.train_cameras[1.0],
                                        os.path.join(self.model_path, "mask_alignment_debug"))

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"),
                                    load_semantics=args.load_semantics)
        else:
            debug_export_path = None
            if args.export_semantic_binding_debug and scene_info.semantics is not None:
                debug_export_path = os.path.join(self.model_path, "semantic_binding_debug.ply")
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent,
                                           semantics=scene_info.semantics,
                                           export_debug_path=debug_export_path)

    def save(self, iteration, mask=None):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), mask)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]