from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from generic_utils import mls_transform_pytorch_rot
from mls import mls_denoising
from gaussian_utils import  apply_indices_to_gaussian_data, quat_multiply, sh_rotate, apply_transformation_matrix_to_points, apply_transformation_matrix_to_points_torch, matrix_to_quaternion_wxyz
from pca_utils import align_to_xy_plane, align_to_xy_plane_with_tips

### 3RD
from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment
from pytorch3d.loss import chamfer_distance
import numpy as np
import fpsample
import torch
import math

###############################################################

def getProjectionMatrix(znear, zfar, fovX, fovY, orthographic=False):
    """Create projection matrix."""
    if not orthographic:
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4)
        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)

        return P
    else:
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = 1
        bottom = -top
        right = tanHalfFovX * 1 / tanHalfFovY
        left = -right
        P = np.zeros((4, 4))
        z_sign = 1.0
        P[0, 0] = 2.0 / (right - left)
        P[0, 3] = -(right + left) / (right - left)
        P[1, 1] = 2.0 / (top - bottom)
        P[1, 3] = -(top + bottom) / (top - bottom)
        P[2, 2] = -2.0 / (zfar - znear)
        P[2, 3] = -(zfar + znear) / (zfar - znear)
        P[3, 3] = z_sign
        return (right - left) / 2, (top - bottom) / 2, P

def get_full_proj_transform(world_view_transform, znear, zfar, FoVx, FoVy, orthographic=False):
    """Get full projection transformation."""
    if not orthographic:
        return world_view_transform
    else:
        tanfovx, tanfovy, projection_matrix = getProjectionMatrix(
            znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy, orthographic=True
        )
        full_proj_transform = world_view_transform @ projection_matrix.T 
        return tanfovx, tanfovy, full_proj_transform

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    """Create world-to-view transformation matrix."""
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def fps_corr_search(xyz, xyz_tar, NUM_CORR):
    source_ctrl_indices = fpsample.bucket_fps_kdline_sampling(xyz, NUM_CORR, h=3, start_idx=0)
    target_ctrl_indices = fpsample.bucket_fps_kdline_sampling(xyz_tar, NUM_CORR, h=3, start_idx=0)
    
    source_ctrl_indices = np.array(source_ctrl_indices, dtype=np.int32)
    target_ctrl_indices = np.array(target_ctrl_indices, dtype=np.int32)
    
    source_ctrl_points_coords = xyz[source_ctrl_indices]
    target_ctrl_points_coords = xyz_tar[target_ctrl_indices]
    distance_matrix = np.linalg.norm(
        source_ctrl_points_coords[:, None, :] - target_ctrl_points_coords[None, :, :], 
        axis=2
    )  # [64, 64]
    _, target_matched_indices = linear_sum_assignment(distance_matrix)
    
    target_matched_indices = np.array(target_matched_indices, dtype=np.int32)
    target_ctrl_indices_matched = target_ctrl_indices[target_matched_indices]
    target_ctrl_indices = target_ctrl_indices_matched
    
    return source_ctrl_indices, target_ctrl_indices

def get_transform_template_mesh_pca(
    source_segment: dict,
    target_segment: dict,
    root_point: np.ndarray = None):

    source_denoised_indices = source_segment.get("denoised_indices")
    target_denoised_indices = target_segment.get("denoised_indices")
    
    if source_denoised_indices is not None:
        denoised_template_segment = apply_indices_to_gaussian_data(source_segment["original_data"], source_denoised_indices)
        denoised_template_labels = source_segment["labels"][source_denoised_indices]
    else:
        denoised_template_segment = source_segment["original_data"]
        denoised_template_labels = source_segment["labels"]
        denoised_indices, _, _ = mls_denoising(denoised_template_segment.xyz, 0.1)
        denoised_template_segment = apply_indices_to_gaussian_data(denoised_template_segment, denoised_indices)
        denoised_template_labels = denoised_template_labels[denoised_indices]
    if target_denoised_indices is not None:
        denoised_target_segment = apply_indices_to_gaussian_data(target_segment["original_data"], target_denoised_indices)
        denoised_target_labels = target_segment["labels"][target_denoised_indices]
    else:
        denoised_target_segment = target_segment["original_data"]
        denoised_target_labels = target_segment["labels"]
        denoised_indices, _, _ = mls_denoising(denoised_target_segment.xyz, 0.1)
        denoised_target_segment = apply_indices_to_gaussian_data(denoised_target_segment, denoised_indices)
        denoised_target_labels = denoised_target_labels[denoised_indices]
    
    
    source_segment_tip_point = source_segment.get("apex_point", None)
    source_segment_base_point = source_segment.get("base_point", None)
    target_segment_tip_point = target_segment.get("apex_point", None)
    target_segment_base_point = target_segment.get("base_point", None)
    
    transformation_matrix_template, _, _ = align_to_xy_plane_with_tips(
        denoised_template_segment.xyz, 
        source_segment_tip_point, 
        source_segment_base_point,
        root_point=root_point
    )
    
    transformation_matrix_target, _, _ = align_to_xy_plane_with_tips(
        denoised_target_segment.xyz, 
        target_segment_tip_point, 
        target_segment_base_point,
        root_point=root_point
    )
    
    inverse_transformation_matrix_target = np.linalg.inv(transformation_matrix_target)

    final_transformation_matrix = inverse_transformation_matrix_target @ transformation_matrix_template
    
    return final_transformation_matrix


def get_transform_template_mesh_mls_corr_kai(
    source_segment: dict,
    target_segment: dict,
    verts: np.ndarray,
    num_corr: int = 64,
    sigma: float = 0.1,
    corr_weights: np.ndarray = None,
    additional_corr_indices_pair: tuple[np.ndarray, np.ndarray] = None,
    additional_corr_pair: tuple[np.ndarray, np.ndarray] = None):

    source_denoised_indices = source_segment.get("denoised_indices")
    target_denoised_indices = target_segment.get("denoised_indices")
    
    if source_denoised_indices is not None:
        denoised_template_segment = apply_indices_to_gaussian_data(source_segment["original_data"], source_denoised_indices)
        denoised_template_labels = source_segment["labels"][source_denoised_indices]
    else:
        denoised_template_segment = source_segment["original_data"]
        denoised_template_labels = source_segment["labels"]
        denoised_indices, _, _ = mls_denoising(denoised_template_segment.xyz, 0.1)
        denoised_template_segment = apply_indices_to_gaussian_data(denoised_template_segment, denoised_indices)
        denoised_template_labels = denoised_template_labels[denoised_indices]
    if target_denoised_indices is not None:
        denoised_target_segment = apply_indices_to_gaussian_data(target_segment["original_data"], target_denoised_indices)
        denoised_target_labels = target_segment["labels"][target_denoised_indices]
    else:
        denoised_target_segment = target_segment["original_data"]
        denoised_target_labels = target_segment["labels"]
        denoised_indices, _, _ = mls_denoising(denoised_target_segment.xyz, 0.1)
        denoised_target_segment = apply_indices_to_gaussian_data(denoised_target_segment, denoised_indices)
        denoised_target_labels = denoised_target_labels[denoised_indices]
    
    transformed_template_xyz = denoised_template_segment.xyz
    transformed_target_xyz = denoised_target_segment.xyz
    
    len_additional_indices = 0
    len_additional_points = 0
    if num_corr > 0:
        # Find correspondence points using FPS
        template_ctrl_indices, target_ctrl_indices = fps_corr_search(
            transformed_template_xyz, 
            transformed_target_xyz, 
            num_corr
        )
        # Add additional correspondences if provided
        if additional_corr_indices_pair is not None:
            template_ctrl_additional_indices, target_ctrl_additional_indices = additional_corr_indices_pair
            template_ctrl_indices = np.concatenate([template_ctrl_indices, template_ctrl_additional_indices])
            target_ctrl_indices = np.concatenate([target_ctrl_indices, target_ctrl_additional_indices])
            len_additional_indices = len(template_ctrl_additional_indices)
            # corr_weights_additional = np.ones((len(template_ctrl_additional_indices), 1), dtype=np.float32)
            # corr_weights = np.concatenate([corr_weights, corr_weights_additional])
        
        if additional_corr_pair is not None:
            template_ctrl_additional_points, target_ctrl_additional_points = additional_corr_pair
            len_additional_points = len(template_ctrl_additional_points)
            # corr_weights_additional = np.ones((len(template_ctrl_additional_points), 1), dtype=np.float32)
            # corr_weights = np.concatenate([corr_weights, corr_weights_additional])
    else:
        template_ctrl_indices = []
        target_ctrl_indices = []
        if additional_corr_indices_pair is not None:
            template_ctrl_additional_indices, target_ctrl_additional_indices = additional_corr_indices_pair
            template_ctrl_indices = np.array(template_ctrl_additional_indices, dtype=np.int32)
            target_ctrl_indices = np.array(target_ctrl_additional_indices, dtype=np.int32)
            len_additional_indices = len(template_ctrl_additional_indices)
            # corr_weights_additional = np.ones((len(template_ctrl_additional_indices), 1), dtype=np.float32)
            # corr_weights = np.array(corr_weights_additional, dtype=np.float32)
        
        if additional_corr_pair is not None:
            template_ctrl_additional_points, target_ctrl_additional_points = additional_corr_pair
            print(template_ctrl_additional_points.shape, target_ctrl_additional_points.shape)
            len_additional_points = len(template_ctrl_additional_points)
            # corr_weights_additional = np.ones((len(template_ctrl_additional_points), 1), dtype=np.float32)
            # corr_weights = np.array(corr_weights_additional, dtype=np.float32)
    
    len_append = len_additional_indices + len_additional_points
    if corr_weights is None:
        corr_weights = np.ones((len_append, 1), dtype=np.float32)
    else:
        corr_weights = np.concatenate([corr_weights, np.ones((len_append, 1), dtype=np.float32)])
            
    # Get correspondence points (convert to torch for MLS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    template_corr_points = torch.from_numpy(transformed_template_xyz[template_ctrl_indices]).to(device).float()
    target_corr_points = torch.from_numpy(transformed_target_xyz[target_ctrl_indices]).to(device).float()
    
    if additional_corr_pair is not None:
        template_additional_corr_points = torch.from_numpy(template_ctrl_additional_points).to(device).float()
        target_additional_corr_points = torch.from_numpy(target_ctrl_additional_points).to(device).float()
        
        template_corr_points = torch.cat([template_corr_points, template_additional_corr_points], dim=0)
        target_corr_points = torch.cat([target_corr_points, target_additional_corr_points], dim=0)
    
    # print(template_corr_points.shape, target_corr_points.shape)
    # print(corr_weights.shape)
    # MLS parameters
    sigma = torch.tensor(sigma, device=device).float()
    
    # Initialize correspondence weights (could be optimized later)
    corr_weights = torch.from_numpy(corr_weights).to(device).float()
    
    # IMPORTANT: Transform vertices to template's XY space first
    # This ensures vertices and template correspondence points are in the same coordinate system
    # verts_in_template_xy = apply_transformation_matrix_to_points(
    #     verts, 
    #     transformation_matrix_template
    # )
    
    # Apply MLS transformation in XY plane
    verts_torch = torch.from_numpy(verts).to(device).float()
    rots_dummy = torch.zeros((verts.shape[0], 4), device=device).float()  # Dummy rotations
    
    try:
        transformed_verts, _ = mls_transform_pytorch_rot(
            verts_torch, 
            rots_dummy,
            template_corr_points, 
            target_corr_points, 
            # sigma=sigma,
            # corr_weights=corr_weights
        )
        
        transformed_gaussians, _ = mls_transform_pytorch_rot(
            torch.from_numpy(transformed_template_xyz).to(device).float(), 
            torch.zeros((transformed_template_xyz.shape[0], 4), device=device).float(),
            template_corr_points, 
            target_corr_points, 
            # sigma=sigma,
            # corr_weights=corr_weights
        )
        
        template_transformation_matrix = source_segment.get("add_info", {}).get("transformation_matrix", None)
        # if template_transformation_matrix is not None:
        #     template_transformation_matrix = align_to_xy_plane_with_tips(
        #         transformed_verts.cpu().numpy(),
        #         source_segment["apex_point"],
        #         source_segment["base_point"]
        #     )
        template_transformation_matrix = np.linalg.inv(template_transformation_matrix)
        
        target_transformation_matrix = target_segment.get("add_info", {}).get("transformation_matrix", None)
        # if target_transformation_matrix is not None:
        #     target_transformation_matrix = align_to_xy_plane_with_tips(
        #         transformed_gaussians.cpu().numpy(),
        #         target_segment["apex_point"],
        #         target_segment["base_point"]
        #     )
        target_transformation_matrix = np.linalg.inv(target_transformation_matrix)
        
        template_corr_points_transformed = template_corr_points.cpu().numpy()
        target_corr_points_transformed = target_corr_points.detach().cpu().numpy()
        
        # print(template_transformation_matrix.shape, target_transformation_matrix.shape)
        # print(template_corr_points_transformed.shape, target_corr_points_transformed.shape)
        
        
        # template_corr_points_transformed = apply_transformation_matrix_to_points(template_corr_points_transformed, template_transformation_matrix)
        # target_corr_points_transformed = apply_transformation_matrix_to_points(target_corr_points_transformed, target_transformation_matrix)
        
        return transformed_verts.cpu().numpy(), transformed_gaussians.cpu().numpy(), \
            (template_corr_points_transformed, target_corr_points_transformed)
        
    except Exception as e:
        print(f"MLS transformation failed: {e}")


def get_transform_template_mesh_mls_corr(
    source_segment: dict,
    target_segment: dict,
    verts: np.ndarray,
    num_corr: int = 64,
    sigma: float = 0.1,
    corr_weights: np.ndarray = None,
    root_point: np.ndarray = None,
    additional_corr_indices_pair: tuple[np.ndarray, np.ndarray] = None,
    additional_corr_pair: tuple[np.ndarray, np.ndarray] = None):

    source_denoised_indices = source_segment.get("denoised_indices")
    target_denoised_indices = target_segment.get("denoised_indices")
    
    if source_denoised_indices is not None:
        denoised_template_segment = apply_indices_to_gaussian_data(source_segment["original_data"], source_denoised_indices)
        denoised_template_labels = source_segment["labels"][source_denoised_indices]
    else:
        denoised_template_segment = source_segment["original_data"]
        denoised_template_labels = source_segment["labels"]
        denoised_indices, _, _ = mls_denoising(denoised_template_segment.xyz, 0.1)
        denoised_template_segment = apply_indices_to_gaussian_data(denoised_template_segment, denoised_indices)
        denoised_template_labels = denoised_template_labels[denoised_indices]
    if target_denoised_indices is not None:
        denoised_target_segment = apply_indices_to_gaussian_data(target_segment["original_data"], target_denoised_indices)
        denoised_target_labels = target_segment["labels"][target_denoised_indices]
    else:
        denoised_target_segment = target_segment["original_data"]
        denoised_target_labels = target_segment["labels"]
        denoised_indices, _, _ = mls_denoising(denoised_target_segment.xyz, 0.1)
        denoised_target_segment = apply_indices_to_gaussian_data(denoised_target_segment, denoised_indices)
        denoised_target_labels = denoised_target_labels[denoised_indices]
    
    source_segment_tip_point = source_segment["apex_point"]
    source_segment_base_point = source_segment["base_point"]
    target_segment_tip_point = target_segment["apex_point"]
    target_segment_base_point = target_segment["base_point"]
    if source_segment_tip_point is not None and source_segment_base_point is not None:
        transformation_matrix_template, _, _ = align_to_xy_plane_with_tips(
            denoised_template_segment.xyz, 
            source_segment_tip_point, 
            source_segment_base_point,
            root_point=root_point
        )
    else:
        # Get PCA alignment matrices
        transformation_matrix_template, _, _ = align_to_xy_plane(
            denoised_template_segment.xyz, 
            stem_labels=denoised_template_labels
        )

    if target_segment_tip_point is not None and target_segment_base_point is not None:
        transformation_matrix_target, _, _ = align_to_xy_plane_with_tips(
            denoised_target_segment.xyz, 
            target_segment_tip_point, 
            target_segment_base_point,
            root_point=root_point
        )
    else:
        transformation_matrix_target, _, _ = align_to_xy_plane(
        denoised_target_segment.xyz, 
        stem_labels=denoised_target_labels
    )

    transformed_template_xyz = apply_transformation_matrix_to_points(
        denoised_template_segment.xyz, 
        transformation_matrix_template
    )
    
    transformed_target_xyz = apply_transformation_matrix_to_points(
        denoised_target_segment.xyz, 
        transformation_matrix_target
    )
    
    len_additional_indices = 0
    len_additional_points = 0
    if num_corr > 0:
        # Find correspondence points using FPS
        template_ctrl_indices, target_ctrl_indices = fps_corr_search(
            transformed_template_xyz, 
            transformed_target_xyz, 
            num_corr
        )
        # Add additional correspondences if provided
        if additional_corr_indices_pair is not None:
            template_ctrl_additional_indices, target_ctrl_additional_indices = additional_corr_indices_pair
            template_ctrl_indices = np.concatenate([template_ctrl_indices, template_ctrl_additional_indices])
            target_ctrl_indices = np.concatenate([target_ctrl_indices, target_ctrl_additional_indices])
            len_additional_indices = len(template_ctrl_additional_indices)
            # corr_weights_additional = np.ones((len(template_ctrl_additional_indices), 1), dtype=np.float32)
            # corr_weights = np.concatenate([corr_weights, corr_weights_additional])
        
        if additional_corr_pair is not None:
            template_ctrl_additional_points, target_ctrl_additional_points = additional_corr_pair
            len_additional_points = len(template_ctrl_additional_points)
            # corr_weights_additional = np.ones((len(template_ctrl_additional_points), 1), dtype=np.float32)
            # corr_weights = np.concatenate([corr_weights, corr_weights_additional])
    else:
        template_ctrl_indices = []
        target_ctrl_indices = []
        if additional_corr_indices_pair is not None:
            template_ctrl_additional_indices, target_ctrl_additional_indices = additional_corr_indices_pair
            template_ctrl_indices = np.array(template_ctrl_additional_indices, dtype=np.int32)
            target_ctrl_indices = np.array(target_ctrl_additional_indices, dtype=np.int32)
            len_additional_indices = len(template_ctrl_additional_indices)
            # corr_weights_additional = np.ones((len(template_ctrl_additional_indices), 1), dtype=np.float32)
            # corr_weights = np.array(corr_weights_additional, dtype=np.float32)
        
        if additional_corr_pair is not None:
            template_ctrl_additional_points, target_ctrl_additional_points = additional_corr_pair
            print(template_ctrl_additional_points.shape, target_ctrl_additional_points.shape)
            len_additional_points = len(template_ctrl_additional_points)
            # corr_weights_additional = np.ones((len(template_ctrl_additional_points), 1), dtype=np.float32)
            # corr_weights = np.array(corr_weights_additional, dtype=np.float32)
    
    len_append = len_additional_indices + len_additional_points
    if corr_weights is None:
        corr_weights = np.ones((len_append, 1), dtype=np.float32)
    else:
        corr_weights = np.concatenate([corr_weights, np.ones((len_append, 1), dtype=np.float32)])
            
    # Get correspondence points (convert to torch for MLS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    template_corr_points = torch.from_numpy(transformed_template_xyz[template_ctrl_indices]).to(device).float()
    target_corr_points = torch.from_numpy(transformed_target_xyz[target_ctrl_indices]).to(device).float()
    
    if additional_corr_pair is not None:
        template_additional_corr_points = torch.from_numpy(template_ctrl_additional_points).to(device).float()
        target_additional_corr_points = torch.from_numpy(target_ctrl_additional_points).to(device).float()
        
        template_corr_points = torch.cat([template_corr_points, template_additional_corr_points], dim=0)
        target_corr_points = torch.cat([target_corr_points, target_additional_corr_points], dim=0)
    
    # MLS parameters
    sigma = torch.tensor(sigma, device=device).float()
    
    # Initialize correspondence weights (could be optimized later)
    corr_weights = torch.from_numpy(corr_weights).to(device).float()
    
    # IMPORTANT: Transform vertices to template's XY space first
    # This ensures vertices and template correspondence points are in the same coordinate system
    verts_in_template_xy = apply_transformation_matrix_to_points(
        verts, 
        transformation_matrix_template
    )
    
    # Apply MLS transformation in XY plane
    verts_torch = torch.from_numpy(verts_in_template_xy).to(device).float()
    rots_dummy = torch.zeros((verts_in_template_xy.shape[0], 4), device=device).float()  # Dummy rotations
    
    try:
        transformed_verts, _ = mls_transform_pytorch_rot(
            verts_torch, 
            rots_dummy,
            template_corr_points, 
            target_corr_points, 
            # sigma=sigma,
            # corr_weights=corr_weights
        )
        
        transformed_gaussians, _ = mls_transform_pytorch_rot(
            torch.from_numpy(transformed_template_xyz).to(device).float(), 
            torch.zeros((transformed_template_xyz.shape[0], 4), device=device).float(),
            template_corr_points, 
            target_corr_points, 
            # sigma=sigma,
            # corr_weights=corr_weights
        )
        
        # Transform back from XY plane to target space
        inverse_target_matrix = np.linalg.inv(transformation_matrix_target)
        inverse_template_matrix = np.linalg.inv(transformation_matrix_template)
        # input_verts = apply_transformation_matrix_to_points(
        #     verts_torch.cpu().numpy(), 
        #     inverse_template_matrix
        # )
        
        transformed_verts = apply_transformation_matrix_to_points(
            transformed_verts.cpu().numpy(), 
            inverse_target_matrix
        )
        
        transformed_gaussians = apply_transformation_matrix_to_points(
            transformed_gaussians.cpu().numpy(), 
            inverse_target_matrix
        )
        
        transformed_template_corr_points = apply_transformation_matrix_to_points(
            template_corr_points.cpu().numpy(),
            inverse_template_matrix
        )
        transformed_target_corr_points = apply_transformation_matrix_to_points(
            target_corr_points.detach().cpu().numpy(),
            inverse_target_matrix
        )
        
        # input_verts = verts_torch.cpu().numpy()
        # transformed_template_corr_points = template_corr_points.cpu().numpy()
        # transformed_target_corr_points = target_corr_points.detach().cpu().numpy()
        
        # import open3d as o3d
        # input_verts_o3d = o3d.geometry.PointCloud()
        # input_verts_o3d.points = o3d.utility.Vector3dVector(input_verts)
        # input_verts_o3d.paint_uniform_color([0.8, 0.8, 0.8])
        # transformed_verts_o3d = o3d.geometry.PointCloud()
        # transformed_verts_o3d.points = o3d.utility.Vector3dVector(transformed_verts)
        # transformed_verts_o3d.paint_uniform_color([0.2, 0.8, 0.2])
        # transformed_template_corr_points_o3d = o3d.geometry.PointCloud()
        # transformed_template_corr_points_o3d.points = o3d.utility.Vector3dVector(transformed_template_corr_points)
        # transformed_template_corr_points_o3d.paint_uniform_color([0.82, 0.2, 0.8])
        # transformed_target_corr_points_o3d = o3d.geometry.PointCloud()
        # transformed_target_corr_points_o3d.points = o3d.utility.Vector3dVector(transformed_target_corr_points)
        # transformed_target_corr_points_o3d.paint_uniform_color([0.8, 0.8, 0.2])
        # o3d.visualization.draw_geometries([input_verts_o3d, transformed_verts_o3d, transformed_template_corr_points_o3d, transformed_target_corr_points_o3d])
        
        return transformed_verts, transformed_gaussians, (transformed_template_corr_points, transformed_target_corr_points)
        
    except Exception as e:
        print(f"MLS transformation failed: {e}")
        return verts, transformed_template_xyz  # Return original vertices as fallback

def get_transform_template_mesh_mls_corr_optim(
    source_segment: dict,
    target_segment: dict,
    verts: np.ndarray,
    num_corr: int = 64,
    sigma: float = 0.1,
    steps: int = 100,
    root_point: np.ndarray = None,
    image_size: int = 128,
    lr_rate: float = 1e-4,
    min_lr_rate: float = 1e-6,
    corr_weights: np.ndarray = None,
    additional_corr_indices_pair: tuple[np.ndarray, np.ndarray] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: ", device)
    source_denoised_indices = source_segment.get("denoised_indices")
    target_denoised_indices = target_segment.get("denoised_indices")
    
    if source_denoised_indices is not None:
        denoised_template_segment = apply_indices_to_gaussian_data(source_segment["original_data"], source_denoised_indices)
        denoised_template_labels = source_segment["labels"][source_denoised_indices]
    else:
        denoised_template_segment = source_segment["original_data"]
        denoised_template_labels = source_segment["labels"]
        denoised_indices, _, _ = mls_denoising(denoised_template_segment.xyz, 0.1)
        denoised_template_segment = apply_indices_to_gaussian_data(denoised_template_segment, denoised_indices)
        denoised_template_labels = denoised_template_labels[denoised_indices]
    if target_denoised_indices is not None:
        denoised_target_segment = apply_indices_to_gaussian_data(target_segment["original_data"], target_denoised_indices)
        denoised_target_labels = target_segment["labels"][target_denoised_indices]
    else:
        denoised_target_segment = target_segment["original_data"]
        denoised_target_labels = target_segment["labels"]
        denoised_indices, _, _ = mls_denoising(denoised_target_segment.xyz, 0.1)
        denoised_target_segment = apply_indices_to_gaussian_data(denoised_target_segment, denoised_indices)
        denoised_target_labels = denoised_target_labels[denoised_indices]
    
    source_segment_tip_point = source_segment["apex_point"]
    source_segment_base_point = source_segment["base_point"]
    target_segment_tip_point = target_segment["apex_point"]
    target_segment_base_point = target_segment["base_point"]
    if source_segment_tip_point is not None and source_segment_base_point is not None:
        transformation_matrix_template, _, _ = align_to_xy_plane_with_tips(
            denoised_template_segment.xyz, 
            source_segment_tip_point, 
            source_segment_base_point,
            root_point=root_point
        )
    else:
        transformation_matrix_template, _, _ = align_to_xy_plane(
            denoised_template_segment.xyz, 
            stem_labels=denoised_template_labels
        )

    if target_segment_tip_point is not None and target_segment_base_point is not None:
        transformation_matrix_target, _, _ = align_to_xy_plane_with_tips(
            denoised_target_segment.xyz, 
            target_segment_tip_point, 
            target_segment_base_point,
            root_point=root_point
        )
    else:
        transformation_matrix_target, _, _ = align_to_xy_plane(
        denoised_target_segment.xyz, 
        stem_labels=denoised_target_labels
    )
    
    transformed_template_xyz = apply_transformation_matrix_to_points(
        denoised_template_segment.xyz, 
        transformation_matrix_template
    )
    transformed_target_xyz = apply_transformation_matrix_to_points(
        denoised_target_segment.xyz, 
        transformation_matrix_target
    )
    transformation_matrix_template_R = transformation_matrix_template[:3, :3]
    transformation_matrix_target_R = transformation_matrix_target[:3, :3]
    q_delta_template = matrix_to_quaternion_wxyz(transformation_matrix_template_R)
    q_delta_target = matrix_to_quaternion_wxyz(transformation_matrix_target_R)
    transformed_template_rots = quat_multiply(denoised_template_segment.rot, q_delta_template)
    transformed_target_rots = quat_multiply(denoised_target_segment.rot, q_delta_target)
    
    transformed_template_shs = sh_rotate(denoised_template_segment.sh, transformation_matrix_template_R)
    transformed_target_shs = sh_rotate(denoised_target_segment.sh, transformation_matrix_target_R)
    
    verts_in_template_xy = apply_transformation_matrix_to_points(
        verts, 
        transformation_matrix_template
    )
    
    # Apply MLS transformation in XY plane
    verts_torch = torch.from_numpy(verts_in_template_xy).to(device).float()
    
    if num_corr > 0:
        # Find correspondence points using FPS
        template_ctrl_indices, target_ctrl_indices = fps_corr_search(
            transformed_template_xyz, 
            transformed_target_xyz, 
            num_corr
        )
    else:
        template_ctrl_indices = []
        target_ctrl_indices = []
        
    if additional_corr_indices_pair is not None:
        template_ctrl_additional_indices, target_ctrl_additional_indices = additional_corr_indices_pair
        print(template_ctrl_additional_indices.shape, target_ctrl_additional_indices.shape)
        if len(template_ctrl_indices) == 0:
            template_ctrl_indices = template_ctrl_additional_indices
            template_corr_points = torch.from_numpy(template_ctrl_indices).to(device).float()
        else:
            template_corr_points_np = transformed_template_xyz[template_ctrl_indices]
            template_corr_points_np = np.concatenate([template_corr_points_np, template_ctrl_additional_indices])
            template_corr_points = torch.from_numpy(template_corr_points_np).to(device).float()
        if len(target_ctrl_indices) == 0:
            target_ctrl_indices = target_ctrl_additional_indices
            target_corr_points = torch.from_numpy(target_ctrl_indices).to(device).float()
        else:
            target_corr_points_np = transformed_target_xyz[target_ctrl_indices]
            target_corr_points_np = np.concatenate([target_corr_points_np, target_ctrl_additional_indices])
            target_corr_points = torch.from_numpy(target_corr_points_np).to(device).float()
    else:
        template_corr_points = torch.from_numpy(transformed_template_xyz[template_ctrl_indices]).to(device).float()
        target_corr_points = torch.from_numpy(transformed_target_xyz[target_ctrl_indices]).to(device).float()
    
    # Get correspondence points (convert to torch for MLS)
    target_corr_points = torch.nn.Parameter(target_corr_points)
    
    camera_R = np.eye(3)
    camera_T = np.array([0, 0, 10]) 
    world_view_transform = np.transpose(getWorld2View2(camera_R, camera_T), (1, 0))

    tanfovx, tanfovy, full_proj_transform_np = get_full_proj_transform(
        world_view_transform,
        znear=0.001,
        zfar=100,
        FoVx=1,
        FoVy=1,
        orthographic=True
    )
    
    world_view_transform = torch.from_numpy(world_view_transform).to(device).float()
    full_proj_transform = torch.from_numpy(full_proj_transform_np).to(device).float()
    
    bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    scaling_modifier = 1.0
    
    sigma = torch.tensor(0.1, device="cuda", requires_grad=True).float()
    
    raster_settings = GaussianRasterizationSettings(
        image_height=image_size,
        image_width=image_size,
        tanfovx=float(tanfovx),
        tanfovy=float(tanfovy),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=3,
        campos=torch.from_numpy(camera_T).cuda().float(),
        prefiltered=False,
        debug=True,
        antialiasing=False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    optimizer = torch.optim.Adam([
        {'params': [sigma], 'lr': lr_rate},
        {'params': [target_corr_points], 'lr': lr_rate},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=min_lr_rate)

    xyz = torch.from_numpy(transformed_template_xyz).to(device).float()
    xyz_tar = torch.from_numpy(transformed_target_xyz).to(device).float()
    rot = torch.from_numpy(transformed_template_rots).to(device).float()
    rot_tar = torch.from_numpy(transformed_target_rots).to(device).float()
    
    shs = torch.from_numpy(transformed_template_shs).to(device).float()
    opacity = torch.from_numpy(denoised_template_segment.opacity).to(device).float()
    scale = torch.from_numpy(denoised_template_segment.scale).to(device).float()
    shs_tar = torch.from_numpy(transformed_target_shs).to(device).float()
    opacity_tar = torch.from_numpy(denoised_target_segment.opacity).to(device).float()
    scale_tar = torch.from_numpy(denoised_target_segment.scale).to(device).float()
    print("START OPTIMIZATION ## ")

    angle_xs = np.linspace(0, 180, steps // 10) # 20
    angle_ys = np.linspace(0, 180, steps // 20) # 10

    composed_angles = [(ax, ay) for ax in angle_xs for ay in angle_ys] # 200

    for step, (angle_x, angle_y) in enumerate(composed_angles):
        
        optimizer.zero_grad()
        Rx = R.from_euler('x', angle_x, degrees=True).as_matrix()
        Ry = R.from_euler('y', angle_y, degrees=True).as_matrix()
        R_view_np = Rx @ Ry
        R_view = torch.from_numpy(R_view_np).float().cuda()
        T_view = torch.eye(4).float().cuda()
        T_view[:3, :3] = R_view
        
        xyz_current = apply_transformation_matrix_to_points_torch(xyz, T_view)
        xyz_tar_current = apply_transformation_matrix_to_points_torch(xyz_tar, T_view)
        
        q_delta = matrix_to_quaternion_wxyz(R_view)           # (4,)
        rot_current = quat_multiply(rot, q_delta)
        rot_tar_current = quat_multiply(rot_tar, q_delta)
        
        shs_current = sh_rotate(shs, R_view)
        shs_tar_current = sh_rotate(shs_tar, R_view)
        
        src_corr_points_current = apply_transformation_matrix_to_points_torch(template_corr_points, T_view)
        tar_corr_points_current = apply_transformation_matrix_to_points_torch(target_corr_points, T_view)
            
        rendered_image_tar, _, depth_image_tar = rasterizer(
                means3D=xyz_tar_current,
                means2D=torch.zeros_like(xyz_tar_current).cuda().float(),
                shs=shs_tar_current.reshape(-1, 16, 3),
                colors_precomp=None,
                opacities=opacity_tar,
                scales=scale_tar,
                rotations=rot_tar_current,
                cov3D_precomp=None
        )
        
        xyz_transformed, rot_transformed = mls_transform_pytorch_rot(
                xyz_current,
                rot_current,
                src_corr_points_current,
                tar_corr_points_current,
            )
        
        cd = chamfer_distance(xyz_transformed.unsqueeze(0), xyz_tar_current.unsqueeze(0))
        rendered_image_transformed, _, depth_image_transformed = rasterizer(
            means3D=xyz_transformed,
            means2D=torch.zeros_like(xyz_transformed).cuda().float(),
            shs=shs_current.reshape(-1, 16, 3),
            colors_precomp=None,
            opacities=opacity,
            scales=scale,
            rotations=rot_transformed,
        )
      
        mask_loss = torch.nn.MSELoss()(depth_image_transformed, depth_image_tar)
   
        lambda_mask = 0.3
        loss = lambda_mask * mask_loss + (1 - lambda_mask) * cd[0] 
        if step % 50 == 0:
            print(f"Step {step} | Loss: {loss.item()}, Mask Loss: {mask_loss.item()}, Chamfer: {cd[0].item()}, LR: {scheduler.get_last_lr()[0]}")
        
        loss.backward()
        optimizer.step()
        scheduler.step()
    print("END OPTIMIZATION ## ")
    
    transformed_verts, _ = mls_transform_pytorch_rot(
        verts_torch,
        torch.zeros((verts_torch.shape[0], 4), device=device).float(),
        template_corr_points,
        target_corr_points,
    )
    
    transformed_gaussians, _ = mls_transform_pytorch_rot(
        torch.from_numpy(transformed_template_xyz).to(device).float(),
        torch.zeros((transformed_template_xyz.shape[0], 4), device=device).float(),
        template_corr_points,
        target_corr_points,
    )
        
    # final_transformed_verts = xyz_transformed.detach().cpu().numpy()
    inverse_target_matrix = np.linalg.inv(transformation_matrix_target)
    inverse_template_matrix = np.linalg.inv(transformation_matrix_template)
    transformed_verts = apply_transformation_matrix_to_points(
        transformed_verts, 
        inverse_target_matrix
    )
    
    transformed_gaussians = apply_transformation_matrix_to_points(
        transformed_gaussians, 
        inverse_target_matrix
    )
    
    template_corr_points = template_corr_points.cpu().numpy()
    target_corr_points = target_corr_points.detach().cpu().numpy()
    
    transformed_template_corr_points = apply_transformation_matrix_to_points(
        template_corr_points,
        inverse_template_matrix
    )
    transformed_target_corr_points = apply_transformation_matrix_to_points(
        target_corr_points,
        inverse_target_matrix
    )
    
    print("FINISH OPTIMIZATION ## ")
    
    return transformed_verts, transformed_gaussians, (transformed_template_corr_points, transformed_target_corr_points)