import numpy as np
import torch
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import fpsample
import trimesh
from scipy import ndimage
from heapq import heappush, heappop
import os
# Local imports
from pca_utils import align_to_xy_plane_with_tips
from gaussian_utils import GaussianData, quaternion_wxyz_to_matrix, matrix_to_quaternion_wxyz, quat_multiply, sh_rotate, compute_cov3d, save_gaussian_data_as_ply
from diff_gaussian_rasterization import my_rasterizer

def compute_sh_color_with_direction(shs, positions, camera_pos, sh_degree=3):

    directions = positions - camera_pos.unsqueeze(0)  # (N, 3)
    directions = torch.nn.functional.normalize(directions, dim=1)

    SH_C0 = 0.28209479177387814
    SH_C1 = 0.4886025119029199
    SH_C2_0 = 1.0925484305920792
    SH_C2_1 = -1.0925484305920792
    SH_C2_2 = 0.31539156525252005
    SH_C2_3 = -1.0925484305920792
    SH_C2_4 = 0.5462742152960396
    SH_C3_0 = -0.5900435899266435
    SH_C3_1 = 2.890611442640554
    SH_C3_2 = -0.4570457994644658
    SH_C3_3 = 0.3731763325901154
    SH_C3_4 = -0.4570457994644658
    SH_C3_5 = 1.445305721320277
    SH_C3_6 = -0.5900435899266435

    colors = SH_C0 * shs[:, :3]

    sh_channels = shs.shape[1]

    if sh_channels > 3 and sh_degree >= 1:
        if sh_channels >= 12:
            x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
            colors += (-SH_C1 * y.unsqueeze(1) * shs[:, 3:6] +
                       SH_C1 * z.unsqueeze(1) * shs[:, 6:9] -
                       SH_C1 * x.unsqueeze(1) * shs[:, 9:12])

            if sh_channels >= 27 and sh_degree >= 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                colors += (SH_C2_0 * xy.unsqueeze(1) * shs[:, 12:15] +
                           SH_C2_1 * yz.unsqueeze(1) * shs[:, 15:18] +
                           SH_C2_2 * (2.0 * zz - xx - yy).unsqueeze(1) * shs[:, 18:21] +
                           SH_C2_3 * xz.unsqueeze(1) * shs[:, 21:24] +
                           SH_C2_4 * (xx - yy).unsqueeze(1) * shs[:, 24:27])

                if sh_channels >= 48 and sh_degree >= 3:
                    colors += (SH_C3_0 * (y * (3.0 * xx - yy)).unsqueeze(1) * shs[:, 27:30] +
                               SH_C3_1 * (xy * z).unsqueeze(1) * shs[:, 30:33] +
                               SH_C3_2 * (y * (4.0 * zz - xx - yy)).unsqueeze(1) * shs[:, 33:36] +
                               SH_C3_3 * (z * (2.0 * zz - 3.0 * xx - 3.0 * yy)).unsqueeze(1) * shs[:, 36:39] +
                               SH_C3_4 * (x * (4.0 * zz - xx - yy)).unsqueeze(1) * shs[:, 39:42] +
                               SH_C3_5 * (z * (xx - yy)).unsqueeze(1) * shs[:, 42:45] +
                               SH_C3_6 * (x * (xx - 3.0 * yy)).unsqueeze(1) * shs[:, 45:48])

    colors += 0.5
    return torch.clamp(colors, 0.0, 1.0)

def fps_sampling_2d(points, num_samples):
  
    if len(points) <= num_samples:
        return np.arange(len(points))
    
    points = np.array(points)
    n_points = len(points)
    
    center = np.mean(points, axis=0)
    distances_to_center = np.linalg.norm(points - center, axis=1)
    first_idx = np.argmin(distances_to_center)
    
    selected_indices = [first_idx]
    remaining_indices = list(range(n_points))
    remaining_indices.remove(first_idx)
    
    for _ in range(num_samples - 1):
        if not remaining_indices:
            break
        
        max_min_distance = -1
        farthest_idx = -1
        
        for candidate_idx in remaining_indices:
            candidate_point = points[candidate_idx]
            
            min_distance_to_selected = float('inf')
            for selected_idx in selected_indices:
                selected_point = points[selected_idx]
                distance = np.linalg.norm(candidate_point - selected_point)
                min_distance_to_selected = min(min_distance_to_selected, distance)
            
            if min_distance_to_selected > max_min_distance:
                max_min_distance = min_distance_to_selected
                farthest_idx = candidate_idx
        
        if farthest_idx != -1:
            selected_indices.append(farthest_idx)
            remaining_indices.remove(farthest_idx)
    
    return np.array(selected_indices)

def gaussian_poisson_resample_intensity(
    g, tip_point, base_point,
    *,
    target_spacing_world=None,
    target_count=None,

    image_size=1024,
    view_side="front",
    intensity_threshold=0.10,   
    black_background=True,      

    poisson_k=30,
    seed=42,

    invalid_if_zero=True,
):

    rng = np.random.default_rng(seed)

    pts_t = torch.from_numpy(g.xyz).float().cuda()
    scl_t = torch.from_numpy(g.scale).float().cuda()
    rot_t = torch.from_numpy(g.rot).float().cuda()
    opa_np = getattr(g, "opacity", None)
    if opa_np is None:
        opa_t = torch.ones((len(g), 1), dtype=torch.float32, device="cuda")
    else:
        opa_t = torch.from_numpy(opa_np).float().cuda()
    shs_np = getattr(g, "sh", None)
    shs_t = None if shs_np is None else torch.from_numpy(shs_np).float().cuda()

    rendered_img, T, x_range, y_range, _, _, depth = render_to_pca(
        points=pts_t, scales=scl_t, rots=rot_t, opacities=opa_t, shs=shs_t,
        tip_point=tip_point, base_point=base_point,
        image_size=image_size, view_side=view_side,
        uv_rendering=False, mask=False, black_background=black_background
    )
    T_inv = np.linalg.inv(T)

    img = np.asarray(rendered_img)
    if img.ndim == 2:               
        intensity = img.astype(np.float32)
    else:                            
        if img.dtype == np.uint8 or img.max() > 1.001:
            img = img.astype(np.float32) / 255.0
        intensity = (0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]).astype(np.float32)

    depth = np.asarray(depth).astype(np.float32)
    H, W  = depth.shape


    valid_d = np.isfinite(depth)
    if invalid_if_zero:
        valid_d &= (depth != 0)

    base_mask = (intensity > float(intensity_threshold)) & valid_d
    inner_mask = np.zeros_like(base_mask, dtype=bool)
    inner_mask[:-1, :-1] = (base_mask[:-1, :-1] & base_mask[1:, :-1] &
                            base_mask[:-1, 1:] & base_mask[1:, 1:])

    x_min, x_max = float(x_range[0]), float(x_range[1])
    y_min, y_max = float(y_range[0]), float(y_range[1])
    Wx = max(W - 1, 1)
    Hy = max(H - 1, 1)
    dx = (x_max - x_min) / Wx
    dy = (y_max - y_min) / Hy

    def pxpy_to_xy(px, py):
        x = x_min + (px / Wx) * (x_max - x_min)
        y = y_min + (py / Hy) * (y_max - y_min)
        return x, y

    def xy_to_pxpy(x, y):
        px = (x - x_min) / (x_max - x_min) * Wx
        py = (y - y_min) / (y_max - y_min) * Hy
        return px, py

    if target_spacing_world is not None:
        r = float(target_spacing_world)
    else:
        area_valid = float(np.count_nonzero(inner_mask)) * dx * dy
        r = np.sqrt(max(area_valid, 1e-12) / (float(target_count) * 2.0 * np.sqrt(3.0)))

    def depth_bilinear_at(x, y):
        px, py = xy_to_pxpy(x, y)
        if not (0 <= px < W - 1 and 0 <= py < H - 1):
            return None
        j0 = int(np.floor(px)); j1 = j0 + 1
        i0 = int(np.floor(py)); i1 = i0 + 1
        if not (inner_mask[i0, j0] and inner_mask[i0, j1] and inner_mask[i1, j0] and inner_mask[i1, j1]):
            return None
        dxp = px - j0; dyp = py - i0
        z00 = depth[i0, j0]; z01 = depth[i0, j1]
        z10 = depth[i1, j0]; z11 = depth[i1, j1]
        z0 = z00 * (1 - dxp) + z01 * dxp
        z1 = z10 * (1 - dxp) + z11 * dxp
        return float(z0 * (1 - dyp) + z1 * dyp)

    a = r / np.sqrt(2.0)
    grid_w = int(np.ceil((x_max - x_min) / a))
    grid_h = int(np.ceil((y_max - y_min) / a))
    grid   = -np.ones((grid_h, grid_w), dtype=np.int32)

    def to_grid(x, y):
        gx = int((x - x_min) / a)
        gy = int((y - y_min) / a)
        return gx, gy

    def in_bbox(x, y):
        return (x_min <= x < x_max) and (y_min <= y < y_max)

    def is_valid_xy(x, y):
        px, py = xy_to_pxpy(x, y)
        if not (0 <= px < W - 1 and 0 <= py < H - 1):
            return False
        return inner_mask[int(np.floor(py)), int(np.floor(px))]

    def far_enough(x, y, samples_xy):
        gx, gy = to_grid(x, y)
        x0 = max(gx - 2, 0); x1 = min(gx + 3, grid_w)
        y0 = max(gy - 2, 0); y1 = min(gy + 3, grid_h)
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                idx = grid[yy, xx]
                if idx >= 0:
                    sx, sy = samples_xy[idx]
                    if (sx - x)**2 + (sy - y)**2 < r*r:
                        return False
        return True

    samples_xy, samples_pxpy, active = [], [], []
    inner_inds = np.argwhere(inner_mask)
    i0, j0 = inner_inds[rng.integers(0, len(inner_inds))]
    x0, y0 = pxpy_to_xy(j0, i0)
    samples_xy.append((x0, y0))
    samples_pxpy.append((j0, i0))
    gx0, gy0 = to_grid(x0, y0)
    grid[gy0, gx0] = 0
    active.append(0)

    while active:
        idx = int(rng.choice(active))
        cx, cy = samples_xy[idx]
        accepted = False
        for _ in range(poisson_k):
            rad = rng.uniform(r, 2*r)
            ang = rng.uniform(0.0, 2.0*np.pi)
            nx = cx + rad * np.cos(ang)
            ny = cy + rad * np.sin(ang)
            if not in_bbox(nx, ny):         continue
            if not is_valid_xy(nx, ny):     continue
            if not far_enough(nx, ny, samples_xy): continue
            samples_xy.append((nx, ny))
            px, py = xy_to_pxpy(nx, ny)
            samples_pxpy.append((px, py))
            ggx, ggy = to_grid(nx, ny)
            grid[ggy, ggx] = len(samples_xy) - 1
            active.append(len(samples_xy) - 1)
            accepted = True
            break
        if not accepted:
            active.remove(idx)

    samples_xy  = np.asarray(samples_xy, dtype=np.float32)
    samples_pxpy= np.asarray(samples_pxpy, dtype=np.float32)

    zs, keep = [], []
    for k, (x, y) in enumerate(samples_xy):
        z = depth_bilinear_at(x, y)
        if z is None or not np.isfinite(z):
            continue
        zs.append(z); keep.append(k)

    xy_kept    = samples_xy[keep]
    pxpy_kept  = samples_pxpy[keep]
    uvz_aligned= np.concatenate([xy_kept, np.asarray(zs, dtype=np.float32)[:,None]], axis=1)

    eps = 1e-12
    ys, xs = np.where(inner_mask)
    jmin, jmax = xs.min(), xs.max()
    imin, imax = ys.min(), ys.max()

    x_eff_min = x_min + (jmin / Wx) * (x_max - x_min)
    x_eff_max = x_min + (jmax / Wx) * (x_max - x_min)
    y_eff_min = y_min + (imin / Hy) * (y_max - y_min)
    y_eff_max = y_min + (imax / Hy) * (y_max - y_min)

    samp_cx = 0.5 * (x_eff_min + x_eff_max)
    samp_cy = 0.5 * (y_eff_min + y_eff_max)
    samp_ex = max(x_eff_max - x_eff_min, eps)
    samp_ey = max(y_eff_max - y_eff_min, eps)

    xyz0  = np.asarray(g.xyz, dtype=np.float32)
    ones0 = np.ones((len(xyz0), 1), dtype=np.float32)
    hom0  = np.concatenate([xyz0, ones0], axis=1)
    aligned0 = (T @ hom0.T).T[:, :3]
    orig_xy  = aligned0[:, :2]
    orig_min = orig_xy.min(axis=0)
    orig_max = orig_xy.max(axis=0)
    orig_cx  = 0.5 * (orig_min[0] + orig_max[0])
    orig_cy  = 0.5 * (orig_min[1] + orig_max[1])
    orig_ex  = max(orig_max[0] - orig_min[0], eps)
    orig_ey  = max(orig_max[1] - orig_min[1], eps)

    sx = orig_ex / samp_ex
    sy = orig_ey / samp_ey
    uvz_aligned[:, 0] = (uvz_aligned[:, 0] - samp_cx) * sx + orig_cx
    uvz_aligned[:, 1] = (uvz_aligned[:, 1] - samp_cy) * sy + orig_cy
    r = float(r * np.sqrt(abs(sx * sy)))

    ones = np.ones((len(uvz_aligned), 1), dtype=np.float32)
    hom  = np.concatenate([uvz_aligned, ones], axis=1)
    world_pts = (T_inv @ hom.T).T[:, :3].astype(np.float32)

    return world_pts, float(r), uvz_aligned, pxpy_kept, inner_mask

def render_to_pca(points, 
                  scales, 
                  rots, 
                  opacities, 
                  shs, 
                labels=None,
                tip_point=None,
                base_point=None,
                root_point=None,
                image_size=1024,
                view_side="front", 
                uv_rendering=False, 
                rotation_angle=None,
                 mask=False, 
                 black_background=False
                 ):
    """Render Gaussian point cloud using PCA method (based on test_ps_simple.py implementation)"""
    if tip_point is not None and base_point is not None:
        transformation_matrix, aligned_points, pca_info = align_to_xy_plane_with_tips(
            points.cpu().numpy(), tip_point, base_point, root_point=root_point
        )
    else:
        raise Exception("Stop here")
    transformation_matrix = torch.from_numpy(transformation_matrix).float().cuda()

    R_matrix = transformation_matrix[:3, :3]
    # Apply R_matrix to root_point [1, 3]
    if root_point is not None:
        root_point_h = torch.ones((1, 4), dtype=torch.float32).cuda()
        root_point_h[0, :3] = torch.from_numpy(root_point).float().cuda()
        transformed_root_h = (transformation_matrix @ root_point_h.T).T
        transformed_root = transformed_root_h[0, :3].cpu().numpy()
    
    q_delta = matrix_to_quaternion_wxyz(R_matrix)
    aligned_rots = quat_multiply(rots, q_delta)
    aligned_shs = sh_rotate(shs, R_matrix)

    aligned_cov = compute_cov3d(scales, aligned_rots)
    aligned_points = torch.from_numpy(aligned_points).cuda().float()
    
    points_2d = aligned_points[:, :2]
    covariances_2d = aligned_cov[:, :2, :2]  # Extract 2D covariances
    
    # Prepare colors with proper SH calculation
    if aligned_shs is not None and aligned_shs.shape[1] >= 3:
        # Calculate camera position (above the center of the point cloud)
        center = aligned_points.mean(dim=0)
        camera_pos = center.clone()
        if view_side == "front":
            if root_point is not None:
                camera_pos[2] = center[2] - transformed_root[2]   # Position camera below the center for back view
            else:
                camera_pos[2] = center[2] - 10   # Default offset if root_point is not provided
        else:
            if root_point is not None:
                camera_pos[2] = center[2] + transformed_root[2]   # Position camera above the center
            else:
                camera_pos[2] = center[2] + 10   # Default offset if root_point is not provided

        rgb_colors = compute_sh_color_with_direction(aligned_shs, aligned_points, camera_pos, sh_degree=3)
        z_values = aligned_points[:, 2]
        colors_array = torch.cat([z_values.reshape(-1, 1), rgb_colors], dim=1)
    else:
        # Depth-based rendering
        z_values = aligned_points[:, 2]
        colors_array = z_values.reshape(-1, 1)
    
    # Back view rendering: completely following test_ps_simple.py implementation
    if view_side == "front":
        aligned_points[:, 2] = -aligned_points[:, 2]
        z_values = aligned_points[:, 2]
        if colors_array.shape[1] == 1:
            colors_array = z_values.reshape(-1, 1)
        else:
            colors_array[:, 0] = z_values

    depths_np = aligned_points[:, 2]
    output_image_tensor, depth_map, x_range, y_range = my_rasterizer(
        points_2d,
        depths_np,
        covariances_2d,
        rgb_colors,
        opacities,
        image_size,
        image_size,
        uv_rendering=True,
        mask=mask,
        black_background=black_background,
    )
    
    rendered_image_uv = output_image_tensor.cpu().numpy()

    depth_map_np = depth_map.cpu().numpy()
    
    if view_side == "front":
        aligned_points[:, 2] = -aligned_points[:, 2]
        depth_map_np = -depth_map_np

    return rendered_image_uv, transformation_matrix.cpu().numpy(), x_range, y_range, aligned_points.cpu().numpy(), pca_info, depth_map_np

def create_aligned_uv_mapping(mesh_vertices, transformation_matrix, 
                              x_range, y_range, margin=0.01):
    """Create aligned UV mapping, ensuring consistency with PCA transformation"""
    # print("=== Creating Aligned UV Mapping ===")
    # Apply same PCA transformation to mesh vertices
    mesh_vertices_homo = np.column_stack([mesh_vertices, np.ones(len(mesh_vertices))])
    transformed_mesh = (transformation_matrix @ mesh_vertices_homo.T).T
    mesh_2d = transformed_mesh[:, :2]  # Transformed 2D coordinates

    # Normalize to [0,1] UV space
    u = (mesh_2d[:, 0] - x_range[0]) / (x_range[1] - x_range[0])
    v = (mesh_2d[:, 1] - y_range[0]) / (y_range[1] - y_range[0])
    u = u * (1 - 2 * margin) + margin
    v = v * (1 - 2 * margin) + margin
    # Clip to [0,1]
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    v = 1.0 - v
    uv_coords = np.column_stack([u, v])
    
    return uv_coords, mesh_2d

def create_double_sided_texture(front_texture, back_texture):
    """Create double-sided texture by combining front and back textures"""
    
    # print("=== Creating Double-Sided Texture ===")
    
    h, w = front_texture.shape[:2]
    
    # Create double-width texture
    if len(front_texture.shape) == 3:
        double_texture = np.zeros((h, w * 2, front_texture.shape[2]), dtype=front_texture.dtype)
        double_texture[:, :w] = front_texture  # Left half: front
        double_texture[:, w:] = back_texture   # Right half: back
    else:
        double_texture = np.zeros((h, w * 2), dtype=front_texture.dtype)
        double_texture[:, :w] = front_texture
        double_texture[:, w:] = back_texture
    
    # print(f"Double-sided texture size: {double_texture.shape}")
    
    return double_texture

def compute_vertex_normals(verts: np.ndarray, faces: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = verts.astype(np.float64, copy=False)
    f = faces.astype(np.int64,  copy=False)
    n = np.zeros_like(v, dtype=np.float64)

    v0, v1, v2 = v[f[:,0]], v[f[:,1]], v[f[:,2]]
    fn = np.cross(v1 - v0, v2 - v0)  

    for k in range(3):
        np.add.at(n, f[:,k], fn)

    lens = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, np.maximum(lens, eps))
    return n.astype(np.float32)

def create_double_sided_uv_mapping(mesh_vertices, triangles, root_point, transformation_matrix, x_range, y_range):
    uv_coords, mesh_2d = create_aligned_uv_mapping(
        mesh_vertices, transformation_matrix, x_range, y_range
    )

    V = mesh_vertices.shape[0]
    F = triangles.shape[0]

    verts_front = mesh_vertices.copy() 
    verts_back  = mesh_vertices.copy() 
    new_vertices = np.vstack([verts_front, verts_back])                # (2V, 3)

    uv_front = uv_coords.copy()
    uv_front[:, 0] = uv_front[:, 0] * 0.5                              # [0, 0.5]
    uv_back  = uv_coords.copy()
    uv_back[:, 0] = 0.5 + uv_back[:, 0] * 0.5                          # [0.5, 1.0]
    vertex_uv_coords = np.vstack([uv_front, uv_back])                  # (2V, 2)

    faces_front = triangles.astype(np.int32)                           # (F, 3)
    faces_back  = np.column_stack([
        triangles[:, 0] + V, triangles[:, 2] + V, triangles[:, 1] + V 
    ]).astype(np.int32)

    front_normals = compute_vertex_normals(mesh_vertices, triangles)  # (V,3)
    back_normals  = -front_normals

    # Use root point to determine front and back normals direction
    to_root = root_point - np.mean(mesh_vertices, axis=0)
    avg_normal = np.mean(front_normals, axis=0)

    print(np.dot(to_root, avg_normal))

    if np.dot(to_root, avg_normal) > 0:
        print("Flipping normals to ensure correct orientation.")
        # swap normals
        temp = front_normals.copy()
        front_normals = back_normals
        back_normals  = temp
        # swap front and back faces
        temp = faces_front.copy()
        faces_front = faces_back - 1024
        faces_back  = temp + 1024

    
    new_faces = np.vstack([faces_front, faces_back])                   # (2F, 3)
    new_normals   = np.vstack([front_normals, back_normals])          # (2V,3)

    return vertex_uv_coords, new_vertices, new_faces, new_normals

def resample_points_from_gaussian(gaussian: GaussianData):
    points = gaussian.xyz
    scales = gaussian.scale
    rots = gaussian.rot
    opacities = gaussian.opacity
    
    distances = cdist(points, points)
    nearest_distances = []
    for i in range(len(points)):
        point_distances = distances[i]
        point_distances = point_distances[point_distances > 1e-8]  
        if len(point_distances) > 0:
            nearest_distances.append(np.min(point_distances))
        else:
            nearest_distances.append(0.0)
    
    nearest_distances = np.array(nearest_distances)
    # neighbor_counts = np.sum(distances < 0.05, axis=1)
    distance_99th = np.percentile(nearest_distances, 99)
    floating_threshold = distance_99th * 1.5
    
    
    has_neighbor = np.sum(distances < floating_threshold, axis=1) > 2
    
    valid_mask = has_neighbor & (opacities[:, 0] >= 0.1)
    valid_indices = np.where(valid_mask)[0]
    
    sampled_points = []
    axes = np.eye(3)  
    
    for i in valid_indices:
        center = points[i]
        current_scales = scales[i]
        rotation = rots[i]
        R = quaternion_wxyz_to_matrix(rotation)
        
        sampled_points.append(center)
        
        for axis_idx in range(3):
            axis_world = R @ axes[axis_idx]
            scale_val = current_scales[axis_idx]
            
            sampled_points.extend([
                center + scale_val * axis_world,
                center - scale_val * axis_world
            ])
    
    sampled_points = np.array(sampled_points)

    target_points_num = len(sampled_points) // 6
    if len(sampled_points) > target_points_num:
        idx = fpsample.bucket_fps_kdline_sampling(sampled_points, target_points_num, h=7)
        sampled_points = sampled_points[idx]
    
    return sampled_points

def detect_edges_from_rendered_image(rendered_image: np.ndarray,
                                     method: str = 'color_boundary',
                                     intensity_threshold: float = 0.1):
    if rendered_image.shape[2] >= 3:
        intensity = np.mean(rendered_image[:, :, :3], axis=2)
    else:
        intensity = rendered_image[:, :, 0]
    
    if method == 'color_boundary':
        intensity_smooth = ndimage.gaussian_filter(intensity, sigma=1.0)
        foreground_mask = intensity_smooth > intensity_threshold
        
        structure_close = np.ones((5, 5))
        foreground_closed = ndimage.binary_closing(foreground_mask, structure=structure_close)
        
        structure_open = np.ones((3, 3))
        foreground_clean = ndimage.binary_opening(foreground_closed, structure=structure_open)
        
        structure_erode = np.ones((5, 5))
        eroded_foreground = ndimage.binary_erosion(foreground_clean, structure=structure_erode)
        edges_binary = foreground_clean & np.logical_not(eroded_foreground)
        
        if np.sum(edges_binary) > 0:
            structure_smooth = np.ones((3, 3))
            edges_dilated = ndimage.binary_dilation(edges_binary, structure=structure_smooth)
            edges_binary = ndimage.binary_erosion(edges_dilated, structure=structure_smooth)
        
    elif method == 'simple_gradient':
        grad_x = np.abs(np.diff(intensity, axis=1, prepend=intensity[:, :1]))
        grad_y = np.abs(np.diff(intensity, axis=0, prepend=intensity[:1, :]))
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        if edge_magnitude.max() > 0:
            edge_magnitude = edge_magnitude / edge_magnitude.max()
        
        edges_binary = edge_magnitude > intensity_threshold
        
    elif method == 'neighbor_diff':
        h, w = intensity.shape
        edges_binary = np.zeros((h, w), dtype=bool)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = intensity[i, j]
                neighbors = [
                    intensity[i-1, j], intensity[i+1, j],  
                    intensity[i, j-1], intensity[i, j+1],  
                    intensity[i-1, j-1], intensity[i-1, j+1],  
                    intensity[i+1, j-1], intensity[i+1, j+1]
                ]
                
                max_diff = max([abs(center - neighbor) for neighbor in neighbors])
                if max_diff > intensity_threshold:
                    edges_binary[i, j] = True
    
    else:
        raise ValueError(f"Unknown edge detection method: {method}")
    
    return edges_binary

def project_2d_coords_to_3d(coordinates_2d, 
                                depth_map, 
                                x_range, 
                                y_range, 
                                image_size
                                ):
    ndc_coords = coordinates_2d / (0.5 * image_size) - 1
    x_world = (ndc_coords[:, 0] + 1) * 0.5 * (x_range[1] - x_range[0]) + x_range[0]
    y_world = (ndc_coords[:, 1] + 1) * 0.5 * (y_range[1] - y_range[0]) + y_range[0]
    
    coordinates_2d_rounded = np.array(coordinates_2d, dtype=int)
    z_values = depth_map[coordinates_2d_rounded[:, 1], coordinates_2d_rounded[:, 0]]
    coordinates_3d = np.column_stack([x_world, y_world, z_values])
    
    # Inverse transform the coordinates
    
    return coordinates_3d

def project_2d_edges_to_3d(edges_binary: np.ndarray,
                           aligned_points: np.ndarray,
                           x_range: tuple,
                           y_range: tuple):
    edge_pixels_y, edge_pixels_x = np.where(edges_binary)
    
    if len(edge_pixels_x) == 0:
        return np.array([])
    
    height, width = edges_binary.shape
    x_world = x_range[0] + (edge_pixels_x / (width - 1)) * (x_range[1] - x_range[0])
    y_world = y_range[0] + (edge_pixels_y / (height - 1)) * (y_range[1] - y_range[0])
    
    edge_3d_indices = []
    
    if hasattr(aligned_points, 'cpu'):
        aligned_points_np = aligned_points.cpu().numpy()
    else:
        aligned_points_np = aligned_points
    
    aligned_points_2d = aligned_points_np[:, :2]  
    
    for i in range(len(x_world)):
        target_2d = np.array([x_world[i], y_world[i]])
        distances = np.linalg.norm(aligned_points_2d - target_2d, axis=1)
        closest_idx = np.argmin(distances)
        
        edge_3d_indices.append(closest_idx)
    
    return np.unique(edge_3d_indices)

def world_to_pixel(point_2d: np.ndarray,
                   x_range: tuple,
                   y_range: tuple,
                   image_size: int):
    x_pixel = int((point_2d[0] - x_range[0]) / (x_range[1] - x_range[0]) * (image_size - 1))
    y_pixel = int((point_2d[1] - y_range[0]) / (y_range[1] - y_range[0]) * (image_size - 1))
    return np.clip([x_pixel, y_pixel], 0, image_size - 1)

def pixel_to_world(pixel_coord: np.ndarray,
                   x_range: tuple,
                   y_range: tuple,
                   image_size: int):
    x_world = x_range[0] + (pixel_coord[0] / (image_size - 1)) * (x_range[1] - x_range[0])
    y_world = y_range[0] + (pixel_coord[1] / (image_size - 1)) * (y_range[1] - y_range[0])
    return np.array([x_world, y_world])

def compute_edge_path_lengths(edges_binary, apex_pixel, stem_pixel):
    """
    Compute the lengths of left and right edge paths from apex to stem.
    
    Args:
        edges_binary: Binary edge map (H, W) with 1 for edges, 0 for background
        apex_pixel: [x, y] coordinates of apex point
        stem_pixel: [x, y] coordinates of stem point
        
    Returns:
        tuple: (left_path_length, right_path_length, path_info)
    """
    def get_neighbors_8connected(y, x, h, w):
        """Get 8-connected neighbors with distances"""
        neighbors = []
        # 8-connected neighborhood with distances
        directions = [
            (-1, -1, np.sqrt(2)), (-1, 0, 1.0), (-1, 1, np.sqrt(2)),  # top row
            (0, -1, 1.0),                        (0, 1, 1.0),           # middle row
            (1, -1, np.sqrt(2)), (1, 0, 1.0),   (1, 1, np.sqrt(2))     # bottom row
        ]
        
        for dy, dx, dist in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                neighbors.append((ny, nx, dist))
        return neighbors
    
    def find_shortest_path_on_edges(start_pixel, end_pixel, edges_binary):
        """Find shortest path between two points staying on edge pixels"""
        h, w = edges_binary.shape
        start_y, start_x = start_pixel[1], start_pixel[0]  # Convert to array indices
        end_y, end_x = end_pixel[1], end_pixel[0]
        
        # Check if start and end points are on edges
        if not (edges_binary[start_y, start_x] and edges_binary[end_y, end_x]):
            return None, float('inf')
        
        # Dijkstra's algorithm on edge pixels only
        distances = np.full((h, w), float('inf'))
        distances[start_y, start_x] = 0.0
        visited = np.zeros((h, w), dtype=bool)
        parent = {}
        
        # Priority queue: (distance, y, x)
        pq = [(0.0, start_y, start_x)]
        
        while pq:
            current_dist, y, x = heappop(pq)
            
            if visited[y, x]:
                continue
                
            visited[y, x] = True
            
            # Found target
            if y == end_y and x == end_x:
                break
                
            # Explore neighbors
            for ny, nx, edge_dist in get_neighbors_8connected(y, x, h, w):
                if not edges_binary[ny, nx] or visited[ny, nx]:
                    continue
                    
                new_dist = current_dist + edge_dist
                if new_dist < distances[ny, nx]:
                    distances[ny, nx] = new_dist
                    parent[(ny, nx)] = (y, x)
                    heappush(pq, (new_dist, ny, nx))
        
        # Reconstruct path
        if distances[end_y, end_x] == float('inf'):
            return None, float('inf')
            
        path = []
        current = (end_y, end_x)
        while current in parent:
            path.append(current)
            current = parent[current]
        path.append((start_y, start_x))
        path.reverse()
        
        return path, distances[end_y, end_x]
    
    def separate_paths_by_side(apex_pixel, stem_pixel, all_edge_coords):
        """Separate edge pixels into left and right sides relative to apex-stem line"""
        # Vector from apex to stem
        apex_to_stem = stem_pixel - apex_pixel
        
        left_edges = []
        right_edges = []
        
        for edge_coord in all_edge_coords:
            # Vector from apex to this edge point
            apex_to_edge = edge_coord - apex_pixel
            
            # Cross product to determine left/right (2D cross product = determinant)
            cross_product = apex_to_stem[0] * apex_to_edge[1] - apex_to_stem[1] * apex_to_edge[0]
            
            if cross_product > 0:
                left_edges.append(edge_coord)
            elif cross_product < 0:
                right_edges.append(edge_coord)
            # cross_product == 0 means point is on the apex-stem line
        
        return np.array(left_edges), np.array(right_edges)
    
    
    # Get all edge coordinates
    edge_pixels = np.where(edges_binary)
    all_edge_coords = np.column_stack((edge_pixels[1], edge_pixels[0]))  # (x, y) format
    
    # Separate edges into left and right sides
    left_edges, right_edges = separate_paths_by_side(apex_pixel, stem_pixel, all_edge_coords)
    
    # Debug: Print edge separation results
    print(f"Edge separation debug:")
    print(f"  Total edge pixels: {len(all_edge_coords)}")
    print(f"  Left edge pixels: {len(left_edges)}")
    print(f"  Right edge pixels: {len(right_edges)}")
    print(f"  Apex pixel: {apex_pixel}, Stem pixel: {stem_pixel}")
    
    # Create edge maps for left and right sides only
    h, w = edges_binary.shape
    left_edge_map = np.zeros((h, w), dtype=bool)
    right_edge_map = np.zeros((h, w), dtype=bool)
    
    if len(left_edges) > 0:
        left_edge_map[left_edges[:, 1], left_edges[:, 0]] = True
    if len(right_edges) > 0:
        right_edge_map[right_edges[:, 1], right_edges[:, 0]] = True
    
    # Ensure apex and stem are included in both maps
    apex_y, apex_x = apex_pixel[1], apex_pixel[0]
    stem_y, stem_x = stem_pixel[1], stem_pixel[0]
    left_edge_map[apex_y, apex_x] = True
    left_edge_map[stem_y, stem_x] = True
    right_edge_map[apex_y, apex_x] = True
    right_edge_map[stem_y, stem_x] = True
    
    # Find shortest paths on each side
    left_path, left_length = find_shortest_path_on_edges(apex_pixel, stem_pixel, left_edge_map)
    right_path, right_length = find_shortest_path_on_edges(apex_pixel, stem_pixel, right_edge_map)
    
    # Debug: Print path finding results
    print(f"Path finding debug:")
    print(f"  Left path length: {left_length:.2f}, valid path: {left_path is not None}")
    print(f"  Right path length: {right_length:.2f}, valid path: {right_path is not None}")
    if left_path is not None:
        print(f"  Left path points: {len(left_path)}")
    if right_path is not None:
        print(f"  Right path points: {len(right_path)}")
    
    # New approach: If either path is disconnected, use a different strategy
    if left_length == float('inf') or right_length == float('inf'):
        print(f"Using improved fallback path finding...")
        
        # Find two completely different paths by using a modified Dijkstra that penalizes shared edges
        def find_two_diverse_paths(start_pixel, end_pixel, edges_binary, apex_to_stem_vector):
            """Find two paths that go around different sides of the leaf"""
            from heapq import heappush, heappop
            
            h, w = edges_binary.shape
            start_y, start_x = start_pixel[1], start_pixel[0]
            end_y, end_x = end_pixel[1], end_pixel[0]
            
            if not (edges_binary[start_y, start_x] and edges_binary[end_y, end_x]):
                return None, None, float('inf'), float('inf')
            
            # Find the first path (shortest)
            distances = np.full((h, w), float('inf'))
            distances[start_y, start_x] = 0.0
            visited = np.zeros((h, w), dtype=bool)
            parent = {}
            pq = [(0.0, start_y, start_x)]
            
            while pq:
                current_dist, y, x = heappop(pq)
                if visited[y, x]:
                    continue
                visited[y, x] = True
                if y == end_y and x == end_x:
                    break
                for ny, nx, edge_dist in get_neighbors_8connected(y, x, h, w):
                    if not edges_binary[ny, nx] or visited[ny, nx]:
                        continue
                    new_dist = current_dist + edge_dist
                    if new_dist < distances[ny, nx]:
                        distances[ny, nx] = new_dist
                        parent[(ny, nx)] = (y, x)
                        heappush(pq, (new_dist, ny, nx))
            
            # Reconstruct first path
            if distances[end_y, end_x] == float('inf'):
                return None, None, float('inf'), float('inf')
            
            path1 = []
            current = (end_y, end_x)
            while current in parent:
                path1.append(current)
                current = parent[current]
            path1.append((start_y, start_x))
            path1.reverse()
            
            # Create a modified edge map that heavily penalizes the first path
            modified_edges = edges_binary.copy().astype(float)
            for py, px in path1:
                # Heavy penalty for using the same edges, but don't completely block them
                modified_edges[py, px] = 0.01  # Very small weight instead of 0
                # Also penalize neighbors to encourage different routes
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = py + dy, px + dx
                        if 0 <= ny < h and 0 <= nx < w and modified_edges[ny, nx] > 0:
                            modified_edges[ny, nx] *= 0.1
            
            # Find the second path with penalties
            distances2 = np.full((h, w), float('inf'))
            distances2[start_y, start_x] = 0.0
            visited2 = np.zeros((h, w), dtype=bool)
            parent2 = {}
            pq2 = [(0.0, start_y, start_x)]
            
            while pq2:
                current_dist, y, x = heappop(pq2)
                if visited2[y, x]:
                    continue
                visited2[y, x] = True
                if y == end_y and x == end_x:
                    break
                for ny, nx, edge_dist in get_neighbors_8connected(y, x, h, w):
                    if modified_edges[ny, nx] <= 0 or visited2[ny, nx]:
                        continue
                    # Use the modified weight
                    actual_dist = edge_dist / modified_edges[ny, nx]
                    new_dist = current_dist + actual_dist
                    if new_dist < distances2[ny, nx]:
                        distances2[ny, nx] = new_dist
                        parent2[(ny, nx)] = (y, x)
                        heappush(pq2, (new_dist, ny, nx))
            
            # Reconstruct second path
            path2 = None
            length2 = float('inf')
            if distances2[end_y, end_x] != float('inf'):
                path2 = []
                current = (end_y, end_x)
                while current in parent2:
                    path2.append(current)
                    current = parent2[current]
                path2.append((start_y, start_x))
                path2.reverse()
                length2 = distances2[end_y, end_x]
            
            return path1, path2, distances[end_y, end_x], length2
        
        # Get apex to stem vector for consistent orientation
        apex_to_stem_vector = stem_pixel - apex_pixel
        
        # Find two diverse paths
        path1, path2, length1, length2 = find_two_diverse_paths(apex_pixel, stem_pixel, edges_binary, apex_to_stem_vector)
        
        if path1 is not None and path2 is not None:
            print(f"  Found two diverse paths: {length1:.2f} and {length2:.2f}")
            
            # Determine which path is left and which is right based on their average position
            # relative to the apex-stem line
            def get_path_side(path, apex_pixel, stem_pixel):
                """Determine if a path is on the left or right side"""
                apex_to_stem = stem_pixel - apex_pixel
                total_cross = 0
                for py, px in path:
                    point = np.array([px, py])  # Convert to (x, y)
                    apex_to_point = point - apex_pixel
                    cross = apex_to_stem[0] * apex_to_point[1] - apex_to_stem[1] * apex_to_point[0]
                    total_cross += cross
                return total_cross > 0  # True for left, False for right
            
            path1_is_left = get_path_side(path1, apex_pixel, stem_pixel)
            path2_is_left = get_path_side(path2, apex_pixel, stem_pixel)
            
            # Assign paths to left and right
            if path1_is_left and not path2_is_left:
                left_path, right_path = path1, path2
                left_length, right_length = length1, length2
                print(f"  Assigned: path1->left, path2->right")
            elif path2_is_left and not path1_is_left:
                left_path, right_path = path2, path1
                left_length, right_length = length2, length1
                print(f"  Assigned: path2->left, path1->right")
            else:
                # If both are on the same side or determination failed, assign arbitrarily
                left_path, right_path = path1, path2
                left_length, right_length = length1, length2
                print(f"  Warning: Could not determine sides, assigned arbitrarily")
        else:
            print(f"  Failed to find two diverse paths")
    
    path_info = {
        'left_path': left_path,
        'right_path': right_path,
        'left_edges_count': len(left_edges),
        'right_edges_count': len(right_edges),
        'total_edges': len(all_edge_coords),
        'used_fallback_left': left_length != float('inf') and len(left_edges) == 0,
        'used_fallback_right': right_length != float('inf') and len(right_edges) == 0
    }
    
    return left_length, right_length, path_info

def resample_and_smooth_path(path, target_spacing=0.5, smoothing_window=5):
    """
    Resample path to uniform spacing and apply strong smoothing to reduce spikes and jaggy edges.
    
    Args:
        path: List of (y, x) coordinates representing the path
        target_spacing: Target distance between consecutive points (smaller = more points)
        smoothing_window: Window size for moving average smoothing (larger = smoother)
        
    Returns:
        np.ndarray: Resampled and smoothed path as (y, x) coordinates
    """
    if path is None or len(path) < 2:
        return path
    
    path_array = np.array(path)
    
    # Calculate cumulative distances
    distances = np.zeros(len(path_array))
    for i in range(1, len(path_array)):
        distances[i] = distances[i-1] + np.linalg.norm(path_array[i] - path_array[i-1])
    
    total_distance = distances[-1]
    if total_distance == 0:
        return path_array
    
    # Resample to finer uniform spacing for better smoothing
    num_resampled = max(2, int(total_distance / target_spacing) + 1)
    resample_distances = np.linspace(0, total_distance, num_resampled)
    
    resampled_path = []
    for sample_dist in resample_distances:
        # Find the segment where this distance falls
        segment_idx = np.searchsorted(distances, sample_dist)
        
        if segment_idx == 0:
            resampled_path.append(path_array[0])
        elif segment_idx >= len(path_array):
            resampled_path.append(path_array[-1])
        else:
            # Linear interpolation between two points
            prev_idx = segment_idx - 1
            prev_dist = distances[prev_idx]
            next_dist = distances[segment_idx]
            
            if next_dist > prev_dist:
                t = (sample_dist - prev_dist) / (next_dist - prev_dist)
                interpolated_point = (1 - t) * path_array[prev_idx] + t * path_array[segment_idx]
                resampled_path.append(interpolated_point)
            else:
                resampled_path.append(path_array[prev_idx])
    
    resampled_path = np.array(resampled_path)
    
    # Apply multiple rounds of smoothing for better results
    if len(resampled_path) > smoothing_window:
        smoothed_path = resampled_path.copy()
        
        # Apply Gaussian smoothing to the coordinates
        sigma = smoothing_window / 3.0  # sigma based on window size
        
        # Smooth x and y coordinates separately
        smoothed_path[:, 0] = ndimage.gaussian_filter1d(smoothed_path[:, 0], sigma=sigma)
        smoothed_path[:, 1] = ndimage.gaussian_filter1d(smoothed_path[:, 1], sigma=sigma)
        
        # Also apply moving average smoothing
        half_window = smoothing_window // 2
        moving_avg_path = np.zeros_like(smoothed_path)
        
        for i in range(len(smoothed_path)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(smoothed_path), i + half_window + 1)
            moving_avg_path[i] = np.mean(smoothed_path[start_idx:end_idx], axis=0)
        
        # Combine Gaussian and moving average (weighted average)
        final_path = 0.7 * smoothed_path + 0.3 * moving_avg_path
        
        # Keep original endpoints to maintain apex/stem positions
        final_path[0] = resampled_path[0]
        final_path[-1] = resampled_path[-1]
        
        return final_path
    else:
        return resampled_path

def sample_points_along_path(path, num_samples):
    """
    Uniformly sample points along a path.
    
    Args:
        path: List of (y, x) coordinates representing the path
        num_samples: Number of points to sample
        
    Returns:
        np.ndarray: Array of sampled (x, y) coordinates
    """
    if path is None or len(path) < 2:
        return np.array([])
    
    # First resample and smooth the path to reduce spikes and jaggy edges
    smoothed_path = resample_and_smooth_path(path, target_spacing=0.5, smoothing_window=7)
    
    if smoothed_path is None or len(smoothed_path) < 2:
        return np.array([])
    
    # Convert smoothed path to (x, y) format
    path_xy = np.column_stack([smoothed_path[:, 1], smoothed_path[:, 0]])  # Convert (y,x) to (x,y)
    
    # Calculate cumulative distances along the path
    distances = np.zeros(len(path_xy))
    for i in range(1, len(path_xy)):
        distances[i] = distances[i-1] + np.linalg.norm(path_xy[i] - path_xy[i-1])
    
    total_distance = distances[-1]
    if total_distance == 0:
        return np.array([path_xy[0]])  # Return start point if path has no length
    
    # Sample uniformly along the path, excluding endpoints (apex and stem)
    # Use num_samples+2 to get the interior points, then exclude first and last
    if num_samples <= 0:
        return np.array([])
    elif num_samples == 1:
        # If only 1 sample requested, take the midpoint
        sample_distances = [total_distance / 2]
    else:
        # Create num_samples+2 points, then exclude endpoints
        sample_distances = np.linspace(0, total_distance, num_samples + 2)[1:-1]
    
    sampled_points = []
    
    for sample_dist in sample_distances:
        # Find the segment where this distance falls
        segment_idx = np.searchsorted(distances, sample_dist)
        
        if segment_idx == 0:
            sampled_points.append(path_xy[0])
        elif segment_idx >= len(path_xy):
            sampled_points.append(path_xy[-1])
        else:
            # Interpolate between two points
            prev_idx = segment_idx - 1
            prev_dist = distances[prev_idx]
            next_dist = distances[segment_idx]
            
            # Linear interpolation factor
            if next_dist > prev_dist:
                t = (sample_dist - prev_dist) / (next_dist - prev_dist)
                interpolated_point = (1 - t) * path_xy[prev_idx] + t * path_xy[segment_idx]
                sampled_points.append(interpolated_point)
            else:
                sampled_points.append(path_xy[prev_idx])
    
    return np.array(sampled_points)

def find_nearest_edge_points_3d(points_2d, edge_3d_indices, aligned_points):
    """
    Find the nearest 3D edge points from the edge_3d_indices for given 2D points.
    
    Args:
        points_2d: Array of 2D points in world coordinates (N, 2)
        edge_3d_indices: Indices of 3D points that correspond to edges
        aligned_points: Aligned 3D points from PCA transformation (M, 3)
        transformation_matrix: 4x4 transformation matrix used for alignment
        
    Returns:
        np.ndarray: Array of nearest 3D edge points (N, 3) in original coordinate system
    """
    if len(points_2d) == 0 or len(edge_3d_indices) == 0:
        return np.array([])
    
    # Get the 3D edge points (still in aligned coordinate system)
    if hasattr(aligned_points, 'cpu'):
        aligned_points_np = aligned_points.cpu().numpy()
    else:
        aligned_points_np = aligned_points
    
    edge_points_aligned = aligned_points_np[edge_3d_indices]
    edge_points_2d = edge_points_aligned[:, :2]  # Take only XY coordinates
    
    # print(f"    Debug: find_nearest_edge_points_3d")
    # print(f"      Total edge 3D points available: {len(edge_3d_indices)}")
    # print(f"      Query 2D points: {len(points_2d)}")
    
    # print(f"🚨 COORDINATE MISMATCH DEBUG:")
    # print(f"      Query points range: x=[{points_2d[:, 0].min():.3f}, {points_2d[:, 0].max():.3f}], y=[{points_2d[:, 1].min():.3f}, {points_2d[:, 1].max():.3f}]")
    # print(f"      Edge 2D points range: x=[{edge_points_2d[:, 0].min():.3f}, {edge_points_2d[:, 0].max():.3f}], y=[{edge_points_2d[:, 1].min():.3f}, {edge_points_2d[:, 1].max():.3f}]")
    # print(f"      📏 Scale difference: query_x_range={points_2d[:, 0].max() - points_2d[:, 0].min():.3f}, edge_x_range={edge_points_2d[:, 0].max() - edge_points_2d[:, 0].min():.3f}")
    
    # Find nearest edge point for each query point
    nearest_edge_points_indices = []
    
    for i, query_point_2d in enumerate(points_2d):
        # Calculate distances to all edge points in 2D
        distances = np.linalg.norm(edge_points_2d - query_point_2d, axis=1)
        # Find the nearest edge point
        nearest_idx = np.argmin(distances)
        nearest_edge_3d_idx = edge_3d_indices[nearest_idx]
        nearest_edge_points_indices.append(nearest_edge_3d_idx)
        
        # Debug: print details for left/right edge points
        # if i == 1:  # First left point
        #     print(f"      First left 2D query: {query_point_2d}")
        #     print(f"      Nearest 3D edge idx: {nearest_edge_3d_idx}")
        #     print(f"      Nearest 3D edge 2D pos: {edge_points_2d[nearest_idx]}")
        #     print(f"      Distance: {distances[nearest_idx]:.4f}")
        # elif i > 1 and len(points_2d) > 3 and i == len(points_2d) - 1:  # Last right point
        #     print(f"      Last right 2D query: {query_point_2d}")
        #     print(f"      Nearest 3D edge idx: {nearest_edge_3d_idx}")
        #     print(f"      Nearest 3D edge 2D pos: {edge_points_2d[nearest_idx]}")
        #     print(f"      Distance: {distances[nearest_idx]:.4f}")
    
    return nearest_edge_points_indices

def generate_template_leaf(new_points: np.ndarray,
                           new_normals: np.ndarray,
                           resample_num=1024, 
                        randomness=True):
    
    print("Length of normals after MLS:", len(new_normals), new_normals.shape)
    
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(new_points)
    pcd.normals = o3d.utility.Vector3dVector(new_normals)
    
    # Statistical outlier removal
    pcd_filtered, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=5.0)
    
    new_points = np.asarray(pcd_filtered.points)
    new_normals = np.asarray(pcd_filtered.normals)

    print(f"Outlier removal: {len(inlier_indices)}/{len(pcd.points)} points kept")

    # Need to fix the normals 
    # First cluster the points by normals
    def check_normal_consistency(normals, consistency_threshold=0.95):
        """
        Check if normals are already consistent (mostly pointing in the same direction)
        
        Args:
            normals: Normal vectors (N, 3)
            consistency_threshold: Minimum ratio of normals pointing in same direction (0.8 = 80%)
            
        Returns:
            is_consistent: bool, True if normals are already consistent
            dominant_ratio: float, ratio of normals in dominant direction
        """
        if len(normals) < 2:
            return True, 1.0
        
        # Compute dot products with the first normal as reference
        reference_normal = normals[0]
        dot_products = np.dot(normals, reference_normal)
        
        # Count normals pointing in same direction (positive dot product)
        same_direction = np.sum(dot_products > 0)
        opposite_direction = len(normals) - same_direction
        
        # Determine dominant direction
        if same_direction >= opposite_direction:
            dominant_ratio = same_direction / len(normals)
        else:
            dominant_ratio = opposite_direction / len(normals)
        
        is_consistent = dominant_ratio >= consistency_threshold
        
        print(f"Normal consistency check:")
        print(f"  Same direction as reference: {same_direction}/{len(normals)} ({same_direction/len(normals):.1%})")
        print(f"  Opposite direction: {opposite_direction}/{len(normals)} ({opposite_direction/len(normals):.1%})")
        print(f"  Dominant direction ratio: {dominant_ratio:.1%}")
        print(f"  Consistent: {is_consistent} (threshold: {consistency_threshold:.1%})")
        
        return is_consistent, dominant_ratio

    # Check if normals are already consistent
    is_consistent, dominant_ratio = check_normal_consistency(new_normals, consistency_threshold=0.8)
    print(f"Normal consistency: {is_consistent}, dominant ratio: {dominant_ratio:.2f}")
    if is_consistent:
        print("✅ Normals are already consistent, skipping clustering and flipping")
        corrected_normals = new_normals
    else:
        print("⚠️ Normals are inconsistent, proceeding with clustering and flipping")
        
        # Need to fix the normals 
        # First cluster the points by normals
        def fix_flipped_normals(points, normals, n_clusters=2, flip_threshold=0.3):
            """
            Cluster normals and flip the minority group to fix inverted normals
            
            Args:
                points: Point coordinates (N, 3)
                normals: Normal vectors (N, 3) 
                n_clusters: Number of clusters (default 2 for front/back faces)
                flip_threshold: If minority group is less than this ratio, flip it
                
            Returns:
                corrected_normals: Fixed normal vectors (N, 3)
            """
            from sklearn.cluster import KMeans
            
            # Cluster normals using K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            normal_labels = kmeans.fit_predict(normals)
            
            # Count points in each cluster
            unique_labels, counts = np.unique(normal_labels, return_counts=True)
            total_points = len(normals)
            
            print(f"Normal clustering results:")
            for i, (label, count) in enumerate(zip(unique_labels, counts)):
                ratio = count / total_points
                print(f"  Cluster {label}: {count} points ({ratio:.1%})")
            
            # Find minority cluster(s) to flip
            corrected_normals = normals.copy()
            
            for label, count in zip(unique_labels, counts):
                ratio = count / total_points
                if ratio < flip_threshold:
                    mask = normal_labels == label
                    corrected_normals[mask] = -corrected_normals[mask]
                    print(f"  → Flipped cluster {label} ({count} points, {ratio:.1%})")
            
            return corrected_normals
        
        corrected_normals = fix_flipped_normals(new_points, new_normals, n_clusters=2, flip_threshold=0.3)

    new_normals = corrected_normals

    if new_points.shape[0] > resample_num:
        if not randomness:
            resampled_indices = fpsample.bucket_fps_kdline_sampling(new_points, resample_num, h=3)
        else:
            resampled_indices = fpsample.bucket_fps_kdline_sampling(new_points, resample_num, h=3, start_idx=0)
    else:
        resampled_indices = np.arange(len(new_points))
    resampled_points = new_points[resampled_indices]
    resampled_normals = new_normals[resampled_indices]
    
    kdtree = cKDTree(resampled_points)
    radii = []
    for i in range(len(resampled_points)):
        dist, idx = kdtree.query(resampled_points[i], k=10)
        radii.append(np.max(dist))
    radii = np.array(radii)
    radii = np.linspace(radii.min() * 0.75, radii.max() * 1.25, 20)
    
    resampled_points_pcd = o3d.geometry.PointCloud()
    resampled_points_pcd.points = o3d.utility.Vector3dVector(resampled_points)
    
    # Align normals fix inversion problem with finding most normals pointing outward

    print("Length of normals before alignment:", len(resampled_normals), resampled_normals.shape)
    resampled_points_pcd.normals = o3d.utility.Vector3dVector(resampled_normals)
    print("Vertex count before reconstruction:", np.asarray(resampled_points_pcd.points).shape)
    # resampled_points_pcd.paint_uniform_color([0.8, 0, 0])

    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        resampled_points_pcd, o3d.utility.DoubleVector(radii))
    print("Vertex count before post-processing:", np.asarray(rec_mesh.vertices).shape)

    # Note: trimesh removes unreferenced vertices by default even with process=False
    # The 2 missing vertices are likely isolated points not used by any triangle
    rec_mesh_trimesh = trimesh.Trimesh(
        vertices=np.asarray(rec_mesh.vertices), 
        faces=np.asarray(rec_mesh.triangles),
        process=False,
        validate=False
    )
    print("Vertex count in trimesh before post-processing:", np.asarray(rec_mesh_trimesh.vertices).shape)
    trimesh.repair.fill_holes(rec_mesh_trimesh)
    trimesh.repair.fix_normals(rec_mesh_trimesh)
    print("Vertex count in trimesh after filling holes:", np.asarray(rec_mesh_trimesh.vertices).shape)
    vertices = np.asarray(rec_mesh_trimesh.vertices)
    triangles = np.asarray(rec_mesh_trimesh.faces)
    
    return vertices, triangles, new_normals

def compute_3d_edge_points(
    texture: np.ndarray,
    aligned_points: np.ndarray,
    x_range: tuple,
    y_range: tuple,
    image_size: int,
    pca_apex_point: np.ndarray,
    pca_base_point: np.ndarray,
    transformation_matrix: np.ndarray,
    num_samples_per_path = 20):
    
    # print(f"🎯 ENTERING compute_3d_edge_points with num_samples_per_path={num_samples_per_path}")
    
    apex_point_3d = pca_apex_point
    apex_point_homogeneous = np.append(apex_point_3d, 1)
    apex_point_aligned = (transformation_matrix @ apex_point_homogeneous)[:3]
    apex_point_2d = apex_point_aligned[:2]
    
    base_point_3d = pca_base_point
    base_point_homogeneous = np.append(base_point_3d, 1)
    base_point_aligned = (transformation_matrix @ base_point_homogeneous)[:3]
    base_point_2d = base_point_aligned[:2]
    
    edges_binary = detect_edges_from_rendered_image(
        texture, method='color_boundary', intensity_threshold=0.1
    )
    edge_3d_indices = project_2d_edges_to_3d(edges_binary, aligned_points, x_range, y_range)
    # edge_3d_points = aligned_points[edge_3d_indices]
    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(aligned_points)
    # pcd.colors = o3d.utility.Vector3dVector(np.ones((len(aligned_points), 3)) * 0.5)
    # apex_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    # apex_sphere.translate(apex_point_aligned)
    # apex_sphere.paint_uniform_color([1.0, 0.0, 0.0])
    # base_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    # base_sphere.translate(base_point_aligned)
    # base_sphere.paint_uniform_color([0.0, 1.0, 0.0])
    # edge_3d_points_pcd = o3d.geometry.PointCloud()
    # edge_3d_points_pcd.points = o3d.utility.Vector3dVector(edge_3d_points)
    # edge_3d_points_pcd.translate(np.array([1e-4, 1e-4, 1e-4]))
    # edge_3d_points_pcd.paint_uniform_color([0.0, 0.0, 1.0])
    # o3d.visualization.draw_geometries([pcd, apex_sphere, base_sphere, edge_3d_points_pcd])
    apex_pixel_pca = world_to_pixel(apex_point_2d, x_range, y_range, image_size)
    base_pixel_pca = world_to_pixel(base_point_2d, x_range, y_range, image_size)
    edge_map = edges_binary.astype(np.uint8).reshape(edges_binary.shape[0], edges_binary.shape[1], 1)
    edge_pixels = np.where(edges_binary)  # Get coordinates of edge pixels
    edge_coords = np.column_stack((edge_pixels[1], edge_pixels[0]))
    
    
    # import matplotlib.pyplot as plt
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ax1.imshow(texture)
    # ax1.set_title('Original Texture')
    # ax1.axis('off')
    
    # ax2.imshow(edges_binary, cmap='gray')
    # if len(edge_coords) > 0:
    #     ax2.scatter(edge_coords[:, 0], edge_coords[:, 1], c='red', s=1, alpha=0.6)
    #     ax2.set_title(f'Edge Coords ({len(edge_coords)} points)')
    # else:
    #     ax2.set_title('Edge Coords (no points found)')
    
    # ax2.scatter(apex_pixel_pca[0], apex_pixel_pca[1], c='blue', s=50, marker='x', label='Apex PCA')
    # ax2.scatter(base_pixel_pca[0], base_pixel_pca[1], c='green', s=50, marker='+', label='Base PCA')
    # ax2.legend()
    # ax2.axis('off')
    
    # plt.tight_layout()
    # plt.show()
    
    distances_to_pca_apex = np.linalg.norm(edge_coords - apex_pixel_pca, axis=1)
    nearest_apex_idx = np.argmin(distances_to_pca_apex)
    apex_pixel = edge_coords[nearest_apex_idx]
    
    distances_to_pca_base = np.linalg.norm(edge_coords - base_pixel_pca, axis=1)
    nearest_base_idx = np.argmin(distances_to_pca_base)
    base_pixel = edge_coords[nearest_base_idx]
    
    # print(f"🔍 DEBUG base point selection:")
    # print(f"  Original base_pixel_pca: {base_pixel_pca}")
    # print(f"  Selected base_pixel: {base_pixel}")
    # print(f"  Distance moved: {np.linalg.norm(base_pixel - base_pixel_pca):.2f} pixels")
    # print(f"  Total edge points: {len(edge_coords)}")
    # print(f"  Min distance to edge: {distances_to_pca_base[nearest_base_idx]:.2f}")
    
    # Show top 5 candidates for base point
    sorted_indices = np.argsort(distances_to_pca_base)
    print(f"  Top 5 base candidates:")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        dist = distances_to_pca_base[idx]
        point = edge_coords[idx]
        print(f"    {i+1}. Point {point} (dist: {dist:.2f})")
        
    if distances_to_pca_base[nearest_base_idx] > 20:  # Large movement
        print(f"  ⚠️ WARNING: Base point moved {distances_to_pca_base[nearest_base_idx]:.2f} pixels from PCA prediction!")
    
    # if True: 
    #     import matplotlib.pyplot as plt
    #     print(f"📊 Creating matplotlib visualization...")
    #     print(f"   matplotlib backend: {plt.get_backend()}")
        
    #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    #     print(f"   Created figure with 3 subplots")
        
    #     ax1.imshow(texture)
    #     ax1.set_title('Original Texture')
    #     ax1.axis('off')
        
    #     ax2.imshow(edges_binary, cmap='gray')
    #     if len(edge_coords) > 0:
    #         ax2.scatter(edge_coords[:, 0], edge_coords[:, 1], c='red', s=1, alpha=0.4, label='All edges')
            
    #         for i in range(min(5, len(sorted_indices))):
    #             idx = sorted_indices[i]
    #             point = edge_coords[idx]
    #             color = 'yellow' if i == 0 else 'orange'
    #             size = 50 if i == 0 else 20
    #             ax2.scatter(point[0], point[1], c=color, s=size, marker='o', 
    #                        edgecolor='black', linewidth=1, label=f'Base candidate {i+1}' if i < 3 else None)
        
    #     ax2.scatter(apex_pixel_pca[0], apex_pixel_pca[1], c='blue', s=100, marker='x', linewidths=3, label='Apex PCA')
    #     ax2.scatter(base_pixel_pca[0], base_pixel_pca[1], c='green', s=100, marker='+', linewidths=3, label='Base PCA')
        
    #     ax2.scatter(apex_pixel[0], apex_pixel[1], c='cyan', s=120, marker='o', edgecolor='blue', linewidth=2, label='Final apex')
    #     ax2.scatter(base_pixel[0], base_pixel[1], c='lime', s=120, marker='s', edgecolor='green', linewidth=2, label='Final base')
        
    #     ax2.plot([base_pixel_pca[0], base_pixel[0]], [base_pixel_pca[1], base_pixel[1]], 'g--', alpha=0.8, linewidth=2, label='Base movement')
        
    #     ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #     ax2.set_title(f'Base Point Selection\n(moved {np.linalg.norm(base_pixel - base_pixel_pca):.1f} pixels)')
        
    #     ax3.imshow(edges_binary, cmap='gray')
        
    #     center = base_pixel_pca
    #     radius = 50
    #     x_min, x_max = max(0, int(center[0]) - radius), min(edges_binary.shape[1], int(center[0]) + radius)
    #     y_min, y_max = max(0, int(center[1]) - radius), min(edges_binary.shape[0], int(center[1]) + radius)
        
    #     ax3.set_xlim(x_min, x_max)
    #     ax3.set_ylim(y_max, y_min)
        
    #     mask = (edge_coords[:, 0] >= x_min) & (edge_coords[:, 0] <= x_max) & \
    #            (edge_coords[:, 1] >= y_min) & (edge_coords[:, 1] <= y_max)
    #     if np.any(mask):
    #         region_edges = edge_coords[mask]
    #         ax3.scatter(region_edges[:, 0], region_edges[:, 1], c='red', s=5, alpha=0.6)
        
    #     ax3.scatter(base_pixel_pca[0], base_pixel_pca[1], c='green', s=100, marker='+', linewidths=3)
    #     ax3.scatter(base_pixel[0], base_pixel[1], c='lime', s=100, marker='s', edgecolor='green', linewidth=2)
    #     ax3.plot([base_pixel_pca[0], base_pixel[0]], [base_pixel_pca[1], base_pixel[1]], 'g--', alpha=0.8, linewidth=2)
        
    #     ax3.set_title(f'Base Region Zoom\nnum_samples_per_path={num_samples_per_path}')
        
    #     plt.tight_layout()
    #     print(f"   Calling plt.show()...")
    #     plt.show()
    #     print(f"   plt.show() completed")
        
    #     try:
    #         plt.savefig('debug_base_point_selection.png', dpi=150, bbox_inches='tight')
    #         print(f"   Also saved as debug_base_point_selection.png")
    #     except Exception as e:
    #         print(f"   Could not save image: {e}")
    
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # ax1.imshow(texture)
    # ax1.set_title('Original Texture')
    # ax1.axis('off')
    
    # ax2.imshow(edges_binary, cmap='gray')
    # if len(edge_coords) > 0:
    #     ax2.scatter(edge_coords[:, 0], edge_coords[:, 1], c='red', s=1, alpha=0.4)
    
    # ax2.scatter(apex_pixel_pca[0], apex_pixel_pca[1], c='blue', s=80, marker='x', label='Apex PCA', linewidths=2)
    # ax2.scatter(base_pixel_pca[0], base_pixel_pca[1], c='green', s=80, marker='+', label='Base PCA', linewidths=2)
    
    # ax2.scatter(apex_pixel[0], apex_pixel[1], c='cyan', s=100, marker='o', label='Nearest Apex Edge', edgecolor='blue', linewidth=2)
    # ax2.scatter(base_pixel[0], base_pixel[1], c='yellow', s=100, marker='s', label='Nearest Base Edge', edgecolor='green', linewidth=2)
    
    # ax2.plot([apex_pixel_pca[0], apex_pixel[0]], [apex_pixel_pca[1], apex_pixel[1]], 'b--', alpha=0.7, linewidth=1)
    # ax2.plot([base_pixel_pca[0], base_pixel[0]], [base_pixel_pca[1], base_pixel[1]], 'g--', alpha=0.7, linewidth=1)
    
    # ax2.legend()
    # ax2.set_title(f'Edge Detection & Nearest Points')
    # ax2.axis('off')
    
    # ax3.imshow(edges_binary, cmap='gray')
    
    # all_points = np.vstack([apex_pixel, base_pixel, apex_pixel_pca, base_pixel_pca])
    # x_min, x_max = max(0, int(all_points[:, 0].min()) - 20), min(edges_binary.shape[1], int(all_points[:, 0].max()) + 20)
    # y_min, y_max = max(0, int(all_points[:, 1].min()) - 20), min(edges_binary.shape[0], int(all_points[:, 1].max()) + 20)
    
    # ax3.set_xlim(x_min, x_max)
    # ax3.set_ylim(y_max, y_min)  
    
    # mask = (edge_coords[:, 0] >= x_min) & (edge_coords[:, 0] <= x_max) & \
    #        (edge_coords[:, 1] >= y_min) & (edge_coords[:, 1] <= y_max)
    # if np.any(mask):
    #     region_edges = edge_coords[mask]
    #     ax3.scatter(region_edges[:, 0], region_edges[:, 1], c='red', s=3, alpha=0.6)
    
    # ax3.scatter(apex_pixel_pca[0], apex_pixel_pca[1], c='blue', s=80, marker='x', linewidths=2)
    # ax3.scatter(base_pixel_pca[0], base_pixel_pca[1], c='green', s=80, marker='+', linewidths=2)
    # ax3.scatter(apex_pixel[0], apex_pixel[1], c='cyan', s=100, marker='o', edgecolor='blue', linewidth=2)
    # ax3.scatter(base_pixel[0], base_pixel[1], c='yellow', s=100, marker='s', edgecolor='green', linewidth=2)
    
    # ax3.plot([apex_pixel_pca[0], apex_pixel[0]], [apex_pixel_pca[1], apex_pixel[1]], 'b--', alpha=0.7, linewidth=2)
    # ax3.plot([base_pixel_pca[0], base_pixel[0]], [base_pixel_pca[1], base_pixel[1]], 'g--', alpha=0.7, linewidth=2)
    
    # ax3.set_title('Zoomed View')
    # ax3.axis('off')
    
    # plt.tight_layout()
    # plt.show()
    
    if len(edge_pixels[0]) > 0:
        edge_coords = np.column_stack((edge_pixels[1], edge_pixels[0]))  # Convert to (x, y) format
        
        # Find the nearest edge pixel to the PCA apex point
        distances_to_pca_apex = np.linalg.norm(edge_coords - apex_pixel_pca, axis=1)
        nearest_apex_idx = np.argmin(distances_to_pca_apex)
        apex_pixel = edge_coords[nearest_apex_idx]  # Corrected apex on edge
        
        # Find the edge pixel farthest from the corrected apex
        distances_to_corrected_apex = np.linalg.norm(edge_coords - apex_pixel, axis=1)
        farthest_idx = np.argmax(distances_to_corrected_apex)
        stem_pixel = edge_coords[farthest_idx]
        
        # print(f"  PCA apex pixel: [{apex_pixel_pca[0]}, {apex_pixel_pca[1]}]")
        # print(f"  Corrected apex pixel (nearest edge): [{apex_pixel[0]}, {apex_pixel[1]}]")
        # print(f"  Distance correction: {distances_to_pca_apex[nearest_apex_idx]:.2f} pixels")
    else:
        # No edges detected - fallback to PCA apex and a default stem point
        print("  Warning: No edges detected in rendered image, using PCA apex as fallback")
        apex_pixel = apex_pixel_pca
        # Use a point far from apex as stem fallback
        stem_pixel = np.array([image_size - apex_pixel_pca[0], image_size - apex_pixel_pca[1]])
        # print(f"  Using fallback apex pixel: [{apex_pixel[0]}, {apex_pixel[1]}]")
        # print(f"  Using fallback stem pixel: [{stem_pixel[0]}, {stem_pixel[1]}]")
    
    apex_point_2d_corrected = pixel_to_world(apex_pixel, x_range, y_range, image_size)
    base_point_2d = pixel_to_world(base_pixel, x_range, y_range, image_size)
    
    left_path_length, right_path_length, path_info = compute_edge_path_lengths(
        edges_binary, apex_pixel, base_pixel
    )   
    
    if path_info['left_path'] is None or path_info['right_path'] is None:
        print("No edge path found")
        structured_points_3d_indices = None
    else:
        print(f"left_path_length: {left_path_length}, right_path_length: {right_path_length}")
        left_sampled_points = sample_points_along_path(path_info['left_path'], num_samples_per_path)
        right_sampled_points = sample_points_along_path(path_info['right_path'], num_samples_per_path)
        
        # print(f"🔍 Path sampling results:")
        # print(f"  Left path samples: {len(left_sampled_points)} points")
        # if len(left_sampled_points) > 0:
        #     print(f"    First left sample: {left_sampled_points[0]} (should be near apex)")
        #     print(f"    Last left sample: {left_sampled_points[-1]} (should be near base)")
        # print(f"  Right path samples: {len(right_sampled_points)} points") 
        # if len(right_sampled_points) > 0:
        #     print(f"    First right sample: {right_sampled_points[0]} (should be near apex)")
        #     print(f"    Last right sample: {right_sampled_points[-1]} (should be near base)")
        # print(f"  Reference points: apex={apex_pixel}, base={base_pixel}")
        
        # import matplotlib.pyplot as plt
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # ax1.imshow(edges_binary, cmap='gray')
        # ax1.set_title(f'Edge Paths & Sampling\n(num_samples_per_path={num_samples_per_path})')
        
        # if path_info['left_path'] is not None and len(path_info['left_path']) > 0:
        #     left_path_array = np.array(path_info['left_path'])
        #     ax1.plot(left_path_array[:, 0], left_path_array[:, 1], 'r-', alpha=0.6, linewidth=2, label=f'Left path ({len(path_info["left_path"])} pts)')
            
        # if path_info['right_path'] is not None and len(path_info['right_path']) > 0:
        #     right_path_array = np.array(path_info['right_path'])
        #     ax1.plot(right_path_array[:, 0], right_path_array[:, 1], 'b-', alpha=0.6, linewidth=2, label=f'Right path ({len(path_info["right_path"])} pts)')
        
        # if len(left_sampled_points) > 0:
        #     left_array = np.array(left_sampled_points)
        #     ax1.scatter(left_array[:, 0], left_array[:, 1], c='red', s=50, marker='o', 
        #                edgecolor='white', linewidth=1, label=f'Left samples ({len(left_sampled_points)})', zorder=5)
        #     ax1.scatter(left_array[0, 0], left_array[0, 1], c='darkred', s=100, marker='^', 
        #                edgecolor='white', linewidth=2, label='Left start', zorder=6)
        #     ax1.scatter(left_array[-1, 0], left_array[-1, 1], c='darkred', s=100, marker='v', 
        #                edgecolor='white', linewidth=2, label='Left end', zorder=6)
            
        # if len(right_sampled_points) > 0:
        #     right_array = np.array(right_sampled_points)
        #     ax1.scatter(right_array[:, 0], right_array[:, 1], c='blue', s=50, marker='s', 
        #                edgecolor='white', linewidth=1, label=f'Right samples ({len(right_sampled_points)})', zorder=5)
        #     ax1.scatter(right_array[0, 0], right_array[0, 1], c='darkblue', s=100, marker='^', 
        #                edgecolor='white', linewidth=2, label='Right start', zorder=6)
        #     ax1.scatter(right_array[-1, 0], right_array[-1, 1], c='darkblue', s=100, marker='v', 
        #                edgecolor='white', linewidth=2, label='Right end', zorder=6)
        
        # ax1.scatter(apex_pixel[0], apex_pixel[1], c='lime', s=150, marker='*', 
        #            edgecolor='black', linewidth=2, label='Apex', zorder=7)
        # ax1.scatter(base_pixel[0], base_pixel[1], c='yellow', s=150, marker='P', 
        #            edgecolor='black', linewidth=2, label='Base', zorder=7)
        
        # ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # ax2.imshow(edges_binary, cmap='gray')
        # ax2.set_title('Sampling Points Detail')
        
        # all_points = []
        # if len(left_sampled_points) > 0:
        #     all_points.extend(left_sampled_points)
        # if len(right_sampled_points) > 0:
        #     all_points.extend(right_sampled_points)
        # all_points.extend([apex_pixel, base_pixel])
        
        # if len(all_points) > 0:
        #     all_points = np.array(all_points)
        #     margin = 30
        #     x_min = max(0, int(all_points[:, 0].min()) - margin)
        #     x_max = min(edges_binary.shape[1], int(all_points[:, 0].max()) + margin)
        #     y_min = max(0, int(all_points[:, 1].min()) - margin)
        #     y_max = min(edges_binary.shape[0], int(all_points[:, 1].max()) + margin)
            
        #     ax2.set_xlim(x_min, x_max)
        #     ax2.set_ylim(y_max, y_min)
            
        #     if len(left_sampled_points) > 0:
        #         left_array = np.array(left_sampled_points)
        #         mask = (left_array[:, 0] >= x_min) & (left_array[:, 0] <= x_max) & \
        #                (left_array[:, 1] >= y_min) & (left_array[:, 1] <= y_max)
        #         if np.any(mask):
        #             visible_left = left_array[mask]
        #             ax2.scatter(visible_left[:, 0], visible_left[:, 1], c='red', s=40, marker='o', edgecolor='white', linewidth=1)
        #             for i, point in enumerate(visible_left):
        #                 ax2.annotate(f'{np.where(mask)[0][i]}', (point[0], point[1]), xytext=(5, 5), 
        #                            textcoords='offset points', fontsize=8, color='red', weight='bold')
            
        #     if len(right_sampled_points) > 0:
        #         right_array = np.array(right_sampled_points)
        #         mask = (right_array[:, 0] >= x_min) & (right_array[:, 0] <= x_max) & \
        #                (right_array[:, 1] >= y_min) & (right_array[:, 1] <= y_max)
        #         if np.any(mask):
        #             visible_right = right_array[mask]
        #             ax2.scatter(visible_right[:, 0], visible_right[:, 1], c='blue', s=40, marker='s', edgecolor='white', linewidth=1)
        #             for i, point in enumerate(visible_right):
        #                 ax2.annotate(f'{np.where(mask)[0][i]}', (point[0], point[1]), xytext=(5, 5), 
        #                            textcoords='offset points', fontsize=8, color='blue', weight='bold')
            
        #     ax2.scatter(apex_pixel[0], apex_pixel[1], c='lime', s=100, marker='*', edgecolor='black', linewidth=2)
        #     ax2.scatter(base_pixel[0], base_pixel[1], c='yellow', s=100, marker='P', edgecolor='black', linewidth=2)
        
        # plt.tight_layout()
        # plt.show()
        
        # try:
        #     plt.savefig('debug_path_sampling.png', dpi=150, bbox_inches='tight')
        #     print(f"   Path sampling visualization saved as debug_path_sampling.png")
        # except Exception as e:
        #     print(f"   Could not save sampling image: {e}")
        
        left_sampled_points_world = []
        if len(left_sampled_points) > 0:
            for point_pixel in left_sampled_points:
                point_world = pixel_to_world(point_pixel, x_range, y_range, image_size)
                left_sampled_points_world.append(point_world)
            left_sampled_points_world = np.array(left_sampled_points_world)
        else:
            left_sampled_points_world = np.array([])
            
        right_sampled_points_world = []
        if len(right_sampled_points) > 0:
            for point_pixel in right_sampled_points:
                point_world = pixel_to_world(point_pixel, x_range, y_range, image_size)
                right_sampled_points_world.append(point_world)
            right_sampled_points_world = np.array(right_sampled_points_world)
        else:
            right_sampled_points_world = np.array([])
            
        ordered_points_2d = []
        ordered_points_2d.append(apex_point_2d_corrected)
        if len(left_sampled_points) > 0:
            ordered_points_2d.extend(left_sampled_points_world)
        ordered_points_2d.append(base_point_2d)
        if len(right_sampled_points) > 0:
            ordered_points_2d.extend(right_sampled_points_world)
            
        structured_points_2d = np.array(ordered_points_2d)
        
        # import matplotlib.pyplot as plt
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ax1.imshow(texture, alpha=0.7)
        
        # apex_pixel_2d = world_to_pixel(apex_point_2d_corrected, x_range, y_range, image_size)
        # base_pixel_2d = world_to_pixel(base_point_2d, x_range, y_range, image_size)
        
        # ax1.scatter(apex_pixel_2d[0], apex_pixel_2d[1], c='red', s=100, marker='*', label='Apex', edgecolor='black', linewidth=2, zorder=10)
        # ax1.scatter(base_pixel_2d[0], base_pixel_2d[1], c='green', s=100, marker='s', label='Base', edgecolor='black', linewidth=2, zorder=10)
        
        # if len(left_sampled_points) > 0:
        #     left_pixels = np.array([world_to_pixel(pt, x_range, y_range, image_size) for pt in left_sampled_points_world])
        #     ax1.scatter(left_pixels[:, 0], left_pixels[:, 1], c='blue', s=50, marker='o', label=f'Left Points ({len(left_sampled_points)})', alpha=0.8, zorder=5)
            
        #     for i in range(len(left_pixels)-1):
        #         ax1.plot([left_pixels[i][0], left_pixels[i+1][0]], [left_pixels[i][1], left_pixels[i+1][1]], 'b-', alpha=0.5, linewidth=1)
            
        #     ax1.plot([apex_pixel_2d[0], left_pixels[0][0]], [apex_pixel_2d[1], left_pixels[0][1]], 'b-', alpha=0.5, linewidth=1)
        #     ax1.plot([left_pixels[-1][0], base_pixel_2d[0]], [left_pixels[-1][1], base_pixel_2d[1]], 'b-', alpha=0.5, linewidth=1)
        # else:
        #     ax1.plot([apex_pixel_2d[0], base_pixel_2d[0]], [apex_pixel_2d[1], base_pixel_2d[1]], 'gray', alpha=0.5, linewidth=1, linestyle='--')
        
        # if len(right_sampled_points) > 0:
        #     right_pixels = np.array([world_to_pixel(pt, x_range, y_range, image_size) for pt in right_sampled_points_world])
        #     ax1.scatter(right_pixels[:, 0], right_pixels[:, 1], c='orange', s=50, marker='^', label=f'Right Points ({len(right_sampled_points)})', alpha=0.8, zorder=5)
            
        #     for i in range(len(right_pixels)-1):
        #         ax1.plot([right_pixels[i][0], right_pixels[i+1][0]], [right_pixels[i][1], right_pixels[i+1][1]], 'orange', alpha=0.5, linewidth=1)
        #     ax1.plot([base_pixel_2d[0], right_pixels[0][0]], [base_pixel_2d[1], right_pixels[0][1]], 'orange', alpha=0.5, linewidth=1)
        
        # ax1.set_title('Structured Points on Texture')
        # ax1.legend()
        # ax1.axis('off')
        
        # ax2.imshow(edges_binary, cmap='gray')
        
        # ax2.scatter(apex_pixel_2d[0], apex_pixel_2d[1], c='red', s=100, marker='*', label='Apex', edgecolor='white', linewidth=2, zorder=10)
        # ax2.scatter(base_pixel_2d[0], base_pixel_2d[1], c='green', s=100, marker='s', label='Base', edgecolor='white', linewidth=2, zorder=10)
        
        # if len(left_sampled_points) > 0:
        #     ax2.scatter(left_pixels[:, 0], left_pixels[:, 1], c='cyan', s=50, marker='o', label=f'Left Points ({len(left_sampled_points)})', alpha=0.8, zorder=5)
        #     for i in range(len(left_pixels)-1):
        #         ax2.plot([left_pixels[i][0], left_pixels[i+1][0]], [left_pixels[i][1], left_pixels[i+1][1]], 'cyan', alpha=0.7, linewidth=1)
        #     ax2.plot([apex_pixel_2d[0], left_pixels[0][0]], [apex_pixel_2d[1], left_pixels[0][1]], 'cyan', alpha=0.7, linewidth=1)
        #     ax2.plot([left_pixels[-1][0], base_pixel_2d[0]], [left_pixels[-1][1], base_pixel_2d[1]], 'cyan', alpha=0.7, linewidth=1)
        
        # if len(right_sampled_points) > 0:
        #     ax2.scatter(right_pixels[:, 0], right_pixels[:, 1], c='yellow', s=50, marker='^', label=f'Right Points ({len(right_sampled_points)})', alpha=0.8, zorder=5)
        #     for i in range(len(right_pixels)-1):
        #         ax2.plot([right_pixels[i][0], right_pixels[i+1][0]], [right_pixels[i][1], right_pixels[i+1][1]], 'yellow', alpha=0.7, linewidth=1)
        #     ax2.plot([base_pixel_2d[0], right_pixels[0][0]], [base_pixel_2d[1], right_pixels[0][1]], 'yellow', alpha=0.7, linewidth=1)
        
        # ax2.set_title('Structured Points on Edge Detection')
        # ax2.legend()
        # ax2.axis('off')
        
        # plt.tight_layout()
        # plt.show()
        
        # print(f"Structured Points 2D Summary:")
        # print(f"  - Apex point: {apex_point_2d_corrected}")
        # print(f"  - Left sampled points: {len(left_sampled_points) if len(left_sampled_points) > 0 else 0}")
        # print(f"  - Base point: {base_point_2d}")
        # print(f"  - Right sampled points: {len(right_sampled_points) if len(right_sampled_points) > 0 else 0}")
        # print(f"  - Total structured points: {len(structured_points_2d)}")
        
        
        
        structured_points_3d_indices = find_nearest_edge_points_3d(
            structured_points_2d, edge_3d_indices, aligned_points,
        )
        
        # if structured_points_3d_indices is not None and len(structured_points_3d_indices) > 0:
        #     import matplotlib.pyplot as plt
        #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
            
        #     ax1.imshow(texture, alpha=0.7)
            
        #     structured_pixels_2d = np.array([world_to_pixel(pt, x_range, y_range, image_size) for pt in structured_points_2d])
        #     ax1.scatter(structured_pixels_2d[:, 0], structured_pixels_2d[:, 1], c='red', s=80, marker='o', 
        #                label=f'Target 2D Points ({len(structured_points_2d)})', edgecolor='black', linewidth=1, zorder=10, alpha=0.8)
        #     found_3d_points = aligned_points[structured_points_3d_indices]
        #     found_2d_world = found_3d_points[:, :2]  
        #     found_pixels_2d = np.array([world_to_pixel(pt, x_range, y_range, image_size) for pt in found_2d_world])
        #     ax1.scatter(found_pixels_2d[:, 0], found_pixels_2d[:, 1], c='blue', s=60, marker='x', 
        #                label=f'Found 3D Points ({len(found_pixels_2d)})', linewidth=2, zorder=15)
            
        #     for i in range(min(len(structured_pixels_2d), len(found_pixels_2d))):
        #         ax1.plot([structured_pixels_2d[i][0], found_pixels_2d[i][0]], 
        #                 [structured_pixels_2d[i][1], found_pixels_2d[i][1]], 
        #                 'green', alpha=0.6, linewidth=1, linestyle='--')
            
        #     distances = []
        #     for i in range(min(len(structured_pixels_2d), len(found_pixels_2d))):
        #         dist = np.linalg.norm(structured_pixels_2d[i] - found_pixels_2d[i])
        #         distances.append(dist)
            
        #     ax1.set_title(f'2D Target vs Found 3D Projection\nAvg Error: {np.mean(distances):.2f}px, Max Error: {np.max(distances):.2f}px')
        #     ax1.legend()
        #     ax1.axis('off')
            
        #     ax2.imshow(edges_binary, cmap='gray')
            
        #     if len(edge_3d_indices) > 0:
        #         edge_3d_points = aligned_points[edge_3d_indices]
        #         edge_2d_world = edge_3d_points[:, :2]
        #         edge_pixels_2d = np.array([world_to_pixel(pt, x_range, y_range, image_size) for pt in edge_2d_world])
        #         ax2.scatter(edge_pixels_2d[:, 0], edge_pixels_2d[:, 1], c='gray', s=1, alpha=0.3, 
        #                    label=f'All Edge 3D Points ({len(edge_3d_indices)})')
            
        #     ax2.scatter(structured_pixels_2d[:, 0], structured_pixels_2d[:, 1], c='red', s=80, marker='o', 
        #                label='Target 2D Points', edgecolor='white', linewidth=1, zorder=10)
        #     ax2.scatter(found_pixels_2d[:, 0], found_pixels_2d[:, 1], c='cyan', s=60, marker='x', 
        #                label='Found 3D Points', linewidth=2, zorder=15)
            
        #     for i in range(min(len(structured_pixels_2d), len(found_pixels_2d))):
        #         ax2.plot([structured_pixels_2d[i][0], found_pixels_2d[i][0]], 
        #                 [structured_pixels_2d[i][1], found_pixels_2d[i][1]], 
        #                 'yellow', alpha=0.7, linewidth=1, linestyle='--')
            
        #     ax2.set_title('Edge Detection with 3D Matching')
        #     ax2.legend()
        #     ax2.axis('off')
            
        #     ax3.imshow(edges_binary, cmap='gray', alpha=0.3)
            
        #     aligned_points_2d = aligned_points[:, :2] 
        #     all_aligned_pixels = np.array([world_to_pixel(pt, x_range, y_range, image_size) for pt in aligned_points_2d])
            
        #     valid_mask = (all_aligned_pixels[:, 0] >= 0) & (all_aligned_pixels[:, 0] < edges_binary.shape[1]) & \
        #                 (all_aligned_pixels[:, 1] >= 0) & (all_aligned_pixels[:, 1] < edges_binary.shape[0])
        #     valid_aligned_pixels = all_aligned_pixels[valid_mask]
            
        #     ax3.scatter(valid_aligned_pixels[:, 0], valid_aligned_pixels[:, 1], c='lightblue', s=0.5, alpha=0.4, 
        #                label=f'All Aligned Points ({len(valid_aligned_pixels)}/{len(aligned_points)})')
            
        #     if len(edge_3d_indices) > 0:
        #         edge_3d_points = aligned_points[edge_3d_indices]
        #         edge_2d_world = edge_3d_points[:, :2]
        #         edge_pixels_2d = np.array([world_to_pixel(pt, x_range, y_range, image_size) for pt in edge_2d_world])
        #         ax3.scatter(edge_pixels_2d[:, 0], edge_pixels_2d[:, 1], c='red', s=2, alpha=0.7, 
        #                    label=f'Edge 3D Points ({len(edge_3d_indices)})')
            
        #     ax3.scatter(structured_pixels_2d[:, 0], structured_pixels_2d[:, 1], c='yellow', s=100, marker='o', 
        #                label='Target 2D Points', edgecolor='black', linewidth=2, zorder=10)
        #     ax3.scatter(found_pixels_2d[:, 0], found_pixels_2d[:, 1], c='green', s=80, marker='x', 
        #                label='Found 3D Points', linewidth=3, zorder=15)
            
        #     for i in range(min(len(structured_pixels_2d), len(found_pixels_2d))):
        #         dist = np.linalg.norm(structured_pixels_2d[i] - found_pixels_2d[i])
        #         if dist > 20: 
        #             ax3.plot([structured_pixels_2d[i][0], found_pixels_2d[i][0]], 
        #                     [structured_pixels_2d[i][1], found_pixels_2d[i][1]], 
        #                     'orange', alpha=0.8, linewidth=2, linestyle='--')

        #             mid_x = (structured_pixels_2d[i][0] + found_pixels_2d[i][0]) / 2
        #             mid_y = (structured_pixels_2d[i][1] + found_pixels_2d[i][1]) / 2
        #             ax3.text(mid_x, mid_y, f'{dist:.0f}', fontsize=8, color='orange', 
        #                     ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            
        #     ax3.set_title('Aligned Points Distribution Analysis')
        #     ax3.legend(loc='upper right', fontsize=8)
        #     ax3.axis('off')
            
        #     plt.tight_layout()
        #     plt.show()
            
        #     print(f"\n3D Point Matching Analysis:")
        #     print(f"  - Target 2D points: {len(structured_points_2d)}")
        #     print(f"  - Found 3D points: {len(structured_points_3d_indices) if structured_points_3d_indices is not None else 0}")
        #     print(f"  - Available edge 3D points: {len(edge_3d_indices)}")
        #     print(f"  - Total aligned points: {len(aligned_points)}")
        #     print(f"  - Valid aligned points in image: {len(valid_aligned_pixels)}/{len(aligned_points)} ({len(valid_aligned_pixels)/len(aligned_points)*100:.1f}%)")
        #     print(f"  - Pixel distance errors: min={np.min(distances):.2f}, max={np.max(distances):.2f}, avg={np.mean(distances):.2f}")
            
        #     large_errors = [d for d in distances if d > 20]
        #     medium_errors = [d for d in distances if 5 < d <= 20]
        #     small_errors = [d for d in distances if d <= 5]
        #     print(f"  - Error distribution: {len(small_errors)} good (≤5px), {len(medium_errors)} medium (5-20px), {len(large_errors)} bad (>20px)")
            
        #     if len(large_errors) > 0:
        #         print(f"  ⚠️  {len(large_errors)} points have large errors > 20px:")
        #         for i, dist in enumerate(distances):
        #             if dist > 20:
        #                 print(f"    Point {i}: target={structured_pixels_2d[i]}, found={found_pixels_2d[i]}, error={dist:.1f}px")
            
        #     if np.max(distances) > 10:  
        #         print(f"\n  🔍 DIAGNOSIS:")
        #         print(f"     - Image size: {edges_binary.shape}")
        #         print(f"     - X range: {x_range}")
        #         print(f"     - Y range: {y_range}")
        #         print(f"     - Check if coordinate transformations are consistent")
        #         print(f"     - Check if edge_3d_indices correctly correspond to edge pixels")
        #         print(f"     - Verify find_nearest_edge_points_3d algorithm")
        #     else:
        #         print(f"  ✅ Matching errors are within reasonable range (< 10px)")
        # else:
        #     print("⚠️  WARNING: structured_points_3d_indices is None or empty!")
        
    add_info = {
        'edge_map': edge_map,
        'apex_pixel': apex_pixel,                
        'stem_pixel': base_pixel,                
        'apex_2d': apex_point_2d_corrected,      
        'stem_2d': base_point_2d,               
        'left_sampled_points': left_sampled_points_world,
        'right_sampled_points': right_sampled_points_world,
        'left_sampled_points_pixel': left_sampled_points,
        'right_sampled_points_pixel': right_sampled_points,
        'num_samples_per_path': num_samples_per_path
    }
        
    return structured_points_3d_indices, add_info

def compute_3d_edge_points_kai(
    texture: np.ndarray,
    depth: np.ndarray,
    aligned_points: np.ndarray,
    x_range: tuple,
    y_range: tuple,
    image_size: int,
    pca_apex_point: np.ndarray,
    pca_base_point: np.ndarray,
    transformation_matrix: np.ndarray,
    num_samples_per_path = 20):
    
    print(f"🎯 ENTERING compute_3d_edge_points with num_samples_per_path={num_samples_per_path}")
    
    apex_point_3d = pca_apex_point
    apex_point_homogeneous = np.append(apex_point_3d, 1)
    apex_point_aligned = (transformation_matrix @ apex_point_homogeneous)[:3]
    apex_point_2d = apex_point_aligned[:2]
    
    base_point_3d = pca_base_point
    base_point_homogeneous = np.append(base_point_3d, 1)
    base_point_aligned = (transformation_matrix @ base_point_homogeneous)[:3]
    base_point_2d = base_point_aligned[:2]
    
    edges_binary = detect_edges_from_rendered_image(
        texture, method='color_boundary', intensity_threshold=0.1
    )
    edge_3d_indices = project_2d_edges_to_3d(edges_binary, aligned_points, x_range, y_range)
  
    apex_pixel_pca = world_to_pixel(apex_point_2d, x_range, y_range, image_size)
    base_pixel_pca = world_to_pixel(base_point_2d, x_range, y_range, image_size)
    edge_map = edges_binary.astype(np.uint8).reshape(edges_binary.shape[0], edges_binary.shape[1], 1)
    edge_pixels = np.where(edges_binary)  # Get coordinates of edge pixels
    edge_coords = np.column_stack((edge_pixels[1], edge_pixels[0]))
    
    distances_to_pca_apex = np.linalg.norm(edge_coords - apex_pixel_pca, axis=1)
    nearest_apex_idx = np.argmin(distances_to_pca_apex)
    apex_pixel = edge_coords[nearest_apex_idx]
    
    distances_to_pca_base = np.linalg.norm(edge_coords - base_pixel_pca, axis=1)
    nearest_base_idx = np.argmin(distances_to_pca_base)
    base_pixel = edge_coords[nearest_base_idx]
    
    # Show top 5 candidates for base point
    sorted_indices = np.argsort(distances_to_pca_base)
    print(f"  Top 5 base candidates:")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        dist = distances_to_pca_base[idx]
        point = edge_coords[idx]
        print(f"    {i+1}. Point {point} (dist: {dist:.2f})")
        
    if distances_to_pca_base[nearest_base_idx] > 20:  # Large movement
        print(f"  ⚠️ WARNING: Base point moved {distances_to_pca_base[nearest_base_idx]:.2f} pixels from PCA prediction!")
    
    if len(edge_pixels[0]) > 0:
        edge_coords = np.column_stack((edge_pixels[1], edge_pixels[0]))  # Convert to (x, y) format
        
        # Find the nearest edge pixel to the PCA apex point
        distances_to_pca_apex = np.linalg.norm(edge_coords - apex_pixel_pca, axis=1)
        nearest_apex_idx = np.argmin(distances_to_pca_apex)
        apex_pixel = edge_coords[nearest_apex_idx]  # Corrected apex on edge
        
        # Find the edge pixel farthest from the corrected apex
        distances_to_corrected_apex = np.linalg.norm(edge_coords - apex_pixel, axis=1)
        farthest_idx = np.argmax(distances_to_corrected_apex)
        stem_pixel = edge_coords[farthest_idx]
        
        print(f"  PCA apex pixel: [{apex_pixel_pca[0]}, {apex_pixel_pca[1]}]")
        print(f"  Corrected apex pixel (nearest edge): [{apex_pixel[0]}, {apex_pixel[1]}]")
        print(f"  Distance correction: {distances_to_pca_apex[nearest_apex_idx]:.2f} pixels")
    else:
        # No edges detected - fallback to PCA apex and a default stem point
        print("  Warning: No edges detected in rendered image, using PCA apex as fallback")
        apex_pixel = apex_pixel_pca
        # Use a point far from apex as stem fallback
        stem_pixel = np.array([image_size - apex_pixel_pca[0], image_size - apex_pixel_pca[1]])
        print(f"  Using fallback apex pixel: [{apex_pixel[0]}, {apex_pixel[1]}]")
        print(f"  Using fallback stem pixel: [{stem_pixel[0]}, {stem_pixel[1]}]")
    
    apex_point_2d_corrected = pixel_to_world(apex_pixel, x_range, y_range, image_size)
    base_point_2d = pixel_to_world(base_pixel, x_range, y_range, image_size)
    
    left_path_length, right_path_length, path_info = compute_edge_path_lengths(
        edges_binary, apex_pixel, base_pixel
    )   
    
    if path_info['left_path'] is None or path_info['right_path'] is None:
        print("No edge path found")
        structured_points_3d_indices = None
    else:
        print(f"left_path_length: {left_path_length}, right_path_length: {right_path_length}")
        left_sampled_points = sample_points_along_path(path_info['left_path'], num_samples_per_path)
        right_sampled_points = sample_points_along_path(path_info['right_path'], num_samples_per_path)
        
        print(f"🔍 Path sampling results:")
        print(f"  Left path samples: {len(left_sampled_points)} points")
        if len(left_sampled_points) > 0:
            print(f"    First left sample: {left_sampled_points[0]} (should be near apex)")
            print(f"    Last left sample: {left_sampled_points[-1]} (should be near base)")
        print(f"  Right path samples: {len(right_sampled_points)} points") 
        if len(right_sampled_points) > 0:
            print(f"    First right sample: {right_sampled_points[0]} (should be near apex)")
            print(f"    Last right sample: {right_sampled_points[-1]} (should be near base)")
        print(f"  Reference points: apex={apex_pixel}, base={base_pixel}")
       
        left_sampled_points_world = []
        if len(left_sampled_points) > 0:
            for point_pixel in left_sampled_points:
                point_world = pixel_to_world(point_pixel, x_range, y_range, image_size)
                left_sampled_points_world.append(point_world)
            left_sampled_points_world = np.array(left_sampled_points_world)
        else:
            left_sampled_points_world = np.array([])
            
        right_sampled_points_world = []
        if len(right_sampled_points) > 0:
            for point_pixel in right_sampled_points:
                point_world = pixel_to_world(point_pixel, x_range, y_range, image_size)
                right_sampled_points_world.append(point_world)
            right_sampled_points_world = np.array(right_sampled_points_world)
        else:
            right_sampled_points_world = np.array([])
            
        ordered_points_2d = []
        ordered_points_2d.append(apex_point_2d_corrected)
        if len(left_sampled_points) > 0:
            ordered_points_2d.extend(left_sampled_points_world)
        ordered_points_2d.append(base_point_2d)
        if len(right_sampled_points) > 0:
            ordered_points_2d.extend(right_sampled_points_world)
            
        structured_points_2d = np.array(ordered_points_2d)
       
        print(f"Structured Points 2D Summary:")
        print(f"  - Apex point: {apex_point_2d_corrected}")
        print(f"  - Left sampled points: {len(left_sampled_points) if len(left_sampled_points) > 0 else 0}")
        print(f"  - Base point: {base_point_2d}")
        print(f"  - Right sampled points: {len(right_sampled_points) if len(right_sampled_points) > 0 else 0}")
        print(f"  - Total structured points: {len(structured_points_2d)}")
        
        structured_points_2d_world = np.array([world_to_pixel(pt, x_range, y_range, image_size) for pt in structured_points_2d])
        structured_points_3d = project_2d_coords_to_3d(
            structured_points_2d_world, depth, x_range, y_range, image_size
        )

    add_info = {
        'edge_map': edge_map,
        'apex_pixel': apex_pixel,                
        'stem_pixel': base_pixel,                
        'apex_2d': apex_point_2d_corrected,      
        'stem_2d': base_point_2d,                
        'left_sampled_points': left_sampled_points_world,
        'right_sampled_points': right_sampled_points_world,
        'left_sampled_points_pixel': left_sampled_points,
        'right_sampled_points_pixel': right_sampled_points,
        'num_samples_per_path': num_samples_per_path
    }
        
    return structured_points_3d, add_info

def extract_leaf_boundary_polygon(apex_stem_info):

    left_sampled_points = apex_stem_info.get('left_sampled_points_pixel')
    right_sampled_points = apex_stem_info.get('right_sampled_points_pixel') 
    apex_pixel = apex_stem_info.get('apex_pixel')
    stem_pixel = apex_stem_info.get('stem_pixel')
    
    if left_sampled_points is None or right_sampled_points is None or len(left_sampled_points) == 0 or len(right_sampled_points) == 0:
        return np.array([])
    
    boundary_points = []
    
    boundary_points.append([apex_pixel[0], apex_pixel[1]])
    
    for point in left_sampled_points:
        boundary_points.append([point[0], point[1]])
    
    boundary_points.append([stem_pixel[0], stem_pixel[1]])
    
    for point in reversed(right_sampled_points):
        boundary_points.append([point[0], point[1]])
    
    return np.array(boundary_points)

def point_in_polygon(point, polygon):

    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def generate_interior_sample_points(boundary_polygon, 
                                    grid_density=20, 
                                    boundary_margin=0.15,
                                    distance_factor=0.8, 
                                    use_fps=False, 
                                    max_fps_points=40, 
                                    fps_indices=None):
   
    if len(boundary_polygon) == 0:
        return np.array([])
    
    min_x, min_y = boundary_polygon.min(axis=0)
    max_x, max_y = boundary_polygon.max(axis=0)
    
    margin_x = (max_x - min_x) * boundary_margin  
    margin_y = (max_y - min_y) * boundary_margin
    
    min_x += margin_x
    max_x -= margin_x
    min_y += margin_y
    max_y -= margin_y
    
    grid_density = int(grid_density)
    x_coords = np.linspace(min_x, max_x, grid_density)
    y_coords = np.linspace(min_y, max_y, grid_density)
    
    interior_points = []
    total_grid_points = grid_density * grid_density
    
    
    for x in x_coords:
        for y in y_coords:
            test_point = np.array([x, y])
            
            if point_in_polygon(test_point, boundary_polygon):
                distances_to_boundary = np.linalg.norm(boundary_polygon - test_point, axis=1)
                min_distance = np.min(distances_to_boundary)
                
                min_required_distance = min(margin_x, margin_y) * distance_factor  
                if min_distance > min_required_distance:
                    interior_points.append(test_point)
    
    interior_points = np.array(interior_points)
    
    if fps_indices is not None:
        if len(interior_points) > max(fps_indices):
            interior_points = interior_points[fps_indices]
        return interior_points
    
    elif use_fps and len(interior_points) > max_fps_points:
        selected_indices = fpsample.bucket_fps_kdline_sampling(interior_points, max_fps_points, h=5)
        interior_points = interior_points[selected_indices]
        return interior_points, selected_indices
    
    elif use_fps:
        return interior_points, np.arange(len(interior_points))
    
    else:
        return interior_points

def compute_mean_value_coordinates(point, polygon):

    point = np.array(point)
    polygon = np.array(polygon)
    n = len(polygon)
    
    if n < 3:
        return np.array([])
    
    vectors = polygon - point
    distances = np.linalg.norm(vectors, axis=1)
    
    epsilon = 1e-10
    for i in range(n):
        if distances[i] < epsilon:
            weights = np.zeros(n)
            weights[i] = 1.0
            return weights
    
    unit_vectors = vectors / distances.reshape(-1, 1)
    
    weights = np.zeros(n)
    
    for i in range(n):
        i_prev = (i - 1) % n
        i_next = (i + 1) % n
        
        v_prev = unit_vectors[i_prev]
        v_curr = unit_vectors[i]
        v_next = unit_vectors[i_next]
        
        dot_prev = np.clip(np.dot(v_prev, v_curr), -1.0, 1.0)
        dot_next = np.clip(np.dot(v_curr, v_next), -1.0, 1.0)
        
        alpha_prev = np.arccos(dot_prev)
        alpha_next = np.arccos(dot_next)
        
        tan_alpha_prev_half = np.tan(alpha_prev / 2.0)
        tan_alpha_next_half = np.tan(alpha_next / 2.0)
        
        weights[i] = (tan_alpha_prev_half + tan_alpha_next_half) / distances[i]
    
    total_weight = np.sum(weights)
    if total_weight > epsilon:
        weights /= total_weight
    else:
        weights = 1.0 / (distances + epsilon)
        weights /= np.sum(weights)
    
    return weights

def compute_source_mvc_weights(
    add_info: dict = None,
    grid_density: int = 60,
    boundary_margin: float = 0.2,
    distance_factor: float = 0.9,
    max_fps_points: int = 25):
    
    max_fps_points *= 2
    
    if add_info is None:
        raise NotImplementedError("add_info not provided")
    else:
        print("Using pre-computed add_info from edge detection")
        aligned_points = add_info['aligned_points']

    # Extract boundary polygon
    boundary_polygon = extract_leaf_boundary_polygon(add_info)
    
    all_interior_points = generate_interior_sample_points(
        boundary_polygon, 
        grid_density, 
        boundary_margin, 
        distance_factor, 
        use_fps=False  
    )
    
    if len(all_interior_points) > max_fps_points:
        fps_indices = fpsample.bucket_fps_kdline_sampling(all_interior_points, max_fps_points, h=5)
        interior_sample_points = all_interior_points[fps_indices]

        if len(interior_sample_points) > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(interior_sample_points)
            min_dist = np.min(distances)
            mean_dist = np.mean(distances)

            if min_dist < mean_dist * 0.3: 
                stricter_fps_points = max(max_fps_points // 2, 10)  
                fps_indices = fpsample.bucket_fps_kdline_sampling(all_interior_points, stricter_fps_points, h=5)
                interior_sample_points = all_interior_points[fps_indices]

                if len(interior_sample_points) > 1:
                    distances = pdist(interior_sample_points)
                    min_dist = np.min(distances)
                    mean_dist = np.mean(distances)
    else:
        interior_sample_points = all_interior_points
    
    mvc_coordinates_list = []
    mvc_indices_list = []
    aligned_points_2d = aligned_points[:, :2].cpu().numpy() if hasattr(aligned_points, 'cpu') else aligned_points[:, :2]
    
    for i, sample_point in enumerate(interior_sample_points):
        mvc_weights = compute_mean_value_coordinates(sample_point, boundary_polygon)
        mvc_coordinates_list.append(mvc_weights)
        
        distances = np.linalg.norm(aligned_points_2d - sample_point, axis=1)
        nearest_gaussian_idx = np.argmin(distances)
        mvc_indices_list.append(nearest_gaussian_idx)
    
    mvc_coordinates = np.array(mvc_coordinates_list)
    
    return mvc_coordinates, interior_sample_points

def reconstruct_target_coordinates_from_weights(target_boundary_polygon, source_mvc_weights):

    target_coordinates_2d = []
    
    for i, weights in enumerate(source_mvc_weights):
        if len(weights) == len(target_boundary_polygon) and np.sum(np.abs(weights)) > 1e-6:
            normalized_weights = weights / np.sum(weights)
            
            reconstructed_point = np.sum(normalized_weights.reshape(-1, 1) * target_boundary_polygon, axis=0)
            target_coordinates_2d.append(reconstructed_point)

    target_coordinates_2d = np.array(target_coordinates_2d)
    
    return target_coordinates_2d

def compute_3d_edge_points_from_gaussian(
    gaussian: GaussianData,
    labels: np.ndarray, 
    image_size: int,
    tip_point: np.ndarray,
    base_point: np.ndarray,
    root_point: np.ndarray = None,
    num_samples_per_path: int = 20,
    debug: bool = False):
    
    print(f"🚀 STARTING compute_3d_edge_points_from_gaussian with num_samples_per_path={num_samples_per_path}")
    print(f"   tip_point: {tip_point}")
    print(f"   base_point: {base_point}")
    points = gaussian.xyz
    scales = gaussian.scale  
    rots = gaussian.rot
    opacities = gaussian.opacity
    shs = gaussian.sh
    
    points = torch.from_numpy(points).cuda().float()
    scales = torch.from_numpy(scales).cuda().float()
    rots = torch.from_numpy(rots).cuda().float()
    opacities = torch.from_numpy(opacities).cuda().float()
    shs = torch.from_numpy(shs).cuda().float()
    
    front_texture, transformation_matrix, x_range, y_range, aligned_points, _, depth_map = render_to_pca(
        points, scales, rots, opacities, shs, 
        labels, 
        tip_point,
        base_point,
        root_point=root_point,
        image_size=image_size, view_side="front", uv_rendering=True, black_background=True
    )
    

    structured_points_3d_indices, add_info = compute_3d_edge_points(
        front_texture, 
        aligned_points, 
        x_range, y_range, image_size, 
        tip_point, 
        base_point, 
        transformation_matrix, 
        num_samples_per_path
    )
    
    if debug:
        structured_points_3d_points, add_info = compute_3d_edge_points_kai(
            front_texture,
            depth_map,
            aligned_points,
            x_range,
            y_range,
            image_size,
            tip_point,
            base_point,
            transformation_matrix,
            num_samples_per_path
        )

        
    add_info['x_range'] = x_range
    add_info['y_range'] = y_range
    add_info['image_size'] = image_size
    add_info['transformation_matrix'] = transformation_matrix
    add_info['aligned_points'] = aligned_points
    add_info['front_texture'] = front_texture
    add_info['depth_map'] = depth_map
    if debug:
        return structured_points_3d_points, add_info
    else:
        return structured_points_3d_indices, add_info

def sigma2d_from_gaussians(xyz_aligned: np.ndarray,
                           scales: np.ndarray,
                           rot_wxyz_aligned: np.ndarray,
                           mod: float = 1.0):

    N = xyz_aligned.shape[0]
    if N == 0:
        return np.zeros((0,2), np.float32), np.zeros((0,2,2), np.float32)

    center_xy = xyz_aligned[:, :2].astype(np.float32)
    Rm = quaternion_wxyz_to_matrix(rot_wxyz_aligned)          # (N,3,3)
    s  = np.sqrt(np.float32(mod)) * scales.astype(np.float32)  # (N,3)

    B = Rm * s[:, None, :]                                # (N,3,3)
    Sigma3D = B @ np.transpose(B, (0, 2, 1))              # (N,3,3)
    Sigma2D = Sigma3D[:, :2, :2]                          # (N,2,2)
    return center_xy, Sigma2D

def bbox_ranges_aabb_diag(xyz_aligned: np.ndarray,
                          scales: np.ndarray,
                          rot_wxyz_aligned: np.ndarray,
                          k_sigma: float = 3.0,
                          mod: float = 1.0):
    center_xy, Sigma2D = sigma2d_from_gaussians(xyz_aligned, scales, rot_wxyz_aligned, mod)
    if center_xy.shape[0] == 0:
        return (0.0, 0.0), (0.0, 0.0), {"rx": None, "ry": None}

    var_x = np.clip(Sigma2D[:, 0, 0], 0.0, None)
    var_y = np.clip(Sigma2D[:, 1, 1], 0.0, None)
    rx = k_sigma * np.sqrt(var_x)
    ry = k_sigma * np.sqrt(var_y)

    x = center_xy[:, 0]
    y = center_xy[:, 1]
    xmin = float(np.min(x - rx))
    xmax = float(np.max(x + rx))
    ymin = float(np.min(y - ry))
    ymax = float(np.max(y + ry))
    return (xmin, xmax), (ymin, ymax), {"rx": rx, "ry": ry}

def bbox_ranges_circle_eigmax(xyz_aligned: np.ndarray,
                              scales: np.ndarray,
                              rot_wxyz_aligned: np.ndarray,
                              k_sigma: float = 3.0,
                              mod: float = 1.0):
    center_xy, Sigma2D = sigma2d_from_gaussians(xyz_aligned, scales, rot_wxyz_aligned, mod)
    if center_xy.shape[0] == 0:
        return (0.0, 0.0), (0.0, 0.0), {"r": None}

    evals = np.linalg.eigvalsh(Sigma2D)             
    lam_max = np.clip(evals[:, 1], 0.0, None)
    r = k_sigma * np.sqrt(lam_max)                  
    x = center_xy[:, 0]; y = center_xy[:, 1]
    xmin = float(np.min(x - r))
    xmax = float(np.max(x + r))
    ymin = float(np.min(y - r))
    ymax = float(np.max(y + r))
    return (xmin, xmax), (ymin, ymax), {"r": r}
   
def generate_uv_mapping_mem(gaussian: GaussianData, 
                            tip_point: np.ndarray,
                            base_point: np.ndarray,
                            mesh_vertices: np.ndarray, 
                            triangles: np.ndarray, 
                            normals: np.ndarray,
                            image_size: int = 1024,
                            root_point: np.ndarray = None):
    root_point = np.asarray(root_point) if root_point is not None else None
    print("Rendering to MEM ############################### GIVEN ROOT POINT:", root_point)
    points = gaussian.xyz
    scales = gaussian.scale  
    rots = gaussian.rot
    opacities = gaussian.opacity
    shs = gaussian.sh
    
    points = torch.from_numpy(points).cuda().float()
    scales = torch.from_numpy(scales).cuda().float()
    rots = torch.from_numpy(rots).cuda().float()
    opacities = torch.from_numpy(opacities).cuda().float()
    shs = torch.from_numpy(shs).cuda().float()
    front_texture, transformation_matrix_, _, _, _, _, _ = render_to_pca(
        points, scales, rots, opacities, shs, 
        None, 
        tip_point,
        base_point,
        root_point=root_point,
        image_size=image_size, view_side="front", uv_rendering=True, black_background=True, 
    )
    back_texture, transformation_matrix, x_range, y_range, _, _, _ = render_to_pca(
        points, scales, rots, opacities, shs, 
        None, 
        tip_point,
        base_point,
        root_point=root_point,
        image_size=image_size, view_side="back", uv_rendering=True, black_background=True, 
    )
    
    double_texture = create_double_sided_texture(front_texture, back_texture)
    if double_texture.dtype != np.uint8:
        double_texture = (double_texture * 255).astype(np.uint8)
    # show image
    Image.fromarray(double_texture).save(f'./double_sided_texture_.png')

    # print(transformation_matrix_, transformation_matrix)

    double_uv_coords, new_vertices, new_faces, new_normals = create_double_sided_uv_mapping(
        mesh_vertices, triangles, root_point, transformation_matrix, x_range, y_range
    )
    
    mesh_data = {
        "vertices": new_vertices,
        "faces": new_faces,
        "normals": new_normals,
        "uvs": double_uv_coords,
    }


    output_path = f'./template_leaf_.obj'
    mtl_path = output_path.replace('.obj', '.mtl')
    with open(mtl_path, 'w') as f:
        f.write("newmtl leaf_material\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write(f"map_Kd double_sided_texture_.png\n")

    V2 = new_vertices.shape[0]
    assert V2 == double_uv_coords.shape[0] == new_normals.shape[0]

    with open(output_path, 'w') as f:
        f.write(f"mtllib template_leaf_.mtl\n")
        f.write("usemtl leaf_material\n\n")

        for v in new_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for uv in double_uv_coords:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        for n in new_normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        for tri in new_faces:
            a, b, c = (tri + 1).tolist()   # OBJ 1-based
            f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
    
    return double_uv_coords, double_texture, mesh_data

def generate_uv_mapping_disk(gaussian: GaussianData, 
                             tip_point: np.ndarray,
                             base_point: np.ndarray,
                             mesh_vertices: np.ndarray, 
                             triangles: np.ndarray,
                            #  normals: np.ndarray,
                             image_size: int = 1024,
                             save_path: str = None):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    print("Rendering to DISK ###############################")
    save_gaussian_data_as_ply(f'{save_path}/template_leaf.ply', gaussian)
    points = gaussian.xyz
    scales = gaussian.scale
    rots = gaussian.rot
    opacities = gaussian.opacity
    shs = gaussian.sh
    
    points = torch.from_numpy(points).cuda().float()
    scales = torch.from_numpy(scales).cuda().float()
    rots = torch.from_numpy(rots).cuda().float()
    opacities = torch.from_numpy(opacities).cuda().float()
    shs = torch.from_numpy(shs).cuda().float()
    
    front_texture, transformation_matrix_, _, _, _, _, _ = render_to_pca(
        points, scales, rots, opacities, shs, None, tip_point, base_point, 
        image_size=1024, view_side="back", black_background=True, uv_rendering=True
    )

    back_texture, transformation_matrix, x_range, y_range, _, _, _ = render_to_pca(
        points, scales, rots, opacities, shs, 
        None, 
        tip_point, 
        base_point, 
        image_size=1024, view_side="front", black_background=True, uv_rendering=True
    )

    double_texture = create_double_sided_texture(front_texture, back_texture)
    if double_texture.dtype != np.uint8:
        double_texture = (double_texture * 255).astype(np.uint8)
    Image.fromarray(double_texture).save(f'{save_path}/double_sided_texture.png')

    double_uv_coords, new_vertices, new_faces, new_normals = create_double_sided_uv_mapping(
        mesh_vertices, triangles, transformation_matrix, x_range, y_range
    )
    

    output_path = f'{save_path}/template_leaf.obj'
    mtl_path = output_path.replace('.obj', '.mtl')

    with open(mtl_path, 'w') as f:
        f.write("newmtl leaf_material\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write("map_Kd double_sided_texture.png\n")

    V2 = new_vertices.shape[0]
    assert V2 == double_uv_coords.shape[0] == new_normals.shape[0]

    with open(output_path, 'w') as f:
        f.write("mtllib template_leaf.mtl\n")
        f.write("usemtl leaf_material\n\n")

        for v in new_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for uv in double_uv_coords:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        for n in new_normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        for tri in new_faces:
            a, b, c = (tri + 1).tolist()   # OBJ 1-based
            f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
    
    return output_path