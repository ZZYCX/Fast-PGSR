import numpy as np
from scipy.spatial import cKDTree
import torch


def export_template_mesh(template_mesh):
    if template_mesh.get("mesh_data") is None:
        return
    
    mesh_data = template_mesh["mesh_data"]
    mesh_name = template_mesh["name"]
    
    import os
    output_dir = "exported_meshes"
    os.makedirs(output_dir, exist_ok=True)
    
    obj_path = os.path.join(output_dir, f"{mesh_name}.obj")
    mtl_path = os.path.join(output_dir, f"{mesh_name}.mtl")
    texture_path = os.path.join(output_dir, f"{mesh_name}_texture.png")
    
    if hasattr(mesh_data, 'texture_data') and mesh_data.texture_data is not None:
        from PIL import Image
        texture_img = Image.fromarray(mesh_data.texture_data)
        texture_img.save(texture_path)
    with open(mtl_path, 'w') as f:
        f.write("# Template leaf material\n")
        f.write(f"newmtl {mesh_name}_material\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        if hasattr(mesh_data, 'texture_data') and mesh_data.texture_data is not None:
            f.write(f"map_Kd {mesh_name}_texture.png\n")
    
    with open(obj_path, 'w') as f:
        f.write(f"# Template leaf mesh: {mesh_name}\n")
        f.write(f"mtllib {mesh_name}.mtl\n")
        f.write(f"usemtl {mesh_name}_material\n\n")
        
        for v in mesh_data.vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        for n in mesh_data.normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        
        if mesh_data.uvs is not None:
            for uv in mesh_data.uvs:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        
        f.write("\n")
        
        for face in mesh_data.faces:
            if mesh_data.uvs is not None:
                f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}/{face[2]+1}\n")
            else:
                f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")

def write_mesh_to_disk(save_path, mesh_data):
    new_vertices = mesh_data.vertices
    new_faces = mesh_data.faces
    double_uv_coords = mesh_data.uvs
    new_normals = mesh_data.normals

    output_path = f'{save_path}/template_leaf.obj'
    mtl_path = output_path.replace('.obj', '.mtl')

    with open(mtl_path, 'w') as f:
        f.write("newmtl leaf_material\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write(f"map_Kd ../double_sided_texture.png\n")

    # Consistency check
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

        # v/vt/vn share same indices
        for tri in new_faces:
            a, b, c = (tri + 1).tolist()   # OBJ 1-based
            f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")

def chamfer_distance(pcd1, pcd2):
    if isinstance(pcd1, np.ndarray) and isinstance(pcd2, np.ndarray):
        pass
    elif isinstance(pcd1, list) and isinstance(pcd2, list):
        # Check if pcd1 and pcd2 are lists of numpy arrays
        pcd1 = np.array(pcd1)
        pcd2 = np.array(pcd2)
    elif isinstance(pcd1, torch.Tensor) and isinstance(pcd2, torch.Tensor):
        # Convert PyTorch tensors to numpy arrays
        pcd1 = pcd1.cpu().numpy()
        pcd2 = pcd2.cpu().numpy()

    tree1 = cKDTree(pcd1)
    tree2 = cKDTree(pcd2)

    dist1, _ = tree1.query(pcd2)  # pcd2 -> pcd1
    dist2, _ = tree2.query(pcd1)  # pcd1 -> pcd2

    cd = np.mean(dist1 ** 2) + np.mean(dist2 ** 2)
    return cd

def mls_transform_pytorch_rot(
    source_points,
    source_rots,
    source_corr_points,
    target_corr_points,
    sigma = 0.1,
    corr_weights=None):
    
    # Compute distances from all points to all control points (N×K)
    distances_sq = torch.cdist(source_points, source_corr_points, p=2)**2
    # Compute distance-based weight matrix (N×K)
    distance_weights = torch.exp(-distances_sq / (2 * sigma**2))
    # If learnable correspondence weights provided, multiply with distance weights
    if corr_weights is not None:
        # corr_weights shape: (K, 1) -> expand to (N, K)
        learnable_weights = corr_weights.squeeze(-1).unsqueeze(0).expand(distance_weights.shape[0], -1)
        weights = distance_weights * learnable_weights
    else:
        weights = distance_weights
    
    sum_weights = torch.sum(weights, dim=1, keepdim=True)  # (N×1)
    
    # Handle zero weight cases
    valid_mask = (sum_weights.squeeze() > 1e-10)
    
    # Normalize weights
    weights_normalized = weights / (sum_weights + 1e-10)  # (N×K)
    
    # Compute centroids (N×3)
    P_centroid = torch.sum(weights_normalized.unsqueeze(-1) * source_corr_points.unsqueeze(0), dim=1)  # (N×3)
    Q_centroid = torch.sum(weights_normalized.unsqueeze(-1) * target_corr_points.unsqueeze(0), dim=1)  # (N×3)
    
    # Compute offset vectors (N×K×3)
    P_prime = source_corr_points.unsqueeze(0) - P_centroid.unsqueeze(1)  # (N×K×3)
    Q_prime = target_corr_points.unsqueeze(0) - Q_centroid.unsqueeze(1)  # (N×K×3)
    
    # Compute M matrix (N×3×3)
    # weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # (N×K×1×1)
    P_prime_weighted = P_prime * weights.unsqueeze(-1)  # (N×K×3)
    M = torch.bmm(P_prime_weighted.transpose(-2, -1), P_prime)  # (N×3×3)
    
    # Compute B matrix (N×3×3)
    Q_prime_weighted = Q_prime * weights.unsqueeze(-1)  # (N×K×3)
    B = torch.bmm(Q_prime_weighted.transpose(-2, -1), P_prime)  # (N×3×3)
    
    # Batch inverse (N×3×3)
    try:
        M_inv = torch.linalg.inv(M)
    except:
        M_inv = torch.linalg.pinv(M)
    
    # Compute affine transformation matrix A (N×3×3)
    A = torch.bmm(B, M_inv)  # (N×3×3)
    
    # Compute translation vector t (N×3)
    t = Q_centroid - torch.bmm(A, P_centroid.unsqueeze(-1)).squeeze(-1)  # (N×3)
    
    # Transform points (N×3)
    S_transformed = torch.bmm(A, source_points.unsqueeze(-1)).squeeze(-1) + t  # (N×3)
    
    # Extract rotation matrix and convert to quaternion
    U, S_svd, Vh = torch.linalg.svd(A)  # (N×3×3)
    R_matrix = torch.bmm(U, Vh)  # (N×3×3)
    
    # Ensure orthogonal matrix (det = 1)
    det_R = torch.det(R_matrix)  # (N,)
    flip_mask = det_R < 0
    if flip_mask.any():
        Vh_corrected = Vh.clone()
        Vh_corrected[flip_mask, -1, :] *= -1
        R_matrix[flip_mask] = torch.bmm(U[flip_mask], Vh_corrected[flip_mask])
    
    # Convert rotation matrix to quaternion (inline)
    N = R_matrix.shape[0]
    R_matrix = R_matrix.contiguous()
    m00, m01, m02 = R_matrix[:, 0, 0], R_matrix[:, 0, 1], R_matrix[:, 0, 2]
    m10, m11, m12 = R_matrix[:, 1, 0], R_matrix[:, 1, 1], R_matrix[:, 1, 2]
    m20, m21, m22 = R_matrix[:, 2, 0], R_matrix[:, 2, 1], R_matrix[:, 2, 2]

    trace = m00 + m11 + m22
    R_quat = torch.zeros((N, 4), dtype=R_matrix.dtype, device=R_matrix.device)

    mask0 = trace > 0
    t0 = torch.sqrt(trace[mask0] + 1.0) * 2.0
    R_quat[mask0, 0] = 0.25 * t0
    R_quat[mask0, 1] = (m21[mask0] - m12[mask0]) / t0
    R_quat[mask0, 2] = (m02[mask0] - m20[mask0]) / t0
    R_quat[mask0, 3] = (m10[mask0] - m01[mask0]) / t0

    mask1 = (~mask0) & (m00 >= m11) & (m00 >= m22)
    t1 = torch.sqrt(1.0 + m00[mask1] - m11[mask1] - m22[mask1]) * 2.0
    R_quat[mask1, 0] = (m21[mask1] - m12[mask1]) / t1
    R_quat[mask1, 1] = 0.25 * t1
    R_quat[mask1, 2] = (m01[mask1] + m10[mask1]) / t1
    R_quat[mask1, 3] = (m02[mask1] + m20[mask1]) / t1

    mask2 = (~mask0) & (~mask1) & (m11 >= m22)
    t2 = torch.sqrt(1.0 + m11[mask2] - m00[mask2] - m22[mask2]) * 2.0
    R_quat[mask2, 0] = (m02[mask2] - m20[mask2]) / t2
    R_quat[mask2, 1] = (m01[mask2] + m10[mask2]) / t2
    R_quat[mask2, 2] = 0.25 * t2
    R_quat[mask2, 3] = (m12[mask2] + m21[mask2]) / t2

    mask3 = (~mask0) & (~mask1) & (~mask2)
    t3 = torch.sqrt(1.0 + m22[mask3] - m00[mask3] - m11[mask3]) * 2.0
    R_quat[mask3, 0] = (m10[mask3] - m01[mask3]) / t3
    R_quat[mask3, 1] = (m02[mask3] + m20[mask3]) / t3
    R_quat[mask3, 2] = (m12[mask3] + m21[mask3]) / t3
    R_quat[mask3, 3] = 0.25 * t3

    R_quat = R_quat / torch.norm(R_quat, dim=-1, keepdim=True).clamp_min(1e-12)
    
    # Quaternion multiplication (inline Hamilton product)
    w1, x1, y1, z1 = R_quat.unbind(-1)
    w2, x2, y2, z2 = source_rots.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    combined_quat = torch.stack([w, x, y, z], dim=-1)
    combined_quat = combined_quat / torch.norm(combined_quat, dim=-1, keepdim=True).clamp_min(1e-12)
    
    R_transformed = combined_quat / torch.norm(combined_quat, dim=-1, keepdim=True)
    
    # Handle points with invalid weights
    S_transformed[~valid_mask] = source_points[~valid_mask]
    R_transformed[~valid_mask] = source_rots[~valid_mask]
    
    return S_transformed, R_transformed