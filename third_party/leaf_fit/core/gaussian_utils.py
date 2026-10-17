from typing import List, Tuple, Dict, Any
from plyfile import PlyData, PlyElement
from dataclasses import dataclass
from e3nn import o3
import numpy as np
import torch

def pack_for_gpu(
    template_mesh: Any,
    groups: List[Tuple[np.ndarray, np.ndarray]],  # Length N; each item (src:(C,3), dst:(C,3)), C is fixed within this run
) -> Dict[str, Any]:
    # Extract and normalize mesh basic data
    template_mesh = template_mesh["mesh_data"]
    vertices = np.asarray(template_mesh.vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices expected (V,3), got {vertices.shape}")

    # indices / faces (optional)
    indices = getattr(template_mesh, "faces", None)
    if indices is None:
        indices = getattr(template_mesh, "indices", None)
    if indices is not None:
        indices = np.asarray(indices, dtype=np.uint32)

    # normals / colors / uvs (optional)
    normals = getattr(template_mesh, "normals", None)
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32)
        if normals.shape[0] != vertices.shape[0]:
            raise ValueError("normals and vertices count mismatch")

    colors = getattr(template_mesh, "colors", None)
    if colors is not None:
        colors = np.asarray(colors)
        if colors.dtype == np.uint8:
            pass
        else:
            colors = colors.astype(np.float32, copy=False)
        if colors.ndim != 2 or colors.shape[1] not in (3, 4):
            raise ValueError("colors expected (V,3) or (V,4)")
        if colors.shape[0] != vertices.shape[0]:
            raise ValueError("colors and vertices count mismatch")

    uvs = getattr(template_mesh, "uvs", None)
    if uvs is not None:
        uvs = np.asarray(uvs, dtype=np.float32)
        if uvs.ndim != 2 or uvs.shape[1] != 2:
            raise ValueError("uvs expected (V,2)")
        if uvs.shape[0] != vertices.shape[0]:
            raise ValueError("uvs and vertices count mismatch")

    # texture (optional): supports ndarray or bytes
    texture_data = getattr(template_mesh, "texture_data", None)
    if texture_data is not None:
        if isinstance(texture_data, (bytes, bytearray)):
            texture_data = np.frombuffer(texture_data, dtype=np.uint8)
        else:
            texture_data = np.asarray(texture_data)
            if texture_data.dtype != np.uint8:
                texture_data = texture_data.astype(np.uint8, copy=False)

    # Validate groups, extract N/C, and pack pairPool
    N = len(groups)
    if N == 0:
        raise ValueError("groups is empty")

    src0, dst0 = groups[0]
    if src0.shape != dst0.shape or src0.ndim != 2 or src0.shape[1] != 3:
        raise ValueError("groups[0] must be (C,3)")
    C = int(src0.shape[0])

    for g, (src, dst) in enumerate(groups):
        if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
            raise ValueError(f"[g={g}] must be (C,3)")
        if int(src.shape[0]) != C:
            raise ValueError(f"[g={g}] C={src.shape[0]} inconsistent with global C={C}")

    # pairPool: order [g][c], each pair occupies 2 vec4 (src.xyz0, dst.xyz0)
    blocks = []
    if C > 0:
        for g in range(N):
            src, dst = groups[g]
            src = src.astype(np.float32, copy=False)
            dst = dst.astype(np.float32, copy=False)
            src4 = np.concatenate([src, np.zeros((C,1), np.float32)], axis=1)
            dst4 = np.concatenate([dst, np.zeros((C,1), np.float32)], axis=1)
            inter = np.empty((C*2, 4), dtype=np.float32)
            inter[0::2] = src4
            inter[1::2] = dst4
            blocks.append(inter)
        pairPool = np.concatenate(blocks, axis=0)
    else:
        pairPool = np.zeros((0,4), np.float32)

    # Organize output fields: only write existing keys
    out: Dict[str, Any] = {
        "vertices": vertices,                # (V,3) float32
        "N": np.array([N], dtype=np.int32),  # number of groups
        "C": np.array([C], dtype=np.int32),  # number of pairs per group (fixed in this run)
        "pairPool": pairPool,                # (2*N*C,4) float32
    }
    if indices is not None:      out["indices"] = indices
    if normals is not None:      out["normals"] = normals
    if colors is not None:       out["colors"] = colors
    if uvs is not None:          out["uvs"] = uvs
    if texture_data is not None: out["texture_data"] = texture_data

    return out

@dataclass
class GaussianData:
    xyz: np.ndarray
    rot: np.ndarray
    scale: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray
    nxnynz: np.ndarray
    filter_3Ds: np.ndarray
    
    def flat(self) -> np.ndarray:
        ret = np.concatenate([self.xyz, self.rot, self.scale, self.opacity, self.sh, self.nxnynz, self.filter_3Ds], axis=-1)
        return np.ascontiguousarray(ret)
    
    def __len__(self):
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        return self.sh.shape[-1]

def save_gaussian_data_as_ply(path: str, gau_data: GaussianData):
    """Save Gaussian data as PLY file"""
    num_gaussians = len(gau_data)
    
    # Apply inverse activation functions
    xyz = gau_data.xyz.astype(np.float32)
    nxnynz = gau_data.nxnynz.astype(np.float32)
    rot = gau_data.rot.astype(np.float32)

    epsilon_opacity = 1e-7
    opacity_clipped = np.clip(gau_data.opacity, epsilon_opacity, 1.0 - epsilon_opacity)
    opacities_for_ply = -np.log(1.0 / opacity_clipped - 1.0).astype(np.float32)

    epsilon_scale = 1e-8
    scales_clipped = np.maximum(gau_data.scale, epsilon_scale)
    scales_for_ply = np.log(scales_clipped).astype(np.float32)
    
    sh = gau_data.sh.astype(np.float32)
    filter_3Ds = gau_data.filter_3Ds.astype(np.float32)
    if num_gaussians > 0:
        f_dc_ply = sh[:, 0:3]
        f_rest_ply = sh[:, 3:]
        
        # De-interleave the f_rest data to save in standard non-interleaved format
        num_rest_coeffs_per_color = f_rest_ply.shape[1] // 3
        f_rest_reshaped = f_rest_ply.reshape(num_gaussians, num_rest_coeffs_per_color, 3)
        f_rest_transposed = f_rest_reshaped.transpose(0, 2, 1)
        f_rest_for_ply = f_rest_transposed.reshape(num_gaussians, -1)
    else:
        f_dc_ply = np.zeros((0, 3), dtype=np.float32)
        f_rest_for_ply = np.zeros((0,0), dtype=np.float32)

    num_f_rest_coeffs = f_rest_for_ply.shape[1]

    # Define PLY element properties
    property_dtype_list = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
    ]
    for i in range(num_f_rest_coeffs):
        property_dtype_list.append((f'f_rest_{i}', 'f4'))
    property_dtype_list.extend([
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
        ('filter_3D', 'f4')
    ])

    # Create structured NumPy array
    elements_data = np.empty(num_gaussians, dtype=property_dtype_list)
    
    if num_gaussians > 0:
        elements_data['x'] = xyz[:, 0]
        elements_data['y'] = xyz[:, 1]
        elements_data['z'] = xyz[:, 2]
        elements_data['nx'] = nxnynz[:, 0]
        elements_data['ny'] = nxnynz[:, 1]
        elements_data['nz'] = nxnynz[:, 2]
        elements_data['f_dc_0'] = f_dc_ply[:, 0]
        elements_data['f_dc_1'] = f_dc_ply[:, 1]
        elements_data['f_dc_2'] = f_dc_ply[:, 2]
        for i in range(num_f_rest_coeffs):
            elements_data[f'f_rest_{i}'] = f_rest_for_ply[:, i]
        elements_data['opacity'] = opacities_for_ply.reshape(-1)
        elements_data['scale_0'] = scales_for_ply[:, 0]
        elements_data['scale_1'] = scales_for_ply[:, 1]
        elements_data['scale_2'] = scales_for_ply[:, 2]
        elements_data['rot_0'] = rot[:, 0]
        elements_data['rot_1'] = rot[:, 1]
        elements_data['rot_2'] = rot[:, 2]
        elements_data['rot_3'] = rot[:, 3]
        elements_data['filter_3D'] = filter_3Ds.reshape(-1)

    # Write to PLY file
    vertex_element = PlyElement.describe(elements_data, 'vertex')
    PlyData([vertex_element], text=False).write(path)

def apply_indices_to_gaussian_data(gau_data: GaussianData, indices: np.ndarray):
    return GaussianData(
        xyz=gau_data.xyz[indices],
        rot=gau_data.rot[indices],
        scale=gau_data.scale[indices],
        opacity=gau_data.opacity[indices],
        sh=gau_data.sh[indices],
        nxnynz=gau_data.nxnynz[indices],
        filter_3Ds=gau_data.filter_3Ds[indices]
    )

def _is_torch(x):
    return isinstance(x, torch.Tensor)

def _ensure_torch(x, device=None, dtype=torch.float32):
    if _is_torch(x):
        return x.to(device or x.device, dtype=dtype)
    return torch.from_numpy(np.asarray(x)).to(device or 'cpu', dtype=dtype)

def apply_transformation_matrix_to_points(points, transformation_matrix):
    """Apply transformation matrix to points only (NumPy version)"""
    # Convert inputs to numpy if needed
    if hasattr(points, 'cpu'):
        points = points.detach().cpu().numpy()
    if hasattr(transformation_matrix, 'cpu'):
        transformation_matrix = transformation_matrix.detach().cpu().numpy()
    
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    aligned_points = (transformation_matrix @ points_homo.T).T[:, :3]
    return aligned_points

def apply_transformation_matrix_to_points_torch(points, transformation_matrix):
    points_homo = torch.cat([points, torch.ones((points.shape[0], 1), device=points.device)], dim=1)
    aligned_points = (transformation_matrix @ points_homo.T).T[:, :3]
    return aligned_points

def matrix_to_quaternion_wxyz(R):
    """
    R: 3x3 rotation matrix (numpy or torch). Returns [w,x,y,z] quaternion.
    - If input is numpy, returns numpy(float32)
    - If input is torch, returns torch(float32) and keeps input device
    """
    if _is_torch(R):
        dev = R.device
        m = R.detach().to('cpu', dtype=torch.float64).numpy()
        use_torch_out = True
    else:
        m = np.asarray(R, dtype=np.float64)
        use_torch_out = False

    t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        i = np.argmax([m[0, 0], m[1, 1], m[2, 2]])
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 - m[0, 0] + m[1, 1] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 - m[0, 0] - m[1, 1] + m[2, 2]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    if q[0] < 0:
        q = -q
    q = q.astype(np.float32)

    if use_torch_out:
        return torch.from_numpy(q).to(dev)
    return q

def quat_multiply(q_old_wxyz, q_delta_wxyz):
    """
    Returns q_new = q_delta ⊗ q_old (left multiply).
    - Supports numpy or torch input:
        * If q_old is numpy, returns numpy(float32)
        * If q_old is torch, returns torch(float32) and keeps q_old.device
    - Shape:
        * q_old: (4,) or (N,4)
        * q_delta: (4,) (auto broadcast to N)
    """
    # Determine output type & device based on q_old's type
    if _is_torch(q_old_wxyz):
        q0 = q_old_wxyz
        dev = q0.device
        q1 = q_delta_wxyz if _is_torch(q_delta_wxyz) else torch.from_numpy(np.asarray(q_delta_wxyz))
        q0 = q0.to(dev, dtype=torch.float32)
        q1 = q1.to(dev, dtype=torch.float32)
        out_as_torch = True
    else:
        dev = 'cpu'
        q0 = _ensure_torch(q_old_wxyz, device=dev)
        q1 = _ensure_torch(q_delta_wxyz, device=dev)
        out_as_torch = False

    if q0.ndim == 1:
        q0 = q0.unsqueeze(0)  # (1,4)
    q1 = q1.reshape(1, 4).expand(q0.shape[0], -1)  # (N,4)

    w0, x0, y0, z0 = torch.split(q0, 1, dim=-1)
    w1, x1, y1, z1 = torch.split(q1, 1, dim=-1)
    out = torch.cat((
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
    ), dim=-1)
    out = torch.nn.functional.normalize(out, dim=-1, eps=1e-12)

    if out_as_torch:
        return out
    return out.detach().cpu().numpy().astype(np.float32)

def sh_rotate(sh_in, R_in):
    """
    Rotate interleaved SH coefficients:
      - Input layout: (N, 3*(L+1)^2), interleaved [coef_k_R, coef_k_G, coef_k_B, ...]
      - R_in: (3,3) rotation matrix
      - Axis permutation: P^{-1} R P
      - Euler angles: ZYZ, using (alpha, -beta, gamma)
      - Only rotate l>=1, l=0(DC) unchanged
      - Left multiply: D_l @ coeffs^T
    Returns:
      - If sh_in is torch.Tensor: returns torch.Tensor, same device as sh_in
      - If sh_in is numpy.ndarray: returns numpy.ndarray (CPU)
    """
    is_torch_in = isinstance(sh_in, torch.Tensor)
    # Normalize input to numpy for shape checking and Lmax calculation
    sh_np = sh_in.detach().cpu().numpy() if is_torch_in else np.asarray(sh_in)
    R_np  = R_in.detach().cpu().numpy()  if isinstance(R_in, torch.Tensor) else np.asarray(R_in)

    assert sh_np.ndim == 2 and sh_np.shape[1] % 3 == 0, "sh shape must be (N, 3*(L+1)^2)"
    N, threeM = sh_np.shape
    M = threeM // 3
    L = int(np.sqrt(M) + 1e-6) - 1
    if L < 1:
        # No first order, return as is (matching input type)
        return sh_in.clone() if is_torch_in else sh_np.astype(np.float32)

    # Determine compute device based on sh_in's device
    if is_torch_in:
        calc_dev = sh_in.device
        out_torch = True
    else:
        calc_dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        out_torch = False

    dtype = torch.float32

    # (N,3M) -> (N,M,3) to compute device
    sh = torch.from_numpy(sh_np).to(calc_dev, dtype) if not is_torch_in else sh_in.to(calc_dev, dtype)
    sh_M3 = sh.view(N, M, 3).contiguous()  # (N, M, 3)

    # Construct angles for P^{-1} R P on CPU, then move D_l to calc_dev 
    P_cpu = torch.tensor([[0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]], dtype=dtype, device='cpu')
    R_cpu = torch.from_numpy(R_np).to('cpu', dtype)
    R_eff = torch.linalg.inv(P_cpu) @ R_cpu @ P_cpu

    # ZYZ Euler angles, (alpha, -beta, gamma)
    alpha, beta, gamma = o3._rotation.matrix_to_angles(R_eff)  # CPU
    D1 = o3.wigner_D(1, alpha, -beta, gamma).to(calc_dev)
    D2 = o3.wigner_D(2, alpha, -beta, gamma).to(calc_dev) if L >= 2 else None
    D3 = o3.wigner_D(3, alpha, -beta, gamma).to(calc_dev) if L >= 3 else None

    # Channel splitting
    R_ch = sh_M3[:, :, 0].clone()  # (N,M)
    G_ch = sh_M3[:, :, 1].clone()
    B_ch = sh_M3[:, :, 2].clone()

    # Order offsets (ACN): l=0:1, l=1:3, l=2:5, l=3:7, ...
    off = 0
    off += 1  # skip l=0
    s1, e1 = off, off + 3
    off = e1
    s2, e2 = off, off + 5
    off = e2
    s3, e3 = off, off + 7

    def rot_channel(chNM: torch.Tensor) -> torch.Tensor:
        # l=1
        blk = chNM[:, s1:e1].T               # (3,N)
        chNM[:, s1:e1] = (D1 @ blk).T
        if L >= 2:
            blk = chNM[:, s2:e2].T           # (5,N)
            chNM[:, s2:e2] = (D2 @ blk).T
        if L >= 3:
            blk = chNM[:, s3:e3].T           # (7,N)
            chNM[:, s3:e3] = (D3 @ blk).T
        return chNM

    R_ch = rot_channel(R_ch)
    G_ch = rot_channel(G_ch)
    B_ch = rot_channel(B_ch)

    sh_M3_rot = torch.stack([R_ch, G_ch, B_ch], dim=-1)    # (N,M,3)
    sh_rot = sh_M3_rot.view(N, 3 * M).contiguous()         # (N,3M)

    # Return type matches input type 
    if out_torch:
        return sh_rot.to(sh_in.device, dtype=sh_in.dtype if sh_in.dtype.is_floating_point else torch.float32)
    else:
        return sh_rot.detach().cpu().numpy().astype(np.float32)

def quaternion_wxyz_to_matrix(q, eps: float = 1e-12):
    """
    Convert quaternion [w,x,y,z] to rotation matrix.
    Supports:
      - numpy or torch input
      - shape (4,) or (N, 4)
    Returns:
      - If input is numpy → numpy, (3,3) or (N,3,3)
      - If input is torch → torch, keeps original device/dtype
    """
    is_torch = isinstance(q, torch.Tensor)

    if is_torch:
        dev, dtype_in = q.device, q.dtype
        qt = q
        if qt.ndim == 1:
            qt = qt.unsqueeze(0)  # (1,4)
        qt = qt.to(dev, dtype=torch.float32)

        # normalize
        n = torch.clamp(torch.linalg.norm(qt, dim=-1, keepdim=True), min=eps)
        qt = qt / n
        w, x, y, z = qt.unbind(dim=-1)  # (N,)

        x2, y2, z2 = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        r00 = 1 - 2 * (y2 + z2)
        r01 = 2 * (xy - wz)
        r02 = 2 * (xz + wy)

        r10 = 2 * (xy + wz)
        r11 = 1 - 2 * (x2 + z2)
        r12 = 2 * (yz - wx)

        r20 = 2 * (xz - wy)
        r21 = 2 * (yz + wx)
        r22 = 1 - 2 * (x2 + y2)

        R = torch.stack([
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1)
        ], dim=-2)  # (N,3,3)

        if q.ndim == 1:
            R = R[0]
        return R.to(dev, dtype=dtype_in if dtype_in.is_floating_point else torch.float32)

    else:
        qn = np.asarray(q, dtype=np.float32)
        squeeze_back = False
        if qn.ndim == 1:
            qn = qn[None, :]  # (1,4)
            squeeze_back = True

        n = np.linalg.norm(qn, axis=-1, keepdims=True)
        n = np.maximum(n, eps)
        qn = qn / n
        w, x, y, z = qn[..., 0], qn[..., 1], qn[..., 2], qn[..., 3]

        x2, y2, z2 = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        r00 = 1 - 2 * (y2 + z2)
        r01 = 2 * (xy - wz)
        r02 = 2 * (xz + wy)

        r10 = 2 * (xy + wz)
        r11 = 1 - 2 * (x2 + z2)
        r12 = 2 * (yz - wx)

        r20 = 2 * (xz - wy)
        r21 = 2 * (yz + wx)
        r22 = 1 - 2 * (x2 + y2)

        R = np.stack([
            np.stack([r00, r01, r02], axis=-1),
            np.stack([r10, r11, r12], axis=-1),
            np.stack([r20, r21, r22], axis=-1)
        ], axis=-2).astype(np.float32)  # (N,3,3)

        if squeeze_back:
            R = R[0]
        return R

def compute_cov3d(scales, rots_or_R, mod: float = 1.0, eps: float = 1e-12):
    """
    Compute 3D covariance: Cov = R · diag(mod * scale^2) · R^T
    Supports:
      - numpy / torch
      - single sample or batch processing
    Args:
      scales   : (..., 3)
      rots_or_R: (..., 4) as quaternion [w,x,y,z], or (..., 3, 3) as rotation matrix
      mod      : scalar coefficient
    Returns:
      - If any input is torch → returns torch, device consistent with that torch input
      - Otherwise returns numpy
    """
    # Determine backend & device
    any_torch = isinstance(scales, torch.Tensor) or isinstance(rots_or_R, torch.Tensor)

    if any_torch:
        main = rots_or_R if isinstance(rots_or_R, torch.Tensor) else scales
        dev = main.device
        dtype = torch.float32

        s = scales if isinstance(scales, torch.Tensor) else torch.from_numpy(np.asarray(scales))
        X = rots_or_R if isinstance(rots_or_R, torch.Tensor) else torch.from_numpy(np.asarray(rots_or_R))
        s = s.to(dev, dtype=dtype)
        X = X.to(dev, dtype=dtype)

        # Ensure batch dimension
        if s.ndim == 1: s = s.unsqueeze(0)
        if X.ndim == 1: X = X.unsqueeze(0)

        # Get R
        if X.shape[-1] == 4 and X.ndim >= 2 and X.shape[-2] != 3:
            R = quaternion_wxyz_to_matrix(X)        # (...,3,3)
        elif X.shape[-2:] == (3, 3):
            R = X
        else:
            raise ValueError("rots_or_R must be (...,4) quaternions or (...,3,3) rotation matrices.")

        # Cov = A @ A^T, A = R * (sqrt(mod)*scales)
        s_scaled = torch.sqrt(torch.tensor(mod, device=dev, dtype=dtype)) * s
        A = R * s_scaled.unsqueeze(-2)
        cov = A @ A.transpose(-2, -1)
        return cov

    else:
        s = np.asarray(scales, dtype=np.float32)
        X = np.asarray(rots_or_R, dtype=np.float32)

        squeeze_cov = False
        if s.ndim == 1:
            s = s[None, :]
            squeeze_cov = (X.ndim == 1) or (X.ndim == 2)

        # Get R
        if (X.ndim >= 2) and (X.shape[-1] == 4) and (X.shape[-2] != 3):
            R = quaternion_wxyz_to_matrix(X)  # (N,3,3)  (...,3,3)
        elif X.ndim >= 2 and X.shape[-2:] == (3, 3):
            R = X
            if R.ndim == 2:
                R = R[None, :, :]
        else:
            raise ValueError("rots_or_R must be (...,4) quaternions or (...,3,3) rotation matrices.")

        s_scaled = np.sqrt(np.float32(mod)) * s              # (N,3)
        A = R * s_scaled[:, None, :]                         # (N,3,3)
        cov = A @ np.transpose(A, (0, 2, 1))                 # (N,3,3)

        if squeeze_cov and cov.shape[0] == 1:
            cov = cov[0]
        return cov.astype(np.float32)

def dummy_gaussian_data():
    """Create sample Gaussian data for testing"""
    gau_xyz = np.array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ]).astype(np.float32).reshape(-1, 3)
    gau_rot = np.array([
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0
    ]).astype(np.float32).reshape(-1, 4)
    gau_s = np.array([
        0.03, 0.03, 0.03,
        0.2, 0.03, 0.03,
        0.03, 0.2, 0.03,
        0.03, 0.03, 0.2
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = np.array([
        1, 0, 1, 
        1, 0, 0, 
        0, 1, 0, 
        0, 0, 1, 
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = (gau_c - 0.5) / 0.28209
    gau_a = np.array([
        1, 1, 1, 1
    ]).astype(np.float32).reshape(-1, 1)
    
    gau_nxnynz = np.array([
        1, 0, 0,
        1, 0, 0,
        1, 0, 0,
        1, 0, 0,
    ]).astype(np.float32).reshape(-1, 3)
    gau_filter_3Ds = np.array([
        1,
        1,
        1,
        1,
    ]).astype(np.float32).reshape(-1, 1)

    return GaussianData(
        gau_xyz,
        gau_rot,
        gau_s,
        gau_a,
        gau_c,
        gau_nxnynz,
        gau_filter_3Ds
    )