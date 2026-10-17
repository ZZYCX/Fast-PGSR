import os
import sys
from plyfile import PlyData, PlyElement

if 'OpenGL_accelerate' in sys.modules:
    del sys.modules['OpenGL_accelerate']
    
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = ''
if 'PYOPENGL_ACCELERATE' not in os.environ:
    os.environ['PYOPENGL_ACCELERATE'] = 'False'

class MockAccelerateModule:
    def __getattr__(self, name):
        raise ImportError("OpenGL_accelerate is disabled for NumPy 2.0 compatibility")
        
if 'OpenGL_accelerate' not in sys.modules:
    sys.modules['OpenGL_accelerate'] = MockAccelerateModule()

import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.GL import (glGenTextures, glBindTexture, glTexParameteri, glTexImage2D, 
                        glGenerateMipmap, GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
                        GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_REPEAT, GL_LINEAR,
                        GL_RGB, GL_RGBA, GL_RED, GL_UNSIGNED_BYTE)
import OpenGL.GL.shaders as shaders
from dataclasses import dataclass

def ensure_gl_array_compatibility(array):
    if isinstance(array, np.ndarray):
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)
        if array.dtype == np.float64:
            array = array.astype(np.float32)
        elif array.dtype == np.int64:
            array = array.astype(np.int32)
    return array

@dataclass
class MeshData:
    vertices: np.ndarray  # shape (N, 3)
    faces: np.ndarray     # shape (M, 3)
    normals: np.ndarray   # shape (N, 3)
    colors: np.ndarray    # shape (N, 3)
    uvs: np.ndarray = None        
    texture_path: str = None      
    has_texture: bool = False     
@dataclass  
class GaussianData:
    xyz: np.ndarray      # shape (N, 3)
    rot: np.ndarray      # shape (N, 4) 
    scale: np.ndarray    # shape (N, 3)
    opacity: np.ndarray  # shape (N, 1)
    sh: np.ndarray       # shape (N, SH_dims)
    nxnynz: np.ndarray   # shape (N, 3)
    filter_3Ds: np.ndarray # shape (N, 1)
    
    def flat(self) -> np.ndarray:
        ret = np.concatenate([self.xyz, self.rot, self.scale, self.opacity, self.sh, self.nxnynz, self.filter_3Ds], axis=-1)
        return np.ascontiguousarray(ret)
    
    def __len__(self):
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        return self.sh.shape[-1]

def create_test_cube():
    vertices = np.array([
        # Back face (z = -1)
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        # Front face (z = 1)  
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        # Left face (x = -1)
        [-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1],
        # Right face (x = 1)
        [1, -1, -1], [1, -1, 1], [1, 1, 1], [1, 1, -1],
        # Bottom face (y = -1)
        [-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1],
        # Top face (y = 1)
        [-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1]
    ], dtype=np.float32)
    
    normals = np.array([
        # Back face
        [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1],
        # Front face
        [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
        # Left face
        [-1, 0, 0], [-1, 0, 0], [-1, 0, 0], [-1, 0, 0],
        # Right face
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        # Bottom face
        [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0],
        # Top face
        [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]
    ], dtype=np.float32)
    
    faces = np.array([
        # Back face
        [0, 1, 2], [0, 2, 3],
        # Front face
        [4, 6, 5], [4, 7, 6],
        # Left face
        [8, 9, 10], [8, 10, 11],
        # Right face
        [12, 14, 13], [12, 15, 14],
        # Bottom face
        [16, 17, 18], [16, 18, 19],
        # Top face
        [20, 22, 21], [20, 23, 22]
    ], dtype=np.uint32)
    
    colors = np.array([
        [0.3, 0.3, 0.8], [0.3, 0.3, 0.8], [0.3, 0.3, 0.8], [0.3, 0.3, 0.8],
        [0.8, 0.3, 0.3], [0.8, 0.3, 0.3], [0.8, 0.3, 0.3], [0.8, 0.3, 0.3],
        [0.3, 0.8, 0.3], [0.3, 0.8, 0.3], [0.3, 0.8, 0.3], [0.3, 0.8, 0.3],
        [0.8, 0.8, 0.3], [0.8, 0.8, 0.3], [0.8, 0.8, 0.3], [0.8, 0.8, 0.3],
        [0.8, 0.3, 0.8], [0.8, 0.3, 0.8], [0.8, 0.3, 0.8], [0.8, 0.3, 0.8],
        [0.3, 0.8, 0.8], [0.3, 0.8, 0.8], [0.3, 0.8, 0.8], [0.3, 0.8, 0.8]
    ], dtype=np.float32)
    
    return MeshData(vertices=vertices, faces=faces, normals=normals, colors=colors)


def load_shaders(vs_path, fs_path):
    with open(vs_path, 'r') as f:
        vertex_shader = f.read()
    with open(fs_path, 'r') as f:
        fragment_shader = f.read()
        
    program = shaders.compileProgram(
        shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER),
    )
    return program

def set_uniform_mat4(program, matrix, name):
    glUseProgram(program)
    location = glGetUniformLocation(program, name)
    glUniformMatrix4fv(location, 1, GL_FALSE, matrix.T.astype(np.float32))

def set_uniform_vec3(program, vec, name):
    glUseProgram(program)
    location = glGetUniformLocation(program, name)
    glUniform3f(location, vec[0], vec[1], vec[2])

def load_obj_file(filepath):

    vertices = []
    normals = []
    uvs = []
    faces = []
    face_normals = []
    face_uvs = []
    material_file = None
    current_material = None
    materials = {}
    
    obj_dir = os.path.dirname(filepath)
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if not parts:
                continue
                
            cmd = parts[0]
            
            if cmd == 'v':  
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif cmd == 'vn':  
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif cmd == 'vt':  
                uvs.append([float(parts[1]), float(parts[2])])
            elif cmd == 'f':  
                face_verts = []
                face_normals_idx = []
                face_uvs_idx = []
                
                for vertex_data in parts[1:]:
                    indices = vertex_data.split('/')
                    face_verts.append(int(indices[0]) - 1)  
                    
                    if len(indices) > 1 and indices[1]:
                        face_uvs_idx.append(int(indices[1]) - 1)
                    else:
                        face_uvs_idx.append(None)
                        
                    if len(indices) > 2 and indices[2]:
                        face_normals_idx.append(int(indices[2]) - 1)
                    else:
                        face_normals_idx.append(None)
                
                if len(face_verts) == 3:
                    faces.append(face_verts)
                    face_normals.append(face_normals_idx)
                    face_uvs.append(face_uvs_idx)
                elif len(face_verts) == 4:
                    faces.append([face_verts[0], face_verts[1], face_verts[2]])
                    faces.append([face_verts[0], face_verts[2], face_verts[3]])
                    face_normals.append([face_normals_idx[0], face_normals_idx[1], face_normals_idx[2]])
                    face_normals.append([face_normals_idx[0], face_normals_idx[2], face_normals_idx[3]])
                    face_uvs.append([face_uvs_idx[0], face_uvs_idx[1], face_uvs_idx[2]])
                    face_uvs.append([face_uvs_idx[0], face_uvs_idx[2], face_uvs_idx[3]])
                    
            elif cmd == 'mtllib':  
                material_file = os.path.join(obj_dir, parts[1])
            elif cmd == 'usemtl':  
                current_material = parts[1]
    
    texture_path = None
    if material_file and os.path.exists(material_file):
        texture_path = load_mtl_file(material_file, obj_dir)
    
    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.uint32)
    
    if normals:
        normals = np.array(normals, dtype=np.float32)
        vertex_normals = np.zeros_like(vertices)
        for i, face in enumerate(faces):
            for j, vertex_idx in enumerate(face):
                if face_normals[i][j] is not None:
                    vertex_normals[vertex_idx] = normals[face_normals[i][j]]
    else:
        vertex_normals = compute_vertex_normals(vertices, faces)
    
    vertex_uvs = None
    has_texture = False
    if uvs and any(any(uv_idx is not None for uv_idx in face_uv) for face_uv in face_uvs):
        uvs = np.array(uvs, dtype=np.float32)
        vertex_uvs = np.zeros((len(vertices), 2), dtype=np.float32)
        for i, face in enumerate(faces):
            for j, vertex_idx in enumerate(face):
                if face_uvs[i][j] is not None:
                    vertex_uvs[vertex_idx] = uvs[face_uvs[i][j]]
        has_texture = texture_path is not None
    
    if has_texture:
        colors = np.ones_like(vertices, dtype=np.float32)  
    else:
        colors = np.full_like(vertices, 0.7, dtype=np.float32)
    
    return MeshData(
        vertices=vertices,
        faces=faces,
        normals=vertex_normals,
        colors=colors,
        uvs=vertex_uvs,
        texture_path=texture_path,
        has_texture=has_texture
    )

def load_mtl_file(mtl_path, obj_dir):

    try:
        with open(mtl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('map_Kd'):  
                    parts = line.split()
                    if len(parts) > 1:
                        texture_filename = parts[1]
                        texture_path = os.path.join(obj_dir, texture_filename)
                        if os.path.exists(texture_path):
                            return texture_path
    except:
        pass
    return None

def compute_vertex_normals(vertices, faces):
    normals = np.zeros_like(vertices)
    
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        
        normals[face[0]] += face_normal
        normals[face[1]] += face_normal
        normals[face[2]] += face_normal
    
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1  
    normals = normals / norms
    
    return normals

def load_texture(image_path):

    if not os.path.exists(image_path):
        return None
        
    try:
        image = Image.open(image_path)
        image = image.convert('RGB')  
        image = image.transpose(Image.FLIP_TOP_BOTTOM)  
        
        img_data = np.array(image, dtype=np.uint8)
        width, height = image.size
        
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)
        
        glBindTexture(GL_TEXTURE_2D, 0)
        
        return texture_id
        
    except Exception as e:
        return None


def load_texture_from_data(texture_data):
    try:
        texture_data = np.flipud(texture_data)
        
        if texture_data.dtype != np.uint8:
            texture_data = (texture_data * 255).astype(np.uint8)
        
        if len(texture_data.shape) == 3:
            height, width, channels = texture_data.shape
            if channels == 3:
                format = GL_RGB
            elif channels == 4:
                format = GL_RGBA
            else:
                return None
        else:
            height, width = texture_data.shape
            format = GL_RED
        
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, texture_data)
        glGenerateMipmap(GL_TEXTURE_2D)
        
        glBindTexture(GL_TEXTURE_2D, 0)
        
        return texture_id
        
    except Exception as e:
        return None
    
    

def load_ply_gaussian(filepath):
        
    max_sh_degree = 3
    plydata = PlyData.read(filepath)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    
    nxnynz = np.stack((np.asarray(plydata.elements[0]["nx"]),
                    np.asarray(plydata.elements[0]["ny"]),
                    np.asarray(plydata.elements[0]["nz"])),  axis=1)
    
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    filter_3Ds = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis]

    xyz = xyz.astype(np.float32)
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)
    scales = np.exp(scales).astype(np.float32)
    opacities = (1/(1 + np.exp(- opacities))).astype(np.float32)  # sigmoid
    shs = np.concatenate([features_dc.reshape(-1, 3), 
                        features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)
    
    return GaussianData(xyz, rots, scales, opacities, shs, nxnynz, filter_3Ds)

_sort_buffer_xyz = None
_sort_buffer_gausid = None
_sort_backend = None

def _sort_gaussian_cpu(gaussians: GaussianData, view_matrix):
    xyz = np.asarray(gaussians.xyz)
    view_matrix = np.asarray(view_matrix)

    xyz_view = view_matrix[None, :3, :3] @ xyz[..., None] + view_matrix[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]

    index = np.argsort(depth)
    index = index.astype(np.int32)
    
    return index

def _sort_gaussian_torch(gaussians: GaussianData, view_matrix):
    import torch
    global _sort_buffer_xyz, _sort_buffer_gausid
    
    if _sort_buffer_gausid != id(gaussians):
        _sort_buffer_xyz = torch.tensor(gaussians.xyz).cuda()
        _sort_buffer_gausid = id(gaussians)

    xyz = _sort_buffer_xyz
    view_matrix = torch.tensor(view_matrix).cuda()
    
    xyz_view = view_matrix[None, :3, :3] @ xyz[..., None] + view_matrix[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]
    
    index = torch.argsort(depth)
    index = index.type(torch.int32).cpu().numpy()
    
    return index

def _initialize_sort_backend():
    global _sort_backend
    
    if _sort_backend is not None:
        return _sort_backend
        
    try:
        import torch
        if torch.cuda.is_available():
            _sort_backend = _sort_gaussian_torch
            return _sort_backend
    except ImportError:
        pass
    
    _sort_backend = _sort_gaussian_cpu
    return _sort_backend

def sort_gaussians_by_depth(gaussians: GaussianData, view_matrix):
    backend = _initialize_sort_backend()
    return backend(gaussians, view_matrix)

def get_sort_backend_name():
    backend = _initialize_sort_backend()
    if backend == _sort_gaussian_torch:
        return "PyTorch CUDA"
    else:
        return "CPU (NumPy)"

def set_sort_backend(backend_name):
    global _sort_backend
    
    if backend_name == "cpu":
        _sort_backend = _sort_gaussian_cpu
    elif backend_name == "torch":
        try:
            import torch
            if torch.cuda.is_available():
                _sort_backend = _sort_gaussian_torch
            else:
                _sort_backend = _sort_gaussian_cpu
        except ImportError:
            _sort_backend = _sort_gaussian_cpu

def get_available_backends():
    backends = ["CPU (NumPy)"]
    try:
        import torch
        if torch.cuda.is_available():
            backends.append("PyTorch CUDA")
    except ImportError:
        pass
    return backends

def center_gaussians(gaussians: GaussianData):
    if gaussians is None or gaussians.xyz is None:
        return gaussians
    
    center = np.mean(gaussians.xyz, axis=0)
    
    centered_xyz = gaussians.xyz - center
    
    centered_gaussians = GaussianData(
        xyz=centered_xyz,
        rot=gaussians.rot,
        scale=gaussians.scale,
        opacity=gaussians.opacity,
        sh=gaussians.sh,
        nxnynz=gaussians.nxnynz,
        filter_3Ds=gaussians.filter_3Ds
    )
    
    return centered_gaussians

def apply_temperature_colors(sh_c0, gaussians, temp_colors):
    
    new_sh = gaussians.sh.copy()
    
    sh_dc_colors = (temp_colors - 0.5) / sh_c0
    
    new_sh[:, 0] = sh_dc_colors[:, 0]  # R
    new_sh[:, 1] = sh_dc_colors[:, 1]  # G  
    new_sh[:, 2] = sh_dc_colors[:, 2]  # B
    
    return GaussianData(
        xyz=gaussians.xyz,
        rot=gaussians.rot,
        scale=gaussians.scale,
        opacity=gaussians.opacity,
        sh=new_sh,
        nxnynz=gaussians.nxnynz,
        filter_3Ds=gaussians.filter_3Ds
    )