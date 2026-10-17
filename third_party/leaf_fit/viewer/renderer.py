import os
import sys

if 'PYOPENGL_ACCELERATE' not in os.environ:
    os.environ['PYOPENGL_ACCELERATE'] = 'False'

from OpenGL import GL as gl
from OpenGL.GL import *
import numpy as np
from dataclasses import dataclass
from utils import MeshData, load_shaders, set_uniform_mat4, set_uniform_vec3, ensure_gl_array_compatibility, load_texture

def load_shaders(vs, fs):
    from OpenGL.GL import shaders
    vertex_shader = open(vs, 'r').read()        
    fragment_shader = open(fs, 'r').read()

    active_shader = shaders.compileProgram(
        shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER),
        shaders.compileShader(fragment_shader, gl.GL_FRAGMENT_SHADER),
    )
    return active_shader

def set_attributes(program, keys, values, vao=None, buffer_ids=None):
    gl.glUseProgram(program)
    if vao is None:
        vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)

    if buffer_ids is None:
        buffer_ids = [None] * len(keys)
    for i, (key, value, b) in enumerate(zip(keys, values, buffer_ids)):
        if b is None:
            b = gl.glGenBuffers(1)
            buffer_ids[i] = b
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, b)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, value.nbytes, value.reshape(-1), gl.GL_STATIC_DRAW)
        length = value.shape[-1]
        pos = gl.glGetAttribLocation(program, key)
        gl.glVertexAttribPointer(pos, length, gl.GL_FLOAT, False, 0, None)
        gl.glEnableVertexAttribArray(pos)
    
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER,0)
    return vao, buffer_ids

def set_faces_tovao(vao, faces: np.ndarray):
    gl.glBindVertexArray(vao)
    element_buffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, element_buffer)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, gl.GL_STATIC_DRAW)
    return element_buffer

def set_storage_buffer_data(program, key, value: np.ndarray, bind_idx, vao=None, buffer_id=None):
    gl.glUseProgram(program)
    if vao is not None:
        gl.glBindVertexArray(vao)
    
    if buffer_id is None:
        buffer_id = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, buffer_id)
    gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, value.nbytes, value.reshape(-1), gl.GL_STATIC_DRAW)
    gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, bind_idx, buffer_id)
    gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
    return buffer_id

def set_uniform_mat4(shader, content, name):
    gl.glUseProgram(shader)
    import glm
    if isinstance(content, glm.mat4):
        content = np.array(content).astype(np.float32)
    else:
        content = content.T
    gl.glUniformMatrix4fv(
        gl.glGetUniformLocation(shader, name), 
        1,
        gl.GL_FALSE,
        content.astype(np.float32)
    )

def set_uniform_1f(shader, content, name):
    gl.glUseProgram(shader)
    gl.glUniform1f(
        gl.glGetUniformLocation(shader, name), 
        content,
    )

def set_uniform_1int(shader, content, name):
    gl.glUseProgram(shader)
    gl.glUniform1i(
        gl.glGetUniformLocation(shader, name), 
        content
    )

def set_uniform_v3(shader, contents, name):
    gl.glUseProgram(shader)
    gl.glUniform3f(
        gl.glGetUniformLocation(shader, name),
        contents[0], contents[1], contents[2]
    )

_sort_buffer_xyz = None
_sort_buffer_gausid = None

def _sort_gaussian_cpu(gaus, view_mat):
    xyz = np.asarray(gaus.xyz)
    view_mat = np.asarray(view_mat)

    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]

    index = np.argsort(depth)
    index = index.astype(np.int32).reshape(-1, 1)
    return index

def _sort_gaussian_torch(gaus, view_mat):
    import torch
    global _sort_buffer_gausid, _sort_buffer_xyz
    if _sort_buffer_gausid != id(gaus):
        _sort_buffer_xyz = torch.tensor(gaus.xyz).cuda()
        _sort_buffer_gausid = id(gaus)

    xyz = _sort_buffer_xyz
    view_mat = torch.tensor(view_mat).cuda()
    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]
    index = torch.argsort(depth)
    index = index.type(torch.int32).reshape(-1, 1).cpu().numpy()
    return index

_sort_gaussian = None
try:
    import torch
    if torch.cuda.is_available():
        print("Detect torch cuda installed, will use torch as sorting backend")
        _sort_gaussian = _sort_gaussian_torch
    else:
        raise ImportError
except ImportError:
    _sort_gaussian = _sort_gaussian_cpu

@dataclass
class GaussianData:
    xyz: np.ndarray
    rot: np.ndarray
    scale: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray
    
    def flat(self) -> np.ndarray:
        ret = np.concatenate([self.xyz, self.rot, self.scale, self.opacity, self.sh], axis=-1)
        return np.ascontiguousarray(ret)
    
    def __len__(self):
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        return self.sh.shape[-1]

class GaussianRenderer:
    def __init__(self, w, h):
        gl.glViewport(0, 0, w, h)
        self.program = load_shaders('shaders/gau_vert.glsl', 'shaders/gau_frag.glsl')

        self.quad_v = np.array([
            -1,  1,
            1,  1,
            1, -1,
            -1, -1
        ], dtype=np.float32).reshape(4, 2)
        self.quad_f = np.array([
            0, 1, 2,
            0, 2, 3
        ], dtype=np.uint32).reshape(2, 3)
        
        vao, buffer_id = set_attributes(self.program, ["position"], [self.quad_v])
        set_faces_tovao(vao, self.quad_f)
        self.vao = vao
        self.gau_bufferid = None
        self.index_bufferid = None
        
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        
        self.gaussians = None

    def update_gaussian_data(self, gaus: GaussianData):
        self.gaussians = gaus
        gaussian_data = gaus.flat()
        self.gau_bufferid = set_storage_buffer_data(self.program, "gaussian_data", gaussian_data, 
                                                     bind_idx=0,
                                                     buffer_id=self.gau_bufferid)
        set_uniform_1int(self.program, gaus.sh_dim, "sh_dim")

    def sort_and_update(self, camera):
        if self.gaussians is None:
            return
        index = _sort_gaussian(self.gaussians, camera.get_view_matrix())
        self.index_bufferid = set_storage_buffer_data(self.program, "gi", index, 
                                                       bind_idx=1,
                                                       buffer_id=self.index_bufferid)
   
    def set_scale_modifier(self, modifier):
        set_uniform_1f(self.program, modifier, "scale_modifier")

    def set_render_mod(self, mod: int):
        set_uniform_1int(self.program, mod, "render_mod")

    def set_render_reso(self, w, h):
        gl.glViewport(0, 0, w, h)

    def update_camera_pose(self, camera):
        view_mat = camera.get_view_matrix()
        set_uniform_mat4(self.program, view_mat, "view_matrix")
        set_uniform_v3(self.program, camera.position, "cam_pos")

    def update_camera_intrin(self, camera):
        proj_mat = camera.get_project_matrix()
        set_uniform_mat4(self.program, proj_mat, "projection_matrix")
        htanfovxy_focal = camera.get_htanfovxy_focal()
        set_uniform_v3(self.program, htanfovxy_focal, "hfovxy_focal")

    def draw(self):
        if self.gaussians is None:
            return
        gl.glUseProgram(self.program)
        gl.glBindVertexArray(self.vao)
        num_gau = len(self.gaussians)
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, len(self.quad_f.reshape(-1)), gl.GL_UNSIGNED_INT, None, num_gau)


class MeshRenderer:
    def __init__(self):
        self.program = load_shaders('shaders/mesh_vert.glsl', 'shaders/mesh_frag.glsl')
        self.vao = None
        self.vertex_buffer = None
        self.normal_buffer = None
        self.color_buffer = None
        self.uv_buffer = None
        self.element_buffer = None
        self.texture_id = None
        self.has_texture = False
        
    def setup_mesh(self, mesh: MeshData):
        if self.texture_id is not None:
            glDeleteTextures([self.texture_id])
            self.texture_id = None
            self.has_texture = False
        
        if self.vao is None:
            self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        if self.vertex_buffer is None:
            self.vertex_buffer = glGenBuffers(1)
        vertices = ensure_gl_array_compatibility(mesh.vertices)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        if self.normal_buffer is None:
            self.normal_buffer = glGenBuffers(1)
        normals = ensure_gl_array_compatibility(mesh.normals)
        glBindBuffer(GL_ARRAY_BUFFER, self.normal_buffer)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)
        
        if self.color_buffer is None:
            self.color_buffer = glGenBuffers(1)
        colors = ensure_gl_array_compatibility(mesh.colors)
        glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
        glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(2)
        
        if mesh.uvs is not None:
            if self.uv_buffer is None:
                self.uv_buffer = glGenBuffers(1)
            uvs = ensure_gl_array_compatibility(mesh.uvs)
            glBindBuffer(GL_ARRAY_BUFFER, self.uv_buffer)
            glBufferData(GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL_STATIC_DRAW)
            glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(3)
        else:
            default_uvs = np.zeros((len(mesh.vertices), 2), dtype=np.float32)
            if self.uv_buffer is None:
                self.uv_buffer = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.uv_buffer)
            glBufferData(GL_ARRAY_BUFFER, default_uvs.nbytes, default_uvs, GL_STATIC_DRAW)
            glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(3)
        
        if mesh.has_texture:
            texture_loaded = False
            
            if hasattr(mesh, 'texture_data') and mesh.texture_data is not None:
                from utils import load_texture_from_data
                self.texture_id = load_texture_from_data(mesh.texture_data)
                texture_loaded = True
                
            elif mesh.texture_path:
                self.texture_id = load_texture(mesh.texture_path)
                texture_loaded = True
            
            self.has_texture = texture_loaded and self.texture_id is not None
        else:
            self.has_texture = False
        
        if self.element_buffer is None:
            self.element_buffer = glGenBuffers(1)
        faces = ensure_gl_array_compatibility(mesh.faces)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)
        
        self.num_faces = len(mesh.faces)
        glBindVertexArray(0)
        
    def render(self, view_matrix, projection_matrix, camera_pos):
        if self.vao is None:
            return
            
        glUseProgram(self.program)
        glBindVertexArray(self.vao)
        
        set_uniform_mat4(self.program, view_matrix, "view_matrix")
        set_uniform_mat4(self.program, projection_matrix, "projection_matrix")
        set_uniform_vec3(self.program, camera_pos, "camera_pos")
        
        has_texture_location = glGetUniformLocation(self.program, "has_texture")
        glUniform1i(has_texture_location, 1 if self.has_texture else 0)
        
        if self.has_texture and self.texture_id is not None:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            texture_location = glGetUniformLocation(self.program, "texture_diffuse")
            glUniform1i(texture_location, 0)
        
        glEnable(GL_DEPTH_TEST)
        
        glEnable(GL_CULL_FACE)
        
        glCullFace(GL_BACK)
        glDrawElements(GL_TRIANGLES, self.num_faces * 3, GL_UNSIGNED_INT, None)
        
        glCullFace(GL_FRONT)  
        glDrawElements(GL_TRIANGLES, self.num_faces * 3, GL_UNSIGNED_INT, None)
        
        glDisable(GL_CULL_FACE)
        
        if self.has_texture:
            glBindTexture(GL_TEXTURE_2D, 0)
        
        glBindVertexArray(0)