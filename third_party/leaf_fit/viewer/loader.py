# ==== MLS via Compute Shader + Instanced Rendering + Texture (robust timers, warmup, fixed camera) ====
# deps: pip install numpy open3d PyOpenGL glfw

import os, sys, ctypes, math, time
import numpy as np
import glfw
import open3d as o3d
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

# ---------------- NumPy 2.0 / OpenGL_accelerate guard ----------------
if 'OpenGL_accelerate' in sys.modules:
    del sys.modules['OpenGL_accelerate']
os.environ['PYOPENGL_PLATFORM']   = ''
os.environ['PYOPENGL_ACCELERATE'] = 'False'
class MockModule:
    def __getattr__(self, name):
        raise ImportError("OpenGL_accelerate is disabled for NumPy 2.0 compatibility")
sys.modules['OpenGL_accelerate'] = MockModule()

# ---------------- Robust query helpers (PyOpenGL raw fallback) ----------------
try:
    from OpenGL.raw.GL.VERSION.GL_3_3 import glGetQueryObjectui64v as raw_glGetQueryObjectui64v
except Exception:
    raw_glGetQueryObjectui64v = None
try:
    from OpenGL.raw.GL.VERSION.GL_1_5 import glGetQueryObjectuiv as raw_glGetQueryObjectuiv
except Exception:
    raw_glGetQueryObjectuiv = None

def _to_int_list(x):
    if isinstance(x, (list, tuple)): return [int(v) for v in x]
    if isinstance(x, np.ndarray):     return [int(v) for v in x.flatten()]
    return [int(x)]
def _to_int_scalar(x): return _to_int_list(x)[0]
def query_available(q): return bool(glGetQueryObjectiv(int(q), GL_QUERY_RESULT_AVAILABLE))
def query_result_ns(q):
    q = int(q)
    try:
        return int(glGetQueryObjectui64v(q, GL_QUERY_RESULT))  # type: ignore
    except Exception:
        pass
    if raw_glGetQueryObjectui64v is not None:
        val = ctypes.c_uint64(0); raw_glGetQueryObjectui64v(q, GL_QUERY_RESULT, val); return int(val.value)
    try:
        return int(glGetQueryObjectuiv(q, GL_QUERY_RESULT))  # type: ignore
    except Exception:
        pass
    if raw_glGetQueryObjectuiv is not None:
        val32 = ctypes.c_uint(0); raw_glGetQueryObjectuiv(q, GL_QUERY_RESULT, val32); return int(val32.value)
    raise RuntimeError("Unable to read GPU timer query result")

# ---------------- Load MLS pack (NPZ) ----------------
PACK_PATH = "/home/cg/my_codes/leaf_to_forest/exp/packs/plant0/fps_mls/Auto_Leaf_1.npz"  
data = np.load(PACK_PATH, allow_pickle=True)
V        = data["vertices"].astype(np.float32)          # (V,3)
pairPool = data["pairPool"].astype(np.float32)          # (2*N*C,4) : [src.xyz0, dst.xyz0, ...]
N, C     = int(data["N"][0]), int(data["C"][0])
indices  = data["indices"]      if "indices"      in data else None
normals  = data["normals"]      if "normals"      in data else None
uvs      = data["uvs"]          if "uvs"          in data else None
tex_img  = data["texture_data"] if "texture_data" in data else None
print(f"[Init] V={len(V)}, N={N}, C={C}, indexed={indices is not None}, texture={tex_img is not None}")

# ---------------- Test cylinder generation if path_info is available ----------------
if 'path_info' in data:
    print(f"[Cylinder Test] Found path_info in pack, testing cylinder generation...")
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from cylinder_generator import generate_stem_cylinders

    try:
        path_info = data["path_info"].item()  # Extract from numpy array if needed
        cylinders = generate_stem_cylinders(path_info)
        print(f"[Cylinder Test] Successfully generated {len(cylinders)} cylinders")

        # Prepare cylinders for OpenGL rendering
        if cylinders:
            print(f"[Cylinder Test] Preparing {len(cylinders)} cylinders for OpenGL rendering...")

            # Extract vertices and faces from all cylinders
            cylinder_vertices = []
            cylinder_faces = []
            cylinder_colors = []
            vertex_offset = 0

            for cylinder in cylinders:
                if cylinder is not None:
                    # Get vertices, faces, and colors
                    verts = np.asarray(cylinder.vertices)
                    faces = np.asarray(cylinder.triangles)
                    colors = np.asarray(cylinder.vertex_colors) if cylinder.has_vertex_colors() else None

                    # Add vertices
                    cylinder_vertices.append(verts)

                    # Add faces with vertex offset
                    offset_faces = faces + vertex_offset
                    cylinder_faces.append(offset_faces)
                    vertex_offset += len(verts)

                    # Add colors (default to green if no colors)
                    if colors is not None:
                        cylinder_colors.append(colors)
                    else:
                        default_colors = np.full((len(verts), 3), [0.2, 0.8, 0.2])
                        cylinder_colors.append(default_colors)

            if cylinder_vertices:
                # Combine all cylinder data
                all_cylinder_vertices = np.vstack(cylinder_vertices).astype(np.float32)
                all_cylinder_faces = np.vstack(cylinder_faces).astype(np.uint32)
                all_cylinder_colors = np.vstack(cylinder_colors).astype(np.float32)

                print(f"[Cylinder Test] Combined: {len(all_cylinder_vertices)} vertices, {len(all_cylinder_faces)} faces")
            else:
                all_cylinder_vertices = all_cylinder_faces = all_cylinder_colors = None
        else:
            print(f"[Cylinder Test] No cylinders generated")
            all_cylinder_vertices = all_cylinder_faces = all_cylinder_colors = None

    except Exception as e:
        print(f"[Cylinder Test] Error during cylinder generation: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"[Cylinder Test] No path_info found in pack")

# ---------------- Init GL ----------------
if not glfw.init(): raise RuntimeError("GLFW init failed")
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
win = glfw.create_window(1280, 720, "MLS Compute + Instanced Render + Texture", None, None)
if not win:
    glfw.terminate(); raise RuntimeError("Create window failed")
glfw.make_context_current(win)
glfw.swap_interval(0)  # VSync off

# ---------------- Camera (fixed, no orbit) ----------------
def compute_center_radius(pts):
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    center = (mn + mx) * 0.5
    r = float(np.max(np.linalg.norm(pts - center, axis=1)))
    return center.astype(np.float32), (r if r>1e-8 else 1.0)

def look_at(eye, center, up):
    f = center - eye; f = f/ (np.linalg.norm(f)+1e-8)
    s = np.cross(f, up); s = s/ (np.linalg.norm(s)+1e-8)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0,:3]=s; M[1,:3]=u; M[2,:3]=-f
    T = np.eye(4, dtype=np.float32); T[:3,3] = -eye
    return M @ T

def persp(fovy_deg, aspect, n, f):
    fct = 1.0/math.tan(math.radians(fovy_deg)/2.0)
    P = np.zeros((4,4), np.float32)
    P[0,0]=fct/max(aspect,1e-6); P[1,1]=fct
    P[2,2]=(f+n)/(n-f); P[2,3]=(2*f*n)/(n-f); P[3,2]=-1
    return P

mesh_center, mesh_radius = compute_center_radius(V)
print(f"[Optimization] Cached center={mesh_center}, radius={mesh_radius:.3f}")

def build_mvp(width, height):
    aspect = width/height
    fovy = 45.0
    theta = math.radians(fovy)/2.0
    dist_fit = (mesh_radius / math.sin(theta)) * 1.2
    dist = dist_fit * 1
    eye = mesh_center + np.array([dist*0.4, dist*0.8, dist*0.4], np.float32)
    up  = np.array([0,1,0], np.float32)
    view = look_at(eye, mesh_center, up)
    proj = persp(fovy, aspect, max(0.01, dist - mesh_radius*2.0), dist + mesh_radius*2.0)
    return (proj @ view).astype(np.float32)

# ---------------- Texture helper ----------------
def load_texture_from_data(texture_data):
    try:
        img = np.array(texture_data, copy=True)
        # img = np.flipud(img)
        if img.dtype != np.uint8:
            img = (img * 255.0).clip(0,255).astype(np.uint8)

        if img.ndim == 3:
            h, w, ch = img.shape
            if ch == 3:
                fmt = GL_RGB; internal = GL_RGB
            elif ch == 4:
                fmt = GL_RGBA; internal = GL_RGBA
            else:
                raise ValueError(f"Unsupported channels: {ch}")
        elif img.ndim == 2:
            h, w = img.shape; ch = 1
            fmt = GL_RED; internal = GL_RED
        else:
            raise ValueError(f"Bad texture shape: {img.shape}")

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0, fmt, GL_UNSIGNED_BYTE, img)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        return tex
    except Exception as e:
        print(f"[Texture] load failed: {e}")
        return None

# ---------------- Buffers ----------------
Vcount = len(V)
# positions (vec4 for alignment)
pos4 = np.concatenate([V, np.zeros((Vcount,1), np.float32)], axis=1)

vao = glGenVertexArrays(1); glBindVertexArray(vao)

if normals is None:
    normals = np.zeros_like(V, np.float32); normals[:,1]=1.0
norm_vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, norm_vbo)
glBufferData(GL_ARRAY_BUFFER, normals.astype(np.float32).nbytes, normals.astype(np.float32), GL_STATIC_DRAW)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
glEnableVertexAttribArray(1)

if uvs is None:
    xy_min = V[:,:2].min(axis=0); xy_rng = V[:,:2].max(axis=0)-xy_min; xy_rng[xy_rng==0]=1
    uvs = ((V[:,:2]-xy_min)/xy_rng).astype(np.float32)
uv_vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, uv_vbo)
glBufferData(GL_ARRAY_BUFFER, uvs.astype(np.float32).nbytes, uvs.astype(np.float32), GL_STATIC_DRAW)
glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
glEnableVertexAttribArray(3)

# Index buffer
use_indexed = indices is not None
if use_indexed:
    idx = indices.flatten().astype(np.uint32) if (indices.ndim==2 and indices.shape[1]==3) else indices.astype(np.uint32)
    ibo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)
else:
    ibo = None

glBindVertexArray(0)

# ---------------- Cylinder VAO (if available) ----------------
cylinder_vao = None
cylinder_vertex_count = 0
if 'all_cylinder_vertices' in locals() and all_cylinder_vertices is not None:
    print(f"[Cylinder Setup] Setting up OpenGL buffers for {len(all_cylinder_vertices)} vertices")

    cylinder_vao = glGenVertexArrays(1)
    glBindVertexArray(cylinder_vao)

    # Vertex buffer
    cylinder_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, cylinder_vbo)
    glBufferData(GL_ARRAY_BUFFER, all_cylinder_vertices.nbytes, all_cylinder_vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)
    glEnableVertexAttribArray(0)

    # Color buffer
    cylinder_color_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, cylinder_color_vbo)
    glBufferData(GL_ARRAY_BUFFER, all_cylinder_colors.nbytes, all_cylinder_colors, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)
    glEnableVertexAttribArray(1)

    # Index buffer
    cylinder_ibo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cylinder_ibo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, all_cylinder_faces.nbytes, all_cylinder_faces, GL_STATIC_DRAW)

    cylinder_vertex_count = len(all_cylinder_faces) * 3
    glBindVertexArray(0)
    print(f"[Cylinder Setup] Cylinder VAO ready with {cylinder_vertex_count} vertices")

# SSBO 0: pairPool
ssbo_pair = glGenBuffers(1)
glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_pair)
glBufferData(GL_SHADER_STORAGE_BUFFER, pairPool.nbytes, pairPool, GL_STATIC_DRAW)
glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_pair)

# SSBO 1: input positions (vec4)
ssbo_in = glGenBuffers(1)
glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_in)
glBufferData(GL_SHADER_STORAGE_BUFFER, pos4.nbytes, pos4, GL_STATIC_DRAW)
glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo_in)

# SSBO 2: deformed positions for all groups (N*V, vec4)
out_count = N * Vcount
ssbo_out = glGenBuffers(1)
glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_out)
glBufferData(GL_SHADER_STORAGE_BUFFER, out_count*4*4, None, GL_DYNAMIC_DRAW)  # vec4
glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo_out)

# ---------------- Texture (optional) ----------------
texture_id = None
use_texture = False
if tex_img is not None:
    texture_id = load_texture_from_data(tex_img)
    if texture_id:
        use_texture = True
        print(f"[Texture] created id={texture_id}")
    else:
        print("[Texture] unavailable, fallback to vertex color")

# ---------------- Compute Shader (MLS) ----------------
COMPUTE_SRC = r"""
#version 430
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

const int MAX_C = 64;

layout(std430, binding = 0) readonly buffer PairPoolBuffer { vec4 pairPool[]; };
layout(std430, binding = 1) readonly buffer InPosBuffer    { vec4 inPos[];     };
layout(std430, binding = 2) writeonly buffer OutPosBuffer  { vec4 outPos[];    };

uniform int   Vcount;   // vertices per leaf
uniform int   N;        // leaves
uniform int   C;        // control points per leaf (<= MAX_C)
uniform float sigma;    // e.g. 0.1

shared vec4 sPairs[2 * MAX_C];

mat3 inverse3x3(mat3 M){
    float a00=M[0][0], a01=M[0][1], a02=M[0][2];
    float a10=M[1][0], a11=M[1][1], a12=M[1][2];
    float a20=M[2][0], a21=M[2][1], a22=M[2][2];

    float c00=(a11*a22-a12*a21);
    float c01=-(a10*a22-a12*a20);
    float c02=(a10*a21-a11*a20);
    float c10=-(a01*a22-a02*a21);
    float c11=(a00*a22-a02*a20);
    float c12=-(a00*a21-a01*a20);
    float c20=(a01*a12-a02*a11);
    float c21=-(a00*a12-a01*a10);
    float c22=(a00*a11-a01*a10);

    float det = a00*c00 + a01*c01 + a02*c02;
    det = max(det, 1e-8);
    float invDet = 1.0/det;

    mat3 adj = mat3(
        c00, c10, c20,
        c01, c11, c21,
        c02, c12, c22
    );
    return invDet * adj;
}

void main(){
    uint WG   = gl_WorkGroupSize.x;                         // 256
    uint lane = gl_LocalInvocationIndex;                    // 0..255

    uint tilesPerGroup = (uint(Vcount) + WG - 1u) / WG;

    uint groupId = gl_WorkGroupID.x;
    uint g   = groupId % uint(N);                           
    uint tile= groupId / uint(N);                           
    uint v   = tile * WG + lane;                            
    if (v >= uint(Vcount)) return;

    uint base = g * uint(C) * 2u;                           

    for (uint i = lane; i < uint(C)*2u; i += WG) {
        sPairs[i] = pairPool[base + i];
    }
    barrier();  

    vec3 original = inPos[v].xyz;
    float inv_2sigma2 = 0.5 / (sigma * sigma);

    float weights[MAX_C];
    float wsum = 0.0;
    for (int i=0;i<C && i<MAX_C;i++){
        vec3 P = sPairs[uint(i)*2u].xyz;
        vec3 d = original - P;
        float w = exp(-dot(d,d) * inv_2sigma2);
        weights[i] = w;
        wsum += w;
    }
    if (wsum < 1e-12){
        outPos[g*uint(Vcount) + v] = vec4(original, 1.0);
        return;
    }
    float inv_wsum = 1.0/wsum;
    for (int i=0;i<C && i<MAX_C;i++) weights[i] *= inv_wsum;

    vec3 Pbar = vec3(0.0), Qbar = vec3(0.0);
    for (int i=0;i<C && i<MAX_C;i++){
        vec3 P = sPairs[uint(i)*2u    ].xyz;
        vec3 Q = sPairs[uint(i)*2u + 1u].xyz;
        float w = weights[i];
        Pbar += w * P;
        Qbar += w * Q;
    }

    // Pass3: M / B
    mat3 M = mat3(0.0);
    mat3 B = mat3(0.0);
    for (int i=0;i<C && i<MAX_C;i++){
        float w = weights[i];
        vec3 P = sPairs[uint(i)*2u    ].xyz;
        vec3 Q = sPairs[uint(i)*2u + 1u].xyz;
        vec3 p = P - Pbar;
        vec3 q = Q - Qbar;
        M += w * outerProduct(p, p);
        B += w * outerProduct(q, p);
    }

    M[0][0]+=1e-6; M[1][1]+=1e-6; M[2][2]+=1e-6;
    mat3 A = B * inverse3x3(M);
    vec3 t = Qbar - A * Pbar;

    vec3 outp = A * original + t;
    outPos[g*uint(Vcount) + v] = vec4(outp, 1.0);   
}
"""

compute_prog = glCreateProgram()
cs = compileShader(COMPUTE_SRC, GL_COMPUTE_SHADER)
glAttachShader(compute_prog, cs)
glLinkProgram(compute_prog)
if glGetProgramiv(compute_prog, GL_LINK_STATUS) != GL_TRUE:
    raise RuntimeError(glGetProgramInfoLog(compute_prog).decode())
glDeleteShader(cs)

# ---------------- Render Shaders (instanced; pos from SSBO; texture) ----------------
VS_SRC = r"""
#version 430
layout(location=1) in vec3 in_normal;
layout(location=3) in vec2 in_uv;

layout(std430, binding = 2) buffer OutPosBuffer { vec4 outPos[]; };

uniform int Vcount;
uniform mat4 uMVP;

out vec3 v_normal;
out vec2 v_uv;

void main(){
    int v = gl_VertexID;
    int g = gl_InstanceID;
    vec3 pos = outPos[g*Vcount + v].xyz;
    v_normal = normalize(in_normal);
    v_uv = in_uv;
    gl_Position = uMVP * vec4(pos, 1.0);
}
"""

FS_SRC = r"""
#version 430
in vec3 v_normal;
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D texture_sampler;
uniform bool use_texture;

void main(){
    vec3 n = normalize(v_normal);
    vec3 L1 = normalize(vec3(0.6,1.0,0.3));
    vec3 L2 = normalize(vec3(-0.3,0.5,-0.8));
    float l = max(dot(n,L1),0.0)*1.2 + max(dot(n,L2),0.0)*0.6 + 0.6;

    vec3 base = use_texture ? texture(texture_sampler, v_uv).rgb
                            : vec3(0.2,0.8,0.2);
    fragColor = vec4(base * l, 1.0);
}
"""

# ---------------- Cylinder Shaders ----------------
CYLINDER_VS_SRC = """
#version 430 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

uniform mat4 uMVP;

out vec3 vColor;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vColor = aColor;
}
"""

CYLINDER_FS_SRC = """
#version 430 core
in vec3 vColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(vColor, 1.0);
}
"""

render_prog = compileProgram(compileShader(VS_SRC, GL_VERTEX_SHADER),
                             compileShader(FS_SRC, GL_FRAGMENT_SHADER))

# Cylinder shader program
cylinder_prog = None
if 'all_cylinder_vertices' in locals() and all_cylinder_vertices is not None:
    cylinder_prog = compileProgram(compileShader(CYLINDER_VS_SRC, GL_VERTEX_SHADER),
                                   compileShader(CYLINDER_FS_SRC, GL_FRAGMENT_SHADER))
    cylinder_loc_uMVP = glGetUniformLocation(cylinder_prog, "uMVP")
loc_uMVP    = glGetUniformLocation(render_prog, "uMVP")
loc_Vcount  = glGetUniformLocation(render_prog, "Vcount")
loc_use_tex = glGetUniformLocation(render_prog, "use_texture")
loc_sampler = glGetUniformLocation(render_prog, "texture_sampler")

# ---------------- Timer queries ----------------
def have_timer():
    try:
        q = _to_int_scalar(glGenQueries(1))
        glBeginQuery(GL_TIME_ELAPSED, q); glEndQuery(GL_TIME_ELAPSED)
        glDeleteQueries(1, [q]); return True
    except Exception as e:
        print("[Timer] not supported:", e); return False

TQ = 3
timer_ok = have_timer()
if timer_ok:
    q_comp = _to_int_list(glGenQueries(TQ))
    q_draw = _to_int_list(glGenQueries(TQ))
    print("[Timer] GPU time queries enabled (triple-buffered)")
else:
    q_comp = q_draw = None
    print("[Timer] Fallback to wallclock only")

# ---------------- Main loop (warmup + fixed frames) ----------------
WARMUP        = 0
TARGET_FRAMES = 10000
LOCAL_SIZE    = 256
num_invocations = N*Vcount
num_groups = (num_invocations + LOCAL_SIZE - 1)//LOCAL_SIZE

glEnable(GL_DEPTH_TEST)
glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

# bind texture (if any) once
if texture_id and use_texture:
    glUseProgram(render_prog)
    glUniform1i(loc_use_tex, GL_TRUE)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glUniform1i(loc_sampler, 0)
else:
    glUseProgram(render_prog)
    glUniform1i(loc_use_tex, GL_FALSE)

sum_frame_ms = 0.0; frames = 0
sum_glsync_ms = 0.0; glsync_samples = 0
sum_comp_ms = 0.0; comp_samples = 0
sum_draw_ms = 0.0; draw_samples = 0

comp_idx = 0; draw_idx = 0
comp_have_prev = False; draw_have_prev = False

print(f"[Run] V={Vcount}, N={N}, C={C}, invocations={num_invocations}, groups={num_groups}, WARMUP={WARMUP}")
print(f"[Note] Compute+Barrier+Draw are included in GL-sync wallclock; GPU queries split Compute/Draw.")

while not glfw.window_should_close(win) and frames < TARGET_FRAMES:
    frame_start = time.perf_counter()
    glfw.poll_events()

    # ---- Build MVP
    fb_w, fb_h = glfw.get_framebuffer_size(win)
    glViewport(0, 0, fb_w, fb_h)
    MVP = build_mvp(fb_w, fb_h)

    # ---- COMPUTE (dispatch) ----
    if timer_ok:
        glBeginQuery(GL_TIME_ELAPSED, int(q_comp[comp_idx]))
    glUseProgram(compute_prog)
    glUniform1i(glGetUniformLocation(compute_prog, "Vcount"), Vcount)
    glUniform1i(glGetUniformLocation(compute_prog, "N"), N)
    glUniform1i(glGetUniformLocation(compute_prog, "C"), C)
    glUniform1f(glGetUniformLocation(compute_prog, "sigma"), 0.1)
    glDispatchCompute(num_groups, 1, 1)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
    if timer_ok:
        glEndQuery(GL_TIME_ELAPSED)

    # ---- DRAW (instanced) ----
    if timer_ok:
        glBeginQuery(GL_TIME_ELAPSED, int(q_draw[draw_idx]))
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUseProgram(render_prog)
    glUniformMatrix4fv(loc_uMVP, 1, GL_FALSE, MVP.T)
    glUniform1i(loc_Vcount, Vcount)

    glBindVertexArray(vao)
    if use_indexed:
        glDrawElementsInstanced(GL_TRIANGLES, idx.size, GL_UNSIGNED_INT, None, N)
    else:
        glDrawArraysInstanced(GL_TRIANGLES, 0, Vcount, N)
    glBindVertexArray(0)

    # ---- DRAW CYLINDERS (if available) ----
    if cylinder_prog is not None and cylinder_vao is not None:
        glUseProgram(cylinder_prog)
        glUniformMatrix4fv(cylinder_loc_uMVP, 1, GL_FALSE, MVP.T)
        glBindVertexArray(cylinder_vao)
        glDrawElements(GL_TRIANGLES, cylinder_vertex_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    if timer_ok:
        glEndQuery(GL_TIME_ELAPSED)

    # ---- GL-sync wallclock for (compute + draw)
    t0 = time.perf_counter()
    glFinish()
    t1 = time.perf_counter()

    glfw.swap_buffers(win)
    frame_end = time.perf_counter()

    # ---- Accumulate after warmup ----
    frame_ms = (frame_end - frame_start)*1000.0
    if frames >= WARMUP:
        sum_frame_ms += frame_ms
        glsync_ms = (t1 - t0)*1000.0
        sum_glsync_ms += glsync_ms; glsync_samples += 1

    # read prev compute / draw query (non-blocking)
    if timer_ok:
        # compute
        prv = (comp_idx - 1) % TQ
        if comp_have_prev and query_available(q_comp[prv]):
            try:
                ns = query_result_ns(q_comp[prv]); ms = ns/1e6
                if frames >= WARMUP: sum_comp_ms += ms; comp_samples += 1
            except Exception: pass
        else:
            comp_have_prev = True
        comp_idx = (comp_idx + 1) % TQ

        # draw
        prv = (draw_idx - 1) % TQ
        if draw_have_prev and query_available(q_draw[prv]):
            try:
                ns = query_result_ns(q_draw[prv]); ms = ns/1e6
                if frames >= WARMUP: sum_draw_ms += ms; draw_samples += 1
            except Exception: pass
        else:
            draw_have_prev = True
        draw_idx = (draw_idx + 1) % TQ

    if (frames+1) % 100 == 0 and frames >= WARMUP:
        print(f"Frame {frames+1:4d}: Total={frame_ms:6.3f}ms | GLsync(Comp+Draw)={(t1-t0)*1000.0:6.3f}ms")

    frames += 1

# try to read the last pending queries (short wait)
if timer_ok:
    # compute
    if comp_have_prev:
        prv = (comp_idx-1)%TQ
        t0p = time.perf_counter()
        while True:
            if query_available(q_comp[prv]) or (time.perf_counter()-t0p)>0.25: break
            time.sleep(0.001)
        if query_available(q_comp[prv]):
            try:
                ns = query_result_ns(q_comp[prv]); ms = ns/1e6
                if frames > WARMUP: sum_comp_ms += ms; comp_samples += 1
            except Exception: pass
    # draw
    if draw_have_prev:
        prv = (draw_idx-1)%TQ
        t0p = time.perf_counter()
        while True:
            if query_available(q_draw[prv]) or (time.perf_counter()-t0p)>0.25: break
            time.sleep(0.001)
        if query_available(q_draw[prv]):
            try:
                ns = query_result_ns(q_draw[prv]); ms = ns/1e6
                if frames > WARMUP: sum_draw_ms += ms; draw_samples += 1
            except Exception: pass

# ---------------- Report ----------------
effective = max(frames - WARMUP, 1)
avg_frame_ms  = sum_frame_ms / effective
print("\n====== Averages over {} frames (after warmup {}) ======".format(effective, WARMUP))
print(f"[Frame]                    avg_ms = {avg_frame_ms:8.3f}    avg_fps = {1000.0/avg_frame_ms:8.2f}")

if glsync_samples>0:
    glsync_ms = sum_glsync_ms / glsync_samples
    print(f"[GL-sync wallclock total]  avg_ms = {glsync_ms:8.3f}    avg_fps = {1000.0/glsync_ms:8.2f}")

if comp_samples>0:
    comp_ms = sum_comp_ms / comp_samples
    print(f"[Compute(GPU timer)]       avg_ms = {comp_ms:8.3f}    avg_fps = {1000.0/comp_ms:8.2f}")
else:
    print("[Compute(GPU timer)] No samples")

if draw_samples>0:
    draw_ms = sum_draw_ms / draw_samples
    print(f"[Draw(GPU timer)]          avg_ms = {draw_ms:8.3f}    avg_fps = {1000.0/draw_ms:8.2f}")
else:
    print("[Draw(GPU timer)] No samples")

if comp_samples>0 and draw_samples>0:
    combined_ms = comp_ms + draw_ms
    combined_fps = 1000.0 / combined_ms
    print(f"[Compute+Draw GPU total]   avg_ms = {combined_ms:8.3f}    avg_fps = {combined_fps:8.2f}")
else:
    print("[Compute+Draw GPU total] Insufficient samples")

# ---------------- Cleanup ----------------
if timer_ok:
    glDeleteQueries(len(q_comp), q_comp)
    glDeleteQueries(len(q_draw), q_draw)
glDeleteProgram(compute_prog)
glDeleteProgram(render_prog)
glDeleteVertexArrays(1, [vao])
glDeleteBuffers(1, [norm_vbo])
glDeleteBuffers(1, [uv_vbo])
if ibo: glDeleteBuffers(1, [ibo])
glDeleteTextures([texture_id] if texture_id else [])
glDeleteBuffers(1, [ssbo_pair])
glDeleteBuffers(1, [ssbo_in])
glDeleteBuffers(1, [ssbo_out])
glfw.terminate()
