# ==== Minimal Mesh Renderer (VC only) ====
# deps: pip install numpy open3d PyOpenGL glfw

import os, sys, ctypes, math, time
import numpy as np
import open3d as o3d
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

# ---------------- NumPy 2.0 / OpenGL_accelerate guard ----------------
if 'OpenGL_accelerate' in sys.modules:
    del sys.modules['OpenGL_accelerate']
os.environ['PYOPENGL_PLATFORM'] = ''
os.environ['PYOPENGL_ACCELERATE'] = 'False'
class MockModule:
    def __getattr__(self, name):
        raise ImportError("OpenGL_accelerate is disabled for NumPy 2.0 compatibility")
sys.modules['OpenGL_accelerate'] = MockModule()

# === RAW fallbacks (need 3 args incl. pointer) ===
try:
    from OpenGL.raw.GL.VERSION.GL_3_3 import glGetQueryObjectui64v as raw_glGetQueryObjectui64v
except Exception:
    raw_glGetQueryObjectui64v = None
try:
    from OpenGL.raw.GL.VERSION.GL_1_5 import glGetQueryObjectuiv as raw_glGetQueryObjectuiv
except Exception:
    raw_glGetQueryObjectuiv = None

# === Helpers for robust query handling ===
def _to_int_list(x):
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.flatten()]
    return [int(x)]
def _to_int_scalar(x): return _to_int_list(x)[0]
def query_available(q): return bool(glGetQueryObjectiv(int(q), GL_QUERY_RESULT_AVAILABLE))
def query_result_ns(q):
    q = int(q)
    try:  # high-level 64-bit
        ns = glGetQueryObjectui64v(q, GL_QUERY_RESULT)  # type: ignore
        return int(ns)
    except Exception:
        pass
    if raw_glGetQueryObjectui64v is not None:
        try:
            val = ctypes.c_uint64(0)
            raw_glGetQueryObjectui64v(q, GL_QUERY_RESULT, val)
            return int(val.value)
        except Exception:
            pass
    try:  # 32-bit fallbacks
        ns32 = glGetQueryObjectuiv(q, GL_QUERY_RESULT)  # type: ignore
        return int(ns32)
    except Exception:
        pass
    if raw_glGetQueryObjectuiv is not None:
        val32 = ctypes.c_uint(0)
        raw_glGetQueryObjectuiv(q, GL_QUERY_RESULT, val32)
        return int(val32.value)
    raise RuntimeError("Unable to read GPU timer query result")

# === Hardcoded PLY path ===
MESH_PATH = "/home/cg/my_codes/leaf_to_forest/exp/meshes_old/plant0/Tree_GOF.ply"

# === Load PLY (Open3D) ===
mesh = o3d.io.read_triangle_mesh(MESH_PATH)
if mesh.is_empty():
    raise RuntimeError(f"Failed to read mesh: {MESH_PATH}")
mesh.compute_vertex_normals()
verts  = np.asarray(mesh.vertices, dtype=np.float32)
faces  = np.asarray(mesh.triangles, dtype=np.uint32).reshape(-1,3)
colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
if colors.size == 0 or colors.shape != verts.shape:
    colors = np.full_like(verts, 0.8, dtype=np.float32)
print(f"[Init] Loaded mesh: V={len(verts)}, F={len(faces)}, colors={colors.size>0}")

# === Init window/context ===
if not glfw.init():
    raise RuntimeError("Failed to init GLFW")
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
win = glfw.create_window(1920, 1080, "Minimal Mesh Renderer (VC only)", None, None)
if not win:
    glfw.terminate(); raise RuntimeError("Failed to create window")
glfw.make_context_current(win)
glfw.swap_interval(0)  # VSync off

# === Camera / MVP ===
def compute_center_size(m: np.ndarray):
    mn = m.min(axis=0); mx = m.max(axis=0)
    center = (mn + mx) / 2.0
    size   = float(np.max(mx - mn))
    return center.astype(np.float32), size

def look_at(eye, center, up):
    f = center - eye; f = f / (np.linalg.norm(f) + 1e-8)
    s = np.cross(f, up); s = s / (np.linalg.norm(s) + 1e-8)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0,:3] = s; M[1,:3] = u; M[2,:3] = -f
    T = np.eye(4, dtype=np.float32); T[:3,3] = -eye
    return M @ T

def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg)/2.0)
    P = np.zeros((4,4), dtype=np.float32)
    P[0,0] = f/max(aspect,1e-6); P[1,1] = f
    P[2,2] = (far+near)/(near-far); P[2,3] = (2*far*near)/(near-far)
    P[3,2] = -1.0
    return P

def build_mvp(width, height, azim, elev_deg, distance_scale=1.0):
    center, _ = mesh_center_cached, mesh_size_cached  # Use pre-computed values
    r = mesh_radius_cached
    aspect = max(width/height, 1e-6)
    fovy = math.radians(45.0)
    fovx = 2.0 * math.atan(math.tan(fovy/2.0) * aspect)
    theta = min(fovy/2.0, fovx/2.0)
    dist_fit = (r / math.sin(theta)) * 1.05
    dist = dist_fit * 0.5
    elev = math.radians(elev_deg)
    eye = center + np.array([
        dist * math.cos(elev) * math.cos(azim),
        dist * math.sin(elev),
        dist * math.cos(elev) * math.sin(azim)
    ], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    view = look_at(eye, center, up)
    proj = perspective(45.0, aspect, max(0.01, dist - r*2.0), dist + r*2.0)
    return (proj @ view).astype(np.float32)

# ---- Cache expensive mesh calculations (DO ONCE, NOT PER FRAME!) ----
mesh_center_cached, mesh_size_cached = compute_center_size(verts)
mesh_radius_cached = float(np.max(np.linalg.norm(verts - mesh_center_cached, axis=1))) or 1.0
print(f"[Optimization] Cached mesh center={mesh_center_cached}, radius={mesh_radius_cached:.3f}")

# ---- Orbit params & bench switches ----
azim = 0.0
elev_deg = 80.0
angular_speed = math.radians(30.0)
BENCH_FIX_CAMERA = True      
orbit_enabled = (not BENCH_FIX_CAMERA)
distance_scale = 1.0

fb_w, fb_h = glfw.get_framebuffer_size(win)
MVP = build_mvp(fb_w, fb_h, azim, elev_deg, distance_scale)

# === Shaders ===
VS = """
#version 410 core
layout(location=0) in vec3 in_pos;
layout(location=1) in vec3 in_col;
uniform mat4 uMVP;
out vec3 v_col;
void main(){
    v_col = in_col;
    gl_Position = uMVP * vec4(in_pos, 1.0);
}
"""
FS = """
#version 410 core
in vec3 v_col;
out vec4 frag;
void main(){ frag = vec4(v_col, 1.0); }
"""
prog = compileProgram(compileShader(VS, GL_VERTEX_SHADER),
                      compileShader(FS, GL_FRAGMENT_SHADER))
glUseProgram(prog)
loc_uMVP = glGetUniformLocation(prog, "uMVP")
glUniformMatrix4fv(loc_uMVP, 1, GL_FALSE, MVP.T)

# === VAO/VBO/EBO ===
interleaved = np.hstack([verts, colors]).astype(np.float32)  # (V,6)
indices = faces.flatten()
vao = glGenVertexArrays(1); glBindVertexArray(vao)
vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL_STATIC_DRAW)
stride = 6*4
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0));  glEnableVertexAttribArray(0)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12)); glEnableVertexAttribArray(1)
ebo = glGenBuffers(1); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
glBindVertexArray(0)

# === Render state ===
glEnable(GL_DEPTH_TEST)
glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

# === Stats (with warmup) ===
last_time = time.perf_counter()
sum_dt = 0.0
frame_count = 0
TARGET_FRAMES = 10000
WARMUP = 0  

# === GPU timer (triple-buffered) ===
gpu_timing_supported = False
gpu_queries = None
QUERY_COUNT = 3
gpu_query_idx = 0
gpu_have_prev = False
sum_gpu_ms = 0.0
gpu_samples = 0

# === GL sync wallclock (CUDA-like) ===
sum_glsync_ms = 0.0
glsync_samples = 0

def check_gpu_timing_support():
    try:
        test = _to_int_scalar(glGenQueries(1))
        glBeginQuery(GL_TIME_ELAPSED, test)
        glEndQuery(GL_TIME_ELAPSED)
        glDeleteQueries(1, [test])
        return True
    except Exception as e:
        print(f"[GPU Timer] not supported: {e}")
        return False

gpu_timing_supported = check_gpu_timing_support()
if gpu_timing_supported:
    try:
        gpu_queries = _to_int_list(glGenQueries(QUERY_COUNT))
        print("[GPU Timer] enabled (triple-buffered)")
    except Exception as e:
        print(f"[GPU Timer] create failed: {e}")
        gpu_timing_supported = False
else:
    print("[GPU Timer] disabled; will report only Frame averages")

print(f"[Run] Rendering {TARGET_FRAMES} frames... (WARMUP={WARMUP})")

# === Main loop (fixed frame count) ===
while not glfw.window_should_close(win) and frame_count < TARGET_FRAMES:
    glfw.poll_events()

    frame_start = time.perf_counter()

    # dt for orbit
    now = time.perf_counter()
    dt = now - last_time
    last_time = now
    if dt <= 0: dt = 1e-9

    if orbit_enabled:
        azim += angular_speed * dt

    # Update MVP
    mvp_start = time.perf_counter()
    fb_w, fb_h = glfw.get_framebuffer_size(win)
    glViewport(0, 0, fb_w, fb_h)
    MVP = build_mvp(fb_w, fb_h, azim, elev_deg, distance_scale)
    glUseProgram(prog)
    glUniformMatrix4fv(loc_uMVP, 1, GL_FALSE, MVP.T)
    mvp_end = time.perf_counter()

    # Clear
    glClearColor(0.05, 0.05, 0.05, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # ----- GPU timer: start THIS frame (only draw is timed) -----
    cur = gpu_query_idx
    prv = (gpu_query_idx - 1) % QUERY_COUNT if QUERY_COUNT > 0 else 0
    if gpu_timing_supported and gpu_queries is not None:
        try:
            glBeginQuery(GL_TIME_ELAPSED, int(gpu_queries[cur]))
        except Exception as e:
            print(f"[GPU Timer] begin failed: {e}")
            gpu_timing_supported = False

    # ---- Draw (GL-sync wallclock) ----
    glBindVertexArray(vao)
    t0 = time.perf_counter()
    glDrawElements(GL_TRIANGLES, indices.size, GL_UNSIGNED_INT, None)
    glBindVertexArray(0)
    glFinish()  
    t1 = time.perf_counter()

    if gpu_timing_supported and gpu_queries is not None:
        try:
            glEndQuery(GL_TIME_ELAPSED)
        except Exception as e:
            print(f"[GPU Timer] end failed: {e}")
            gpu_timing_supported = False

    if gpu_timing_supported and gpu_queries is not None:
        try:
            if gpu_have_prev and query_available(gpu_queries[prv]):
                try:
                    elapsed_ns = query_result_ns(gpu_queries[prv])
                    gpu_ms = elapsed_ns / 1e6
                    if frame_count >= WARMUP:
                        sum_gpu_ms += gpu_ms
                        gpu_samples += 1
                except Exception:
                    pass
            else:
                gpu_have_prev = True
            gpu_query_idx = (gpu_query_idx + 1) % QUERY_COUNT if QUERY_COUNT > 0 else 0
        except Exception as e:
            print(f"[GPU Timer] end/read failed: {e}")
            gpu_timing_supported = False

    pre_swap_time = time.perf_counter()

    swap_start = time.perf_counter()
    glfw.swap_buffers(win)
    swap_end = time.perf_counter()

    frame_end = time.perf_counter()

    frame_dur = frame_end - frame_start
    swap_dur = swap_end - swap_start
    pre_swap_dur = pre_swap_time - frame_start

    if frame_dur <= 0: frame_dur = 1e-9
    if frame_count >= WARMUP:
        sum_dt += frame_dur
        sum_glsync_ms += (t1 - t0) * 1000.0
        glsync_samples += 1

    if frame_count % 100 == 0 and frame_count > WARMUP:
        mvp_dur = (mvp_end - mvp_start) * 1000
        print(f"Frame {frame_count:4d}: Total={frame_dur*1000:.3f}ms | Render={((t1-t0)*1000):.3f}ms | MVP={mvp_dur:.3f}ms | PreSwap={pre_swap_dur*1000:.3f}ms | Swap={swap_dur*1000:.3f}ms")

    frame_count += 1

if gpu_timing_supported and gpu_queries is not None and gpu_have_prev:
    prv = (gpu_query_idx - 1) % QUERY_COUNT
    t_wait = time.perf_counter()
    while True:
        if query_available(gpu_queries[prv]) or (time.perf_counter() - t_wait) > 0.25:
            break
        time.sleep(0.001)
    if query_available(gpu_queries[prv]):
        try:
            elapsed_ns = query_result_ns(gpu_queries[prv])
            gpu_ms = elapsed_ns / 1e6
            if frame_count > WARMUP:
                sum_gpu_ms += gpu_ms
                gpu_samples += 1
        except Exception as e:
            print(f"[GPU Timer] final read failed: {e}")

# === Averages (after warmup) ===
effective_frames = max(frame_count - WARMUP, 1)
avg_frame_ms  = (sum_dt / effective_frames) * 1000.0
avg_frame_fps = 1000.0 / avg_frame_ms if avg_frame_ms > 0 else float('inf')

print("\n====== Averages over {} frames (after warmup {}) ======".format(effective_frames, WARMUP))
print(f"[Frame]                     avg_ms = {avg_frame_ms:8.3f}    avg_fps = {avg_frame_fps:8.2f}")

if glsync_samples > 0:
    avg_glsync_ms  = sum_glsync_ms / glsync_samples
    avg_glsync_fps = 1000.0 / avg_glsync_ms if avg_glsync_ms > 0 else float('inf')
    print(f"[Render(GL sync wallclock)] avg_ms = {avg_glsync_ms:8.3f}    avg_fps = {avg_glsync_fps:8.2f}")

if gpu_timing_supported and gpu_samples > 0:
    avg_gpu_ms   = sum_gpu_ms / gpu_samples
    avg_gpu_fps  = 1000.0 / avg_gpu_ms if avg_gpu_ms > 0 else float('inf')
    print(f"[Render(GPU timer)]         avg_ms = {avg_gpu_ms:8.3f}    avg_fps = {avg_gpu_fps:8.2f}")
else:
    print("[Render(GPU timer)] No GPU timing samples collected (unsupported or unavailable)")

# === Cleanup ===
if gpu_timing_supported and gpu_queries is not None:
    glDeleteQueries(len(gpu_queries), gpu_queries)
glDeleteProgram(prog)
glDeleteVertexArrays(1, [vao])
glDeleteBuffers(1, [vbo])
glDeleteBuffers(1, [ebo])
glfw.terminate()
