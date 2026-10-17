#!/usr/bin/env python3

# System related imports
import sys
import os
import time
import random

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'core'))
from viz_utils import vis_temperature_field, generate_non_gold_color, capture_screenshot
from auto_segment import fix_plant_root_direction_legacy, get_segment_mask, \
 get_temperature_field, find_local_tips, find_path_from_tip_to_root
from gen_template_leaf import compute_3d_edge_points_from_gaussian, compute_source_mvc_weights, reconstruct_target_coordinates_from_weights, extract_leaf_boundary_polygon
from gen_template_leaf import generate_template_leaf, generate_uv_mapping_mem, project_2d_coords_to_3d
from template_transform import get_transform_template_mesh_pca, get_transform_template_mesh_mls_corr, get_transform_template_mesh_mls_corr_optim, get_transform_template_mesh_mls_corr_kai
from mls import mls_denoising
from gaussian_utils import apply_indices_to_gaussian_data,  pack_for_gpu, save_gaussian_data_as_ply, apply_transformation_matrix_to_points
from generic_utils import chamfer_distance, write_mesh_to_disk, export_template_mesh

import numpy as np

if 'OpenGL_accelerate' in sys.modules:
    del sys.modules['OpenGL_accelerate']
    
# Set environment variables
os.environ['PYOPENGL_PLATFORM'] = ''
os.environ['PYOPENGL_ACCELERATE'] = 'False'

class MockModule:
    def __getattr__(self, name):
        raise ImportError("OpenGL_accelerate is disabled for NumPy 2.0 compatibility")
        
sys.modules['OpenGL_accelerate'] = MockModule()

from tqdm import tqdm
import glfw
import imgui
import fpsample
from imgui.integrations.glfw import GlfwRenderer
import OpenGL.GL as gl
from diff_gaussian_rasterization import gsplat_bvh
import potpourri3d as pp3d
from scipy.spatial import cKDTree
from utils import GaussianData, MeshData
from camera import Camera
from renderer import GaussianRenderer, MeshRenderer
from sklearn.neighbors import NearestNeighbors
from utils import apply_temperature_colors, load_ply_gaussian, get_sort_backend_name, \
    set_sort_backend, get_available_backends, center_gaussians


# Global variables
SH_C0 = 0.28209479177387814
# Predefined color array
colors = [
    [1.0, 0.0, 0.0],  # Red
    [0.0, 1.0, 0.0],  # Green
    [0.0, 0.0, 1.0],  # Blue
    [1.0, 0.0, 1.0],  # Purple
    [0.0, 1.0, 1.0],  # Cyan
    [0.5, 0.0, 1.0],  # Purple Blue
    [1.0, 0.0, 0.5],  # Pink
    [1.0, 0.5, 0.0],  # Orange
    [1.0, 0.2, 0.2],  # Light Red
    [0.2, 0.8, 0.2],  # Light Green
]
colors_edge = {
    "apex": [0.5, 1.0, 0.0],      # Chartreuse
    "base": [0.95, 0.84, 0.0],    # Dandelion
    "left": [0.7, 0.13, 0.13],    # Firebrick
    "right": [0.0, 0.18, 0.65]    # International Klein Blue
}

camera = None
mesh_renderer = None
gaussian_renderer = None
impl = None
window_width = 1024
window_height = 768

# Global variables for segmentation
path_info = None

# UI state
show_control_panel = True
show_simple_gaussian_picker = True
show_segments_menu = True
show_mesh_manager = False
bg_is_white = True  # Background color toggle, False=black, True=white
take_screenshot = False  # Screenshot flag
show_root_sphere = True  # Root sphere display toggle
g_scale_modifier = 1.0
g_auto_sort = True

# Template transformation parameters
edge_sampling_count = 20  # Edge sampling count, default 20
mls_num_corr = 64  # MLS correspondence points, default 64

# MVC estimation parameters
mvc_grid_density = 30  # MVC grid density, default 30
mvc_boundary_margin = 0.01  # MVC boundary margin, default 0.01
mvc_max_fps_points = 20  # MVC max FPS points, default 20

# Debug visualization parameters
show_edge_debug = False  # Show edge point debug visualization

# Mesh manager
template_meshes = []  # Store template meshes
edge_visualizations = []  # Store edge point visualization objects

# Segment hover management
hovered_segment_id = None  # Current hovered segment index
segment_original_colors = {}  # Store segment original colors
mouse_position = (0, 0)  # Current mouse position, for showing hover tooltip
last_hover_check_pos = (-1, -1)  # Last hover check mouse position
last_hover_check_time = 0.0  # Last hover check time
hover_check_interval = 0.05  # Hover check interval (20fps)
bvh_result_cache = None  # BVH query result cache
bvh_cache_mouse_pos = (-1, -1)  # Cached mouse position

# Sphere cache (avoid rebuilding every frame)
_root_sphere_cache = None

# Render mode (copied from original)
g_render_mode_tables = ["Gaussian Ball", 
                        "Flat Ball", 
                        "Billboard", 
                        "Depth", 
                        "SH:0", 
                        "SH:0~1", 
                        "SH:0~2", 
                        "SH:0~3 (default)"]
g_render_mode = 7  # Default to SH:0~3 mode

# Auto segmentation method selection
g_segmentation_method_tables = [
    "euclidean_density",
    "geodesic_density", 
    "geodesic_tip_density",
    "geometry_feature_euclidean",
    "geometry_feature_geodesic",
    "geometry_feature_geodesic_donuts", 
    "geometry_feature_geodesic_tip",
    "euclidean_graph",
    "geodesic_tip_graph",
    'gt',
    'neighbor_corrected'
]
g_segmentation_method = 8  # Default to geodesic_tip_graph

# Template Segment selection
g_template_segment_index = 0  # Default to "None"

# BVH construction constants
LEAF_SIZE = 16  # BVH leaf node size
NUM_BIN = 16   # Number of bins for BVH construction
current_bvh = None

# Leaf selection feature variables
selection_mode = False  # Whether in selection mode
brush_mode = False  # Whether in brush mode
leaf_tip_idx = None  # Leaf tip Gaussian index
current_selection = []  # Current selected Gaussian index list
drag_selection_radius = 0.15  # Geodesic distance radius for drag selection
drag_reference_depth = None  # Reference depth at drag start
drag_depth_tolerance_ratio = 0.1  # Depth difference ratio allowed during drag (reference * 1.1x)
cached_geodesic_distances = None  # Cached geodesic distance data
cached_root_distances = None
segments = []  # Persistent segments [{"indices": [], "color": [r,g,b], "name": "Segment 1"}]



original_sh_backup = None  # Backup of original SH color data
previously_selected = []  # Previously selected indices, for restoring colors

# Sphere selection feature variables
sphere_selection_points = []  # Store up to 2 selected point positions [{"position": [x,y,z], "gaussian_idx": idx}, ...]
sphere_click_count = 0  # Current click count (0, 1, 2 cycle)

segments_expanded = {}  # Track expand state of each segment {segment_name: bool}
root_idx = None  # Gaussian index of plant root

# Temperature field visualization
show_temperature_field = False  # Whether to show temperature field
temperature_colors = None  # Temperature field color cache
heat_solver = None  # PointCloudHeatSolver cache
sparse_heat_solver = None  # SparsePointCloudHeatSolver cache
sparse_indices = None  # Original index mapping for downsampling
original_to_sparse_mapping = None  # Original index to downsampled index mapping
leaf_tip_spheres = []  # Leaf tip sphere list
path_spheres = []  # Path sphere list

# Gaussian related
current_gaussians = None  # Current displayed Gaussian data (modified by segment operations)
original_plant_gaussians = None  # Immutable backup of plant overall data
intersected_idx = None  # Current selected Gaussian index

current_gaussian_path = ""
gaussian_picker_error = ""
seg_path = ""
mesh_path = ""
deform_path = ""
pack_path = ""

simple_gaussian_path = "../data"
simple_gaussian_path = os.path.abspath(simple_gaussian_path)

def init_glfw():
    global window_width, window_height
    
    if not glfw.init():
        print("Failed to initialize GLFW")
        sys.exit(1)
        
    primary_monitor = glfw.get_primary_monitor()
    video_mode = glfw.get_video_mode(primary_monitor)
    window_width = video_mode.size.width
    window_height = video_mode.size.height
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    window = glfw.create_window(window_width, window_height, "Gaussian Viewer", None, None)
    if not window:
        glfw.terminate()
        print("Failed to create window")
        sys.exit(1)
        
    glfw.make_context_current(window)
    glfw.swap_interval(1) 
    
    return window

def mouse_button_callback(window, button, action, mods):
    global selection_mode, brush_mode, leaf_tip_idx, current_selection, heat_solver
    global drag_reference_depth, drag_depth_tolerance_ratio
    global cached_geodesic_distances, intersected_idx
    
    if imgui.get_io().want_capture_mouse:
        return
        
    pressed = action == glfw.PRESS
    
    if button == glfw.MOUSE_BUTTON_LEFT:
        if pressed and (mods & glfw.MOD_CONTROL):
            handle_sphere_selection(window)
        elif pressed and glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
            handle_root_reposition(window)
        elif pressed and not selection_mode:
            s_key_pressed = glfw.get_key(window, glfw.KEY_S) == glfw.PRESS
            if s_key_pressed:
                handle_segment_click(window)
            camera.is_leftmouse_pressed = pressed
        else:
            camera.is_leftmouse_pressed = pressed
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        camera.is_rightmouse_pressed = pressed
    elif button == glfw.MOUSE_BUTTON_MIDDLE:
        
        if (mods & glfw.MOD_CONTROL) or (mods & glfw.MOD_SHIFT):
            if pressed:
                clear_all_highlights()
                clear_segment_hover() 
                selection_mode = False
                leaf_tip_idx = None
                current_selection = []
                
                global drag_reference_depth
                drag_reference_depth = None
                
                if current_gaussians is not None and current_bvh is not None:
                    mouse_x, mouse_y = glfw.get_cursor_pos(window)
                    
                    ray_origin, ray_direction = screen_to_ray(mouse_x, mouse_y, camera, window_width, window_height)
                    intersected_idx = find_intersected_gaussian(ray_origin, ray_direction, current_bvh, current_gaussians)
                    print(f"intersected_idx: {intersected_idx}")
                    if intersected_idx is not None:
                        
                        print(f"Selected leaf tip: Gaussian {intersected_idx}")
                        print("heat_solver: ", heat_solver)
                        if heat_solver is not None:
                            print(f"Computing geodesic distance from leaf tip {intersected_idx}...")
                            cached_geodesic_distances = heat_solver.compute_distance(intersected_idx)
                            print(f"Geodesic distance computation completed, cached {len(cached_geodesic_distances)} distance values")
                        
                        selection_mode = True
                        if mods & glfw.MOD_SHIFT:
                            brush_mode = True
                        leaf_tip_idx = intersected_idx
                        current_selection = [] 
                        clear_segment_hover()  
                        camera.is_middlemouse_pressed = True  
                    else:
                        print("No intersecting Gaussian found, selection cleared")
                        camera.is_middlemouse_pressed = False
                else:
                    print("Need to load PLY file first to use selection")
                    camera.is_middlemouse_pressed = False
            else:
                camera.is_middlemouse_pressed = False
                brush_mode = False
                
                if len(current_selection) > 0:
                    current_selection.append(current_selection[np.argmin(cached_root_distances[current_selection])])
                    current_selection = list(set(current_selection))

                drag_reference_depth = None
                
                if selection_mode:
                    print(f"Ctrl+Middle mouse released, selection mode ended. Current selection: {len(current_selection)} gaussians")
        else:
            camera.is_middlemouse_pressed = pressed

def cursor_pos_callback(window, xpos, ypos):
    global current_selection, brush_mode, mouse_position, heat_solver
    
    mouse_position = (xpos, ypos)
    
    if imgui.get_io().want_capture_mouse:
        camera.is_leftmouse_pressed = False
        camera.is_rightmouse_pressed = False
        camera.is_middlemouse_pressed = False
        return
    
    if selection_mode and leaf_tip_idx is not None and camera.is_middlemouse_pressed:
        ray_origin, ray_direction = screen_to_ray(xpos, ypos, camera, window_width, window_height)
        current_idx = find_intersected_gaussian(ray_origin, ray_direction, current_bvh, current_gaussians)
        
        if current_idx is not None:            
            if cached_geodesic_distances is not None:
                try:
                    if brush_mode:
                        if original_to_sparse_mapping is not None and sparse_heat_solver is not None:
                            sparse_idx = original_to_sparse_mapping[current_idx]
                            sparse_distances = sparse_heat_solver.compute_distance(sparse_idx)
                            full_distances = sparse_distances[original_to_sparse_mapping]
                            current_selection.extend(np.where(full_distances <= drag_selection_radius)[0].tolist())
                            
                        current_selection = list(set(current_selection))
                        current_selection.sort(key=lambda x: cached_root_distances[x], reverse=True)
                        current_selection.insert(0, leaf_tip_idx)
                    else:
                        
                        
                        current_distance = cached_geodesic_distances[current_idx]
                        current_selection = np.where(cached_geodesic_distances <= current_distance)[0].tolist()
                        current_selection.sort(key=lambda x: cached_root_distances[x], reverse=True)
                    
                    current_selection = list(set(current_selection))
                    
                    highlight_current_selection([1.0, 1.0, 0.0])  
                        
                except Exception as e:
                    print(f"Use cached geodesic distance failed: {e}")
            else:
                print("Warning: No available geodesic distance cache")
    
    s_key_pressed = glfw.get_key(window, glfw.KEY_S) == glfw.PRESS
    if (s_key_pressed and not selection_mode and 
        not camera.is_leftmouse_pressed and not camera.is_rightmouse_pressed):
        update_segment_hover_optimized(xpos, ypos)
    elif hovered_segment_id is not None:
        clear_segment_hover()
    
    camera.process_mouse(xpos, ypos)

def key_callback(window, key, scancode, action, mods):
    global show_control_panel, show_simple_gaussian_picker, show_segments_menu, show_mesh_manager
    global selection_mode, current_selection, take_screenshot
    
    if imgui.get_io().want_capture_keyboard:
        return
        
    if action == glfw.RELEASE and key == glfw.KEY_LEFT_CONTROL:
        if selection_mode and camera.is_middlemouse_pressed:
            camera.is_middlemouse_pressed = False
        
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_F1:
            show_control_panel = not show_control_panel
        elif key == glfw.KEY_F2:
            show_simple_gaussian_picker = not show_simple_gaussian_picker
        elif key == glfw.KEY_F3:
            show_segments_menu = not show_segments_menu
        elif key == glfw.KEY_F4:
            show_mesh_manager = not show_mesh_manager
        elif key == glfw.KEY_P:
            camera.print_camera_pose()
        elif key == glfw.KEY_O:
            take_screenshot = True
        elif key == glfw.KEY_EQUAL or key == glfw.KEY_KP_ADD:  
            camera.process_wheel(0, 1)  
        elif key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT: 
            camera.process_wheel(0, -1)  

def window_size_callback(window, width, height):
    global window_width, window_height
    window_width = width
    window_height = height
    camera.update_resolution(height, width)  
    gaussian_renderer.set_render_reso(width, height)
    gl.glViewport(0, 0, width, height)

def screen_to_ray(mouse_x, mouse_y, camera, window_width, window_height):
    ndc_x = (2.0 * mouse_x) / window_width - 1.0
    ndc_y = 1.0 - (2.0 * mouse_y) / window_height  
    
    view_matrix = np.array(camera.get_view_matrix(), dtype=np.float32)
    proj_matrix = np.array(camera.get_project_matrix(), dtype=np.float32)
    
    view_proj_matrix = proj_matrix @ view_matrix
    inv_view_proj_matrix = np.linalg.inv(view_proj_matrix)
    
    ray_start_ndc = np.array([ndc_x, ndc_y, -1.0, 1.0])  
    ray_end_ndc = np.array([ndc_x, ndc_y, 1.0, 1.0])     
    
    ray_start_world = inv_view_proj_matrix @ ray_start_ndc
    ray_end_world = inv_view_proj_matrix @ ray_end_ndc
    
    ray_start_world = ray_start_world[:3] / ray_start_world[3]
    ray_end_world = ray_end_world[:3] / ray_end_world[3]
    
    ray_direction = ray_end_world - ray_start_world
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    
    return ray_start_world, ray_direction

def find_intersected_gaussian(ray_origin, ray_direction, bvh_data, gaussians, filter=True):
    global drag_reference_depth, drag_depth_tolerance_ratio
    
    if bvh_data is None or gaussians is None:
        return None
        
    try:
        ray = gsplat_bvh.Ray(
            ray_origin.astype(np.float32),
            ray_direction.astype(np.float32)
        )
        
        result = gsplat_bvh.intersect_ray_bvh(
            ray,
            bvh_data[0],  # bvh_nodes
            bvh_data[1],  # aabbs
        )
        
        if len(result) == 0:
            return None
            
        result = sorted(result, key=lambda x: x[1])
        closest_distance = result[0][1]
        
        if drag_reference_depth is not None and filter:
            depth_tolerance = drag_reference_depth * drag_depth_tolerance_ratio
            depth_diff = abs(closest_distance - drag_reference_depth)
            
            if depth_diff > depth_tolerance:
                return None
        
        best_idx = result[0][0]
        
        drag_reference_depth = closest_distance
        return best_idx
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

def highlight_current_selection(color=[1.0, 1.0, 0.0]):
    global previously_selected
    
    if current_gaussians is None or len(current_selection) == 0:
        return
    
    current_gaussians.sh[previously_selected] = original_sh_backup[previously_selected]
    
    sh_color = np.array(color, dtype=np.float32)
    sh_coeffs = (sh_color - 0.5) / SH_C0  
    current_gaussians.sh[current_selection, 0:3] = sh_coeffs
    
    previously_selected = current_selection.copy()

def clear_all_highlights():
    global previously_selected, cached_geodesic_distances, current_selection
    if current_gaussians is not None and original_sh_backup is not None:
        # total_to_clear = len(previously_selected) + len(current_selection)
        current_gaussians.sh[:] = original_sh_backup[:]
        previously_selected = []
        current_selection = []  
    cached_geodesic_distances = None

def create_sphere_mesh(center, radius=0.1, color=[1.0, 0.0, 0.0]):
    lon_segments = 16  
    lat_segments = 12  
    
    vertices = []
    normals = []
    
    for lat in range(lat_segments + 1):
        theta = np.pi * lat / lat_segments  
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        for lon in range(lon_segments + 1):
            phi = 2 * np.pi * lon / lon_segments 
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            
            x = radius * sin_theta * cos_phi
            y = radius * cos_theta
            z = radius * sin_theta * sin_phi
            
            vertex = np.array([x, y, z]) + center
            vertices.append(vertex)
            
            normal = np.array([x, y, z]) / radius
            normals.append(normal)
    
    vertices = np.array(vertices, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    
    faces = []
    for lat in range(lat_segments):
        for lon in range(lon_segments):
            first = lat * (lon_segments + 1) + lon
            second = first + lon_segments + 1
            faces.append([first, second, first + 1])
            faces.append([second, second + 1, first + 1])
    
    faces = np.array(faces, dtype=np.uint32)
    colors = np.tile(color, (len(vertices), 1)).astype(np.float32)
    sphere_mesh = MeshData(vertices, faces, normals, colors)
    return sphere_mesh

def handle_sphere_selection(window):
    global sphere_selection_points, sphere_click_count, current_selection
    
    if current_gaussians is None or current_bvh is None:
        return
    
    mouse_x, mouse_y = glfw.get_cursor_pos(window)
    ray_origin, ray_direction = screen_to_ray(mouse_x, mouse_y, camera, window_width, window_height)
    intersected_idx = find_intersected_gaussian(ray_origin, ray_direction, current_bvh, current_gaussians)
    
    if intersected_idx is not None:
        gaussian_position = current_gaussians.xyz[intersected_idx]
        if current_gaussians is not None:
            scene_scale = np.max(np.std(current_gaussians.xyz, axis=0))
            sphere_radius = max(0.05, scene_scale * 0.02)  
        else:
            sphere_radius = 0.1  
        
        if sphere_click_count == 0:
            sphere_selection_points = [{
                "position": gaussian_position.copy(),
                "gaussian_idx": intersected_idx,
                "mesh": create_sphere_mesh(gaussian_position, sphere_radius, [1.0, 0.0, 0.0])  
            }]
            sphere_click_count = 1
            
        elif sphere_click_count == 1:
            sphere_selection_points.append({
                "position": gaussian_position.copy(),
                "gaussian_idx": intersected_idx,
                "mesh": create_sphere_mesh(gaussian_position, sphere_radius, [0.0, 1.0, 0.0]) 
            })
            sphere_click_count = 2
            
            perform_sphere_geodesic_selection()
            
      
        elif sphere_click_count == 2:
            clear_all_highlights()  
            current_selection = []  
            
            sphere_selection_points = [{
                "position": gaussian_position.copy(),
                "gaussian_idx": intersected_idx,
                "mesh": create_sphere_mesh(gaussian_position, sphere_radius, [1.0, 0.0, 0.0])  
            }]
            sphere_click_count = 1

def handle_root_reposition(window):
    global root_idx, cached_root_distances, sparse_heat_solver, original_to_sparse_mapping
    
    if current_gaussians is None or current_bvh is None:
        return
    mouse_x, mouse_y = glfw.get_cursor_pos(window)
    ray_origin, ray_direction = screen_to_ray(mouse_x, mouse_y, camera, window_width, window_height)
    intersected_idx = find_intersected_gaussian(ray_origin, ray_direction, current_bvh, current_gaussians, filter=False)
    if intersected_idx is not None:
        # old_root_idx = root_idx
        root_idx = intersected_idx
        if sparse_heat_solver is not None and original_to_sparse_mapping is not None:
            sparse_root_idx = original_to_sparse_mapping[root_idx]
            cached_root_distances = sparse_heat_solver.compute_distance(sparse_root_idx)

def perform_sphere_geodesic_selection():
    global sphere_selection_points, current_selection, heat_solver
    
    if len(sphere_selection_points) != 2:
        return
    
    if heat_solver is None:
        return
    
    red_sphere_idx = sphere_selection_points[0]["gaussian_idx"]
    green_sphere_idx = sphere_selection_points[1]["gaussian_idx"]
    
    try:
        clear_all_highlights()
        red_to_all_distances = heat_solver.compute_distance(red_sphere_idx)
        threshold_distance = red_to_all_distances[green_sphere_idx]
        
        current_selection = np.where(red_to_all_distances <= threshold_distance)[0].tolist()
        current_selection = list(set(current_selection))
        current_selection.sort(key=lambda x: cached_root_distances[x], reverse=True)
        
        highlight_current_selection([1.0, 1.0, 0.0]) 
    except Exception as e:
        import traceback
        traceback.print_exc()

def test_segment_ray_intersection(segment_indices, ray_origin, ray_direction):

    if current_gaussians is None or current_bvh is None or len(segment_indices) == 0:
        return None
    
    try:
        ray = gsplat_bvh.Ray(
            ray_origin.astype(np.float32),
            ray_direction.astype(np.float32)
        )
        
        result = gsplat_bvh.intersect_ray_bvh(
            ray,
            current_bvh[0],  # bvh_nodes
            current_bvh[1],  # aabbs
        )
        
        if result is None or len(result) == 0:
            return None
        
        segment_indices_set = set(segment_indices)
        min_distance = float('inf')
        has_intersection = False
        
        for intersection in result:
            if hasattr(intersection, 'gaussian_idx') and hasattr(intersection, 'distance'):
                gaussian_idx = intersection.gaussian_idx
                distance = intersection.distance
            elif isinstance(intersection, (tuple, list)) and len(intersection) >= 2:
                gaussian_idx, distance = intersection[0], intersection[1]
            
            if gaussian_idx in segment_indices_set and distance > 0:
                if distance < min_distance:
                    min_distance = distance
                    has_intersection = True
        
        return min_distance if has_intersection else None
        
    except Exception as e:
        if current_gaussians is None or len(segment_indices) == 0:
            return None
            
        segment_points = current_gaussians.xyz[segment_indices]
        segment_scales = current_gaussians.scale[segment_indices]
        
        min_distance = float('inf')
        has_intersection = False
        
        for i, (point, scale) in enumerate(zip(segment_points, segment_scales)):
            radius = np.max(scale) * g_scale_modifier
            
            oc = ray_origin - point
            
            b = np.dot(oc, ray_direction)
            c = np.dot(oc, oc) - radius * radius
            discriminant = b * b - c
            
            if discriminant >= 0:
                sqrt_discriminant = np.sqrt(discriminant)
                t1 = -b - sqrt_discriminant
                t2 = -b + sqrt_discriminant
                
                if t1 > 0:
                    distance = t1
                elif t2 > 0:
                    distance = t2
                else:
                    continue  
                
                if distance < min_distance:
                    min_distance = distance
                    has_intersection = True
        
        return min_distance if has_intersection else None

def find_hovered_segment(mouse_x, mouse_y):
    global segments, camera, window_width, window_height
    
    if len(segments) < 1:
        return None  
    
    ray_origin, ray_direction = screen_to_ray(mouse_x, mouse_y, camera, window_width, window_height)
    if ray_origin is None:
        return None
    
    closest_segment_id = None
    min_distance = float('inf')
    
    for i, segment in enumerate(segments):
        segment_indices = segment["indices"]
        distance = test_segment_ray_intersection(segment_indices, ray_origin, ray_direction)
        
        if distance is not None and distance < min_distance:
            min_distance = distance
            closest_segment_id = i
    
    return closest_segment_id

def update_segment_hover_optimized(mouse_x, mouse_y):
    global hovered_segment_id, segment_original_colors, segments
    global last_hover_check_pos, last_hover_check_time, hover_check_interval
    global bvh_result_cache, bvh_cache_mouse_pos
    
    current_time = time.time()
    
    pos_threshold = 5  
    if (abs(mouse_x - last_hover_check_pos[0]) < pos_threshold and 
        abs(mouse_y - last_hover_check_pos[1]) < pos_threshold and
        current_time - last_hover_check_time < hover_check_interval):
        return  
    
    last_hover_check_time = current_time
    last_hover_check_pos = (mouse_x, mouse_y)
    
    new_hovered_id = find_hovered_segment_optimized(mouse_x, mouse_y)
    
    if new_hovered_id != hovered_segment_id:
        if hovered_segment_id is not None and hovered_segment_id < len(segments):
            segment = segments[hovered_segment_id]
            if hovered_segment_id in segment_original_colors:
                segment["color"] = segment_original_colors[hovered_segment_id]
                if segment.get("original_data") is not None:
                    update_segment_color_in_data(segment, segment_original_colors[hovered_segment_id])
        
        if new_hovered_id is not None:
            segment = segments[new_hovered_id]
            if new_hovered_id not in segment_original_colors:
                segment_original_colors[new_hovered_id] = segment["color"].copy()
            segment["color"] = np.array([1.0, 0.84, 0.0]) 
            
            if segment.get("original_data") is not None:
                update_segment_color_in_data(segment, [1.0, 0.84, 0.0])
        
        hovered_segment_id = new_hovered_id

def find_hovered_segment_optimized(mouse_x, mouse_y):
    global segments, camera, window_width, window_height
    global bvh_result_cache, bvh_cache_mouse_pos
    
    if len(segments) < 1:
        return None
    
    pos_threshold = 2  
    if (bvh_result_cache is not None and
        abs(mouse_x - bvh_cache_mouse_pos[0]) < pos_threshold and
        abs(mouse_y - bvh_cache_mouse_pos[1]) < pos_threshold):
        bvh_intersections = bvh_result_cache
    else:
        ray_origin, ray_direction = screen_to_ray(mouse_x, mouse_y, camera, window_width, window_height)
        if ray_origin is None:
            return None
        
        try:
            ray = gsplat_bvh.Ray(
                ray_origin.astype(np.float32),
                ray_direction.astype(np.float32)
            )
            
            bvh_intersections = gsplat_bvh.intersect_ray_bvh(
                ray,
                current_bvh[0],  # bvh_nodes
                current_bvh[1],  # aabbs
            )
            
            bvh_result_cache = bvh_intersections
            bvh_cache_mouse_pos = (mouse_x, mouse_y)
            
        except Exception as e:
            return None
    
    if bvh_intersections is None or len(bvh_intersections) == 0:
        return None
    
    segment_indices_sets = []
    for segment in segments:
        segment_indices_sets.append(set(segment["indices"]))
    
    closest_segment_id = None
    min_distance = float('inf')
    
    for intersection in bvh_intersections:
        if hasattr(intersection, 'gaussian_idx') and hasattr(intersection, 'distance'):
            gaussian_idx = intersection.gaussian_idx
            distance = intersection.distance
        elif isinstance(intersection, (tuple, list)) and len(intersection) >= 2:
            gaussian_idx, distance = intersection[0], intersection[1]
        else:
            continue
        
        if distance <= 0:
            continue
        
        for i, segment_indices_set in enumerate(segment_indices_sets):
            if gaussian_idx in segment_indices_set:
                if distance < min_distance:
                    min_distance = distance
                    closest_segment_id = i
                break  
    
    return closest_segment_id

def draw_segment_hover_tooltip():
    global hovered_segment_id, segments, mouse_position, window_width, window_height
    
    if hovered_segment_id is None or hovered_segment_id >= len(segments):
        return
    
    segment = segments[hovered_segment_id]
    segment_name = segment.get("name", f"Segment {hovered_segment_id}")
    
    mouse_x, mouse_y = mouse_position
    tooltip_offset_x = 15
    tooltip_offset_y = 15
    
    tooltip_x = mouse_x + tooltip_offset_x
    tooltip_y = mouse_y + tooltip_offset_y
    
    tooltip_width = len(segment_name) * 8 + 16  
    if tooltip_x + tooltip_width > window_width:
        tooltip_x = mouse_x - tooltip_width - 5  
    
    tooltip_height = 30  
    if tooltip_y + tooltip_height > window_height:
        tooltip_y = mouse_y - tooltip_height - 5  
    
    imgui.set_next_window_position(tooltip_x, tooltip_y)
    imgui.set_next_window_size(0, 0)  
    
    window_flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | 
                   imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR |
                   imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_SAVED_SETTINGS |
                   imgui.WINDOW_NO_FOCUS_ON_APPEARING | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS)
    
    imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.84, 0.0, 1.0)  
    
    expanded, opened = imgui.begin(f"##segment_tooltip", True, window_flags)
    if expanded:
        imgui.text(f"Segment: {segment_name}")
    imgui.end()
    
    imgui.pop_style_color(1)

def update_segment_color_in_data(segment, color):
    if segment.get("original_data") is None:
        return
    
    segment_data = segment["colored_data"]
    target_color = np.array(color, dtype=np.float32)
    
    num_points = len(segment_data.xyz)
    pure_rgb = np.tile(target_color, (num_points, 1))
    pure_sh_dc = (pure_rgb - 0.5) / SH_C0
    
    segment_data.sh[:, 0:3] = pure_sh_dc  

def clear_segment_hover():
    global hovered_segment_id, segment_original_colors, segments
    global bvh_result_cache, bvh_cache_mouse_pos
    
    if hovered_segment_id is not None and hovered_segment_id < len(segments):
        segment = segments[hovered_segment_id]
        if hovered_segment_id in segment_original_colors:
            segment["color"] = segment_original_colors[hovered_segment_id]
            if segment.get("original_data") is not None:
                update_segment_color_in_data(segment, segment_original_colors[hovered_segment_id])
    
    hovered_segment_id = None
    bvh_result_cache = None
    bvh_cache_mouse_pos = (-1, -1)

def handle_segment_click(window):
    global segments, segments_expanded, current_bvh, current_gaussians
    mouse_x, mouse_y = glfw.get_cursor_pos(window)
    clicked_segment_id = find_hovered_segment_optimized(mouse_x, mouse_y)
    
    if clicked_segment_id is not None and clicked_segment_id < len(segments):
        segment = segments[clicked_segment_id]
        segment_name = segment["name"]
        segments_expanded[segment_name] = True
        

def clear_sphere_selection():
    global sphere_selection_points, sphere_click_count, current_selection
    sphere_selection_points = []
    sphere_click_count = 0
    current_selection = []
    clear_all_highlights()

def apply_pca_transformation_to_segment(target_segment, source_template_mesh = None):
    if source_template_mesh is None:
        global template_meshes
        
        source_segment = None
        source_template_mesh = None
        for mesh in template_meshes:
            if mesh.get("source_segment") is not None:
                source_template_mesh = mesh
                break
    
    try:
        source_segment_name = source_template_mesh["source_segment"]
        for segment in segments:
            if segment["name"] == source_segment_name:
                source_segment = segment
                break
            
        transformation_matrix = get_transform_template_mesh_pca(
            source_segment,
            target_segment,
            original_plant_gaussians.xyz[root_idx]
        )
        
        original_mesh_data = source_template_mesh["mesh_data"]
        transformed_vertices = original_mesh_data.vertices.copy()
        
        homogeneous_vertices = np.hstack([transformed_vertices, np.ones((len(transformed_vertices), 1))])
        transformed_homogeneous = (transformation_matrix @ homogeneous_vertices.T).T
        transformed_vertices = transformed_homogeneous[:, :3]
        
        original_gaussian_data = source_segment["original_data"]
        transformed_gaussian_data = apply_transformation_matrix_to_points(
            original_gaussian_data.xyz,
            transformation_matrix
        )
        
        transformed_mesh_data = MeshData(
            vertices=transformed_vertices.astype(np.float32),
            faces=original_mesh_data.faces.copy(),
            normals=original_mesh_data.normals.copy(),  
            colors=original_mesh_data.colors.copy(),
            uvs=original_mesh_data.uvs.copy(),
            has_texture=original_mesh_data.has_texture,
            texture_path=original_mesh_data.texture_path
        )
        
        if hasattr(original_mesh_data, 'texture_data'):
            transformed_mesh_data.texture_data = original_mesh_data.texture_data
        
        transformed_template_mesh = {
            "name": f"PCA_{target_segment['name']}_from_{source_segment_name}",
            "source_segment": source_segment_name,  
            "target_segment": target_segment["name"],  
            "created_time": "PCA transformed",
            "mesh_data": transformed_mesh_data,
            "visible": True,
            "transformation_type": "PCA",  
            "hidden_gaussian_indices": target_segment["indices"].copy() if target_segment.get("indices") is not None else None 
        }
        
        template_meshes.append(transformed_template_mesh)
        
        current_gaussians.opacity[target_segment["indices"]] = 0.0
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
    return transformed_template_mesh, transformed_gaussian_data

# ONLY FPS
def apply_mls_fps_transformation_to_segment(target_segment, source_template_mesh = None):
    if source_template_mesh is None:
        global template_meshes
        
        source_segment = None
        source_template_mesh = None
        for mesh in template_meshes:
            if mesh.get("source_segment") is not None:
                source_template_mesh = mesh
                break
    
    try:
        source_segment_name = source_template_mesh["source_segment"]
        for segment in segments:
            if segment["name"] == source_segment_name:
                source_segment = segment
                break

        original_mesh_data = source_template_mesh["mesh_data"]
        original_vertices = original_mesh_data.vertices.copy()
        
        global mls_num_corr
        transformed_vertices, transformed_gaussian_data,  corr_pair = get_transform_template_mesh_mls_corr(
            source_segment,
            target_segment,
            original_vertices,
            root_point=original_plant_gaussians.xyz[root_idx],
            num_corr=mls_num_corr,
            corr_weights=np.ones((mls_num_corr, 1), dtype=np.float32),
            sigma=0.1
        )
        
        transformed_mesh_data = MeshData(
            vertices=transformed_vertices.astype(np.float32),
            faces=original_mesh_data.faces.copy(),
            normals=original_mesh_data.normals.copy(),  
            colors=original_mesh_data.colors.copy(),
            uvs=original_mesh_data.uvs.copy(),
            has_texture=original_mesh_data.has_texture,
            texture_path=original_mesh_data.texture_path
        )
        
        if hasattr(original_mesh_data, 'texture_data'):
            transformed_mesh_data.texture_data = original_mesh_data.texture_data
        
        transformed_template_mesh = {
            "name": f"MLS_{target_segment['name']}_from_{source_segment_name}",
            "source_segment": source_segment_name,  
            "target_segment": target_segment["name"],  
            "created_time": "MLS transformed",
            "mesh_data": transformed_mesh_data,
            "visible": True,
            "transformation_type": "MLS", 
            "hidden_gaussian_indices": target_segment["indices"].copy() if target_segment.get("indices") is not None else None  
        }
        
        template_meshes.append(transformed_template_mesh)
        
        current_gaussians.opacity[target_segment["indices"]] = 0.0
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
    return transformed_template_mesh, transformed_gaussian_data, corr_pair

# ONLY MVC
def apply_mls_mvc_transformation_to_segment(target_segment, source_template_mesh = None):
    if source_template_mesh is None:
        global template_meshes
        
        source_segment = None
        source_template_mesh = None
        for mesh in template_meshes:
            if mesh.get("source_segment") is not None:
                source_template_mesh = mesh
                break
    
    try:
        source_segment_name = source_template_mesh["source_segment"]
        for segment in segments:
            if segment["name"] == source_segment_name:
                source_segment = segment
                break

        original_mesh_data = source_template_mesh["mesh_data"]
        original_vertices = original_mesh_data.vertices.copy()
        
        source_mvc = source_segment.get("mvc", None)
        target_mvc = target_segment.get("mvc", None)
        
        source_coordinates_2d = source_mvc.get("coordinates_2d", None)
        target_coordinates_2d = target_mvc.get("coordinates_2d", None)
        
        source_add_info = source_segment.get("add_info", {})
        target_add_info = target_segment.get("add_info", {})
        source_x_range = source_add_info.get("x_range", None)
        source_y_range = source_add_info.get("y_range", None)
        target_x_range = target_add_info.get("x_range", None)
        target_y_range = target_add_info.get("y_range", None)
        source_image_size = source_add_info.get("image_size", None)
        target_image_size = target_add_info.get("image_size", None)
        source_depth_map = source_add_info.get("depth_map", None)
        target_depth_map = target_add_info.get("depth_map", None)
        source_transform_matrix = source_add_info.get("transformation_matrix", None)
        target_transform_matrix = target_add_info.get("transformation_matrix", None)
        
        source_transform_matrix_inv = np.linalg.inv(source_transform_matrix)
        target_transform_matrix_inv = np.linalg.inv(target_transform_matrix)
        
        source_coordinates_3d = project_2d_coords_to_3d(source_coordinates_2d, source_depth_map, source_x_range, source_y_range, source_image_size)
        target_coordinates_3d = project_2d_coords_to_3d(target_coordinates_2d, target_depth_map, target_x_range, target_y_range, target_image_size)
        
        source_coordinates_3d_homo = np.hstack([source_coordinates_3d, np.ones((source_coordinates_3d.shape[0], 1))])
        target_coordinates_3d_homo = np.hstack([target_coordinates_3d, np.ones((target_coordinates_3d.shape[0], 1))])
        
        source_aligned = (source_transform_matrix_inv @ source_coordinates_3d_homo.T).T[:, :3]
        target_aligned = (target_transform_matrix_inv @ target_coordinates_3d_homo.T).T[:, :3]
        
        additional_corr_pair = (source_aligned, target_aligned)
        transformed_vertices, transformed_gaussian_data, corr_pair = get_transform_template_mesh_mls_corr_kai(
            source_segment,
            target_segment,
            original_vertices,
            num_corr=0,
            corr_weights=None,
            sigma=0.1,
            additional_corr_pair=additional_corr_pair
        )
        
        transformed_mesh_data = MeshData(
            vertices=transformed_vertices.astype(np.float32),
            faces=original_mesh_data.faces.copy(),
            normals=original_mesh_data.normals.copy(),  
            colors=original_mesh_data.colors.copy(),
            uvs=original_mesh_data.uvs.copy(),
            has_texture=original_mesh_data.has_texture,
            texture_path=original_mesh_data.texture_path
        )
        
        if hasattr(original_mesh_data, 'texture_data'):
            transformed_mesh_data.texture_data = original_mesh_data.texture_data
        
        transformed_template_mesh = {
            "name": f"MLS_{target_segment['name']}_from_{source_segment_name}",
            "source_segment": source_segment_name, 
            "target_segment": target_segment["name"], 
            "created_time": "MLS transformed",
            "mesh_data": transformed_mesh_data,
            "visible": True,
            "transformation_type": "MLS",  
            "hidden_gaussian_indices": target_segment["indices"].copy() if target_segment.get("indices") is not None else None 
        }
        
        template_meshes.append(transformed_template_mesh)
        
        current_gaussians.opacity[target_segment["indices"]] = 0.0
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
    return transformed_template_mesh, transformed_gaussian_data, corr_pair

def apply_mls_fps_optim_transformation_to_segment(target_segment, source_template_mesh = None, only_edge=False, both=False):
    if both and only_edge:
        raise ValueError("both and only_edge cannot be True at the same time")
    
    if source_template_mesh is None:
        global template_meshes
        
        source_segment = None
        source_template_mesh = None
        for mesh in template_meshes:
            if mesh.get("source_segment") is not None:
                source_template_mesh = mesh
                break
    
    try:
        source_segment_name = source_template_mesh["source_segment"]
        for segment in segments:
            if segment["name"] == source_segment_name:
                source_segment = segment
                break

        original_mesh_data = source_template_mesh["mesh_data"]
        original_vertices = original_mesh_data.vertices.copy()
        

        global mls_num_corr

        if both:
            num_corr = mls_num_corr
            source_additional_indices = source_segment.get("edge_indices", None)
            target_additional_indices = target_segment.get("edge_indices", None)
            
            additional_corr_indices_pair = (
                source_additional_indices,
                target_additional_indices
            )
        else:
            if only_edge:
                num_corr = 0
                source_additional_indices = source_segment.get("edge_indices", None)
                target_additional_indices = target_segment.get("edge_indices", None)
                
                additional_corr_indices_pair = (
                    source_additional_indices,
                    target_additional_indices
                )
            else:
                num_corr = mls_num_corr
                additional_corr_indices_pair = None
        transformed_vertices, transformed_gaussian_data, corr_pair = get_transform_template_mesh_mls_corr_optim(
            source_segment,
            target_segment,
            original_vertices,
            num_corr=num_corr,
            corr_weights=None,
            root_point=original_plant_gaussians.xyz[root_idx],
            sigma=0.1,
            steps=200, 
            image_size=64,  
            lr_rate=7e-3,  
            additional_corr_indices_pair=additional_corr_indices_pair,
        )
        
        transformed_mesh_data = MeshData(
            vertices=transformed_vertices.astype(np.float32),
            faces=original_mesh_data.faces.copy(),
            normals=original_mesh_data.normals.copy(), 
            colors=original_mesh_data.colors.copy(),
            uvs=original_mesh_data.uvs.copy(),
            has_texture=original_mesh_data.has_texture,
            texture_path=original_mesh_data.texture_path
        )
        
        if hasattr(original_mesh_data, 'texture_data'):
            transformed_mesh_data.texture_data = original_mesh_data.texture_data
        
        transformed_template_mesh = {
            "name": f"MLS_{target_segment['name']}_from_{source_segment_name}",
            "source_segment": source_segment_name, 
            "target_segment": target_segment["name"], 
            "created_time": "MLS transformed",
            "mesh_data": transformed_mesh_data,
            "visible": True,
            "transformation_type": "MLS", 
            "hidden_gaussian_indices": target_segment["indices"].copy() if target_segment.get("indices") is not None else None 
        }
        
        template_meshes.append(transformed_template_mesh)
        
        current_gaussians.opacity[target_segment["indices"]] = 0.0
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
    return transformed_template_mesh, transformed_gaussian_data, corr_pair

def apply_mls_mvc_optim_transformation_to_segment(target_segment, source_template_mesh = None, with_edge=False):
    if source_template_mesh is None:
        global template_meshes
        
        source_segment = None
        source_template_mesh = None
        for mesh in template_meshes:
            if mesh.get("source_segment") is not None:
                source_template_mesh = mesh
                break
    
    try:
        source_segment_name = source_template_mesh["source_segment"]
        for segment in segments:
            if segment["name"] == source_segment_name:
                source_segment = segment
                break

        original_mesh_data = source_template_mesh["mesh_data"]
        original_vertices = original_mesh_data.vertices.copy()
        
        source_mvc = source_segment.get("mvc", None)
        target_mvc = target_segment.get("mvc", None)
        
        source_coordinates_2d = source_mvc.get("coordinates_2d", None)
        target_coordinates_2d = target_mvc.get("coordinates_2d", None)
        
        source_add_info = source_segment.get("add_info", {})
        target_add_info = target_segment.get("add_info", {})
        source_x_range = source_add_info.get("x_range", None)
        source_y_range = source_add_info.get("y_range", None)
        target_x_range = target_add_info.get("x_range", None)
        target_y_range = target_add_info.get("y_range", None)
        source_image_size = source_add_info.get("image_size", None)
        target_image_size = target_add_info.get("image_size", None)
        source_depth_map = source_add_info.get("depth_map", None)
        target_depth_map = target_add_info.get("depth_map", None)

        source_coordinates_3d = project_2d_coords_to_3d(source_coordinates_2d, source_depth_map, source_x_range, source_y_range, source_image_size)
        target_coordinates_3d = project_2d_coords_to_3d(target_coordinates_2d, target_depth_map, target_x_range, target_y_range, target_image_size)

        if with_edge:
            source_additional_indices = None
            target_additional_indices = None
            if source_segment.get("edge_indices") is not None:
                source_additional_indices = source_segment["edge_indices"]
            if target_segment.get("edge_indices") is not None:
                target_additional_indices = target_segment["edge_indices"]
            
            if source_additional_indices is not None and target_additional_indices is not None:
                final_additional_corr_pair = (
                    np.concatenate([source_additional_indices, source_coordinates_3d]),
                    np.concatenate([target_additional_indices, target_coordinates_3d])
                )
            else:
                raise ValueError("Source segment does not have edge indices")
        else:
            final_additional_corr_pair = (
                source_coordinates_3d,
                target_coordinates_3d
            )
        
        transformed_vertices, transformed_gaussian_data, corr_pair = get_transform_template_mesh_mls_corr_optim(
            source_segment,
            target_segment,
            original_vertices,
            num_corr=0,
            root_point=original_plant_gaussians.xyz[root_idx],
            corr_weights=None,
            sigma=0.1,
            steps=200,  
            image_size=64, 
            lr_rate=7e-3,  
            additional_corr_indices_pair=final_additional_corr_pair,
        )
        
        transformed_mesh_data = MeshData(
            vertices=transformed_vertices.astype(np.float32),
            faces=original_mesh_data.faces.copy(),
            normals=original_mesh_data.normals.copy(),  
            colors=original_mesh_data.colors.copy(),
            uvs=original_mesh_data.uvs.copy(),
            has_texture=original_mesh_data.has_texture,
            texture_path=original_mesh_data.texture_path
        )
        
        if hasattr(original_mesh_data, 'texture_data'):
            transformed_mesh_data.texture_data = original_mesh_data.texture_data
        
        
        transformed_template_mesh = {
            "name": f"MLS_{target_segment['name']}_from_{source_segment_name}",
            "source_segment": source_segment_name,  
            "target_segment": target_segment["name"],  
            "created_time": "MLS transformed",
            "mesh_data": transformed_mesh_data,
            "visible": True,
            "transformation_type": "MLS",  
            "hidden_gaussian_indices": target_segment["indices"].copy() if target_segment.get("indices") is not None else None  
        }
        
        template_meshes.append(transformed_template_mesh)
        
        
        current_gaussians.opacity[target_segment["indices"]] = 0.0
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
    return transformed_template_mesh, transformed_gaussian_data, corr_pair

def manual_segment(selected_indices, segment_name, apex_idx=None, base_idx=None):
    global original_plant_gaussians, segments
    
    if original_plant_gaussians is None or len(selected_indices) == 0:
        print("Error: No available backup Gaussian data or selected indices are empty")
        return
    
    selected_indices = np.array(sorted(selected_indices))
    
    print(f"Create manual segment: {segment_name}, containing {len(selected_indices)} Gaussians")
    
    segment_index = int(segment_name.split("_")[-1]) - 1
    
    if segment_index > len(colors) - 1:
        segment_color = generate_non_gold_color()
    else:
        segment_color = colors[segment_index]
    
    labels = calculate_segment_labels(original_plant_gaussians, 
                                      selected_indices, apex_idx, base_idx)
    
    colored_segment_data, original_segment_data = create_segment_from_gaussians(
        original_plant_gaussians,
        selected_indices, 
        segment_color,
        opacity_factor=0.7
    )
    

    new_segment = {
        "name": segment_name,
        "indices": selected_indices,
        "color": segment_color,
        "visible": True,
        "colored_data": colored_segment_data,
        "original_data": original_segment_data,
        "labels": labels,  
        "apex_idx": apex_idx,  
        "base_idx": base_idx,  
        "apex_point": original_plant_gaussians.xyz[apex_idx],
        "base_point": original_plant_gaussians.xyz[base_idx],
        "is_auto": False,  
        "source_data": original_plant_gaussians  
    }
    segments.append(new_segment)
    
    print(f"Manual segment created: {segment_name}")

def estimate_mvc_from_segment(segment, 
                              grid_density=20,
                              boundary_margin=0.1,
                              distance_factor=0.9,
                              max_fps_points=5,
                              source_weights=None):

    try:
        segment_name = segment['name']
        
        
        if source_weights is None:
            mvc_weights, coordinates_2d = compute_source_mvc_weights(
                add_info=segment.get("add_info", None),
                grid_density=grid_density,
                boundary_margin=boundary_margin,
                distance_factor=distance_factor,
                max_fps_points=max_fps_points
            )
            
            segment["mvc"] = {"weights": mvc_weights,
                              "coordinates_2d": coordinates_2d}
            return mvc_weights, coordinates_2d
            
        else:
            target_boundary_polygon = extract_leaf_boundary_polygon(segment.get("add_info", None))
            
            target_coordinates_2d = reconstruct_target_coordinates_from_weights(
                target_boundary_polygon, source_weights
            )
            
            segment["mvc"] = {"weights": source_weights, 
                              "coordinates_2d": target_coordinates_2d}
            return target_coordinates_2d
        
    except Exception as e:
        print(f"Failed to estimate MVC from segment {segment.get('name', 'unknown')}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def estimate_edge_from_segment(segment, vis=False):
    global edge_visualizations
    try:
        print(f"Start estimating edge from segment: {segment['name']}")
        
        indices = segment["indices"]
        original_gaussian_data = segment["original_data"]
        segment_tip = segment["apex_point"]
        segment_base = segment["base_point"]
        denoised_indices = segment.get("denoised_indices", None)
        if denoised_indices is None:
            denoised_indices, _, _ = mls_denoising(original_gaussian_data.xyz, 0.1, expand=1.25)
        indices = indices[denoised_indices]
        original_gaussian_data = apply_indices_to_gaussian_data(original_gaussian_data, denoised_indices)
        
        segment_labels = segment["labels"]  
        segment_labels = segment_labels[denoised_indices]
        global edge_sampling_count
        debug = True
        edge_points_indices, add_info = compute_3d_edge_points_from_gaussian(
            original_gaussian_data, 
            segment_labels, 
            image_size=512, 
            tip_point=segment_tip,
            base_point=segment_base,
            num_samples_per_path=edge_sampling_count,
            debug=debug
        )
        segment["add_info"] = add_info
        if debug:
            if edge_points_indices is not None and len(edge_points_indices) > 0:
                transformation_matrix = add_info['transformation_matrix']
                transformation_matrix = np.linalg.inv(transformation_matrix)
                
                leaf_scale = np.max(np.std(original_plant_gaussians.xyz[indices], axis=0))
                sphere_radius = max(0.02, leaf_scale * 0.05)  
                
                if len(edge_points_indices) >= 2:  
                    total_points = len(edge_points_indices)
                    print(f"Total edge points: {total_points}")
                    
                    edge_groups = []
                    current_idx = 0
                    
                    if current_idx < total_points:
                        apex_3d_pos = edge_points_indices[current_idx]
                        edge_groups.append({
                            "type": "apex",
                            "color": colors_edge["apex"],
                            "position": apex_3d_pos,
                            "index": current_idx
                        })
                        current_idx += 1
                    
                    left_samples_count = len(segment["add_info"].get("left_sampled_points", []))
                    right_samples_count = len(segment["add_info"].get("right_sampled_points", []))
                    
                    print(f"📊 Edge point structure:")
                    print(f"   Total points: {total_points}")  
                    print(f"   Expected: 1(apex) + {left_samples_count}(left) + 1(base) + {right_samples_count}(right) = {1 + left_samples_count + 1 + right_samples_count}")
                    
                    for i in range(left_samples_count):
                        if current_idx < total_points:
                            left_3d_pos = edge_points_indices[current_idx]
                            edge_groups.append({
                                "type": "left",
                                "color": colors_edge["left"],
                                "position": left_3d_pos,
                                "index": current_idx
                            })
                            current_idx += 1
                    
                    if current_idx < total_points:
                        base_3d_pos = edge_points_indices[current_idx]
                        edge_groups.append({
                            "type": "base",
                            "color": colors_edge["base"],
                            "position": base_3d_pos,
                            "index": current_idx
                        })
                        print(f"   🎯 Base point at index {current_idx}: {base_3d_pos}")
                        current_idx += 1
                    
                    for i in range(right_samples_count):
                        if current_idx < total_points:
                            right_3d_pos = edge_points_indices[current_idx]
                            edge_groups.append({
                                "type": "right",
                                "color": colors_edge["right"],
                                "position": right_3d_pos,
                                "index": current_idx
                            })
                            current_idx += 1
                    
                    edge_points_record = {
                        "name": f"{segment['name']}_edge_points",
                        "segment_name": segment["name"],
                        "edge_points_indices": edge_points_indices,  
                        "type": "edge_points_data",
                        "visible": False  
                    }
                    edge_visualizations.append(edge_points_record)
                    
                    if vis:
                        for group in edge_groups:
                            sphere_mesh_data = create_sphere_mesh(
                                center=group["position"],
                                radius=sphere_radius,
                                color=group["color"]
                            )
                            
                            edge_mesh = {
                                "name": f"{segment['name']}_edge_{group['type']}_{group['index']}",
                                "mesh_data": sphere_mesh_data,  
                                "color": group["color"],
                                "visible": True,
                                "type": "edge_point"
                            }
                            
                            edge_visualizations.append(edge_mesh)
        else:
            if edge_points_indices is not None and len(edge_points_indices) > 0:
                leaf_scale = np.max(np.std(original_plant_gaussians.xyz[indices], axis=0))
                sphere_radius = max(0.02, leaf_scale * 0.05)  
                
                if len(edge_points_indices) >= 2:  
                    total_points = len(edge_points_indices)
                    print(f"Total edge points: {total_points}")
                    
                    edge_groups = []
                    current_idx = 0
                    
                    if current_idx < total_points:
                        segment_internal_idx = edge_points_indices[current_idx]
                        original_idx = indices[segment_internal_idx] 
                        apex_3d_pos = original_plant_gaussians.xyz[original_idx]
                        edge_groups.append({
                            "type": "apex",
                            "color": colors_edge["apex"],
                            "position": apex_3d_pos,
                            "index": current_idx
                        })
                        current_idx += 1
                    
                    left_samples_count = len(segment["add_info"].get("left_sampled_points", []))
                    right_samples_count = len(segment["add_info"].get("right_sampled_points", []))
                    
                    print(f"📊 Edge point structure:")
                    print(f"   Total points: {total_points}")  
                    print(f"   Expected: 1(apex) + {left_samples_count}(left) + 1(base) + {right_samples_count}(right) = {1 + left_samples_count + 1 + right_samples_count}")
                    
                    for i in range(left_samples_count):
                        if current_idx < total_points:
                            segment_internal_idx = edge_points_indices[current_idx]
                            original_idx = indices[segment_internal_idx]
                            left_3d_pos = original_plant_gaussians.xyz[original_idx]
                            edge_groups.append({
                                "type": "left",
                                "color": colors_edge["left"],
                                "position": left_3d_pos,
                                "index": current_idx
                            })
                            current_idx += 1
                    
                    if current_idx < total_points:
                        segment_internal_idx = edge_points_indices[current_idx]
                        original_idx = indices[segment_internal_idx]
                        base_3d_pos = original_plant_gaussians.xyz[original_idx]
                        edge_groups.append({
                            "type": "base",
                            "color": colors_edge["base"],
                            "position": base_3d_pos,
                            "index": current_idx
                        })
                        print(f"   🎯 Base point at index {current_idx}: {base_3d_pos}")
                        current_idx += 1
                    
                    for i in range(right_samples_count):
                        if current_idx < total_points:
                            segment_internal_idx = edge_points_indices[current_idx]
                            original_idx = indices[segment_internal_idx]
                            right_3d_pos = original_plant_gaussians.xyz[original_idx]
                            edge_groups.append({
                                "type": "right",
                                "color": colors_edge["right"],
                                "position": right_3d_pos,
                                "index": current_idx
                            })
                            current_idx += 1
                    
                    edge_points_record = {
                        "name": f"{segment['name']}_edge_points",
                        "segment_name": segment["name"],
                        "edge_points_indices": edge_points_indices, 
                        "type": "edge_points_data",
                        "visible": False  
                    }
                    edge_visualizations.append(edge_points_record)
                    
                    if vis:
                        for group in edge_groups:
                            sphere_mesh_data = create_sphere_mesh(
                                center=group["position"],
                                radius=sphere_radius,
                                color=group["color"]
                            )
                            
                            edge_mesh = {
                                "name": f"{segment['name']}_edge_{group['type']}_{group['index']}",
                                "mesh_data": sphere_mesh_data,  
                                "color": group["color"],
                                "visible": True,
                                "type": "edge_point"
                            }
                            
                            edge_visualizations.append(edge_mesh)
            
    except Exception as e:
        print(f"Failed to estimate edge from segment: {str(e)}")
        import traceback
        traceback.print_exc()

    return edge_points_record

def make_template_leaf_from_segment(segment, image_size=512, dense=False, write_to_disk=False):
    global template_meshes, show_mesh_manager
    if segment["name"].lower() == "stem":
        print("Error: Cannot make template leaf from stem")
        return
    
    tip_point = segment["apex_point"]
    base_point = segment["base_point"]
    original_gaussian_data = segment["original_data"]
    sampled_points = original_gaussian_data.xyz
    denoised_indices, new_points_, new_normals_ = mls_denoising(sampled_points, 0.12, expand=1.)

    vertices, triangles, normals = generate_template_leaf(
        new_points_,
        new_normals_
    )
    segment["leaf"] = {"vertices": vertices, "triangles": triangles}
    
    _, texture_image, mesh_data = generate_uv_mapping_mem(
        apply_indices_to_gaussian_data(original_gaussian_data, denoised_indices), 
        tip_point, base_point,
        vertices, triangles, normals,
        image_size=image_size,
        root_point=original_plant_gaussians.xyz[root_idx]
    )
    
    template_mesh_data = MeshData(
        vertices=mesh_data["vertices"].astype(np.float32),
        faces=mesh_data["faces"].astype(np.uint32),
        normals=mesh_data["normals"].astype(np.float32),  
        colors=np.ones((len(mesh_data["vertices"]), 3), dtype=np.float32) * 0.8,
        uvs=mesh_data["uvs"].astype(np.float32),  
        has_texture=True,
        texture_path=None  
    )
    
    template_mesh_data.texture_data = texture_image
    
    template_mesh = {
        "name": f"Template_{segment['name']}",
        "source_segment": segment['name'],
        "point_count": len(segment["indices"]),
        "created_time": "generated",
        "mesh_data": template_mesh_data,  
        "visible": True,
        "hidden_gaussian_indices": segment["indices"].copy() if segment.get("indices") is not None else None  
    }
    
    template_meshes.append(template_mesh)
    
    current_gaussians.opacity[segment["indices"]] = 0.0
    
    segment["visible"] = False
    show_mesh_manager = True
        
    return template_mesh

def create_combined_gaussian_data():
    all_data_parts = []
    
    if current_gaussians is not None:
        if show_temperature_field and temperature_colors is not None:
            temp_gaussians = apply_temperature_colors(SH_C0, current_gaussians, temperature_colors)
            all_data_parts.append(temp_gaussians)
        else:
            all_data_parts.append(current_gaussians)
    
    for segment in segments:
        if segment["visible"] and segment.get("colored_data") is not None:
            all_data_parts.append(segment["colored_data"])
    
    if len(all_data_parts) == 0:
        return None
    elif len(all_data_parts) == 1:
        return all_data_parts[0]
    
    combined_xyz = np.concatenate([data.xyz for data in all_data_parts], axis=0)
    combined_nxnynz = np.concatenate([data.nxnynz for data in all_data_parts], axis=0)
    combined_rot = np.concatenate([data.rot for data in all_data_parts], axis=0)
    combined_scale = np.concatenate([data.scale for data in all_data_parts], axis=0)
    combined_opacity = np.concatenate([data.opacity for data in all_data_parts], axis=0)
    combined_sh = np.concatenate([data.sh for data in all_data_parts], axis=0)
    combined_filter_3Ds = np.concatenate([data.filter_3Ds for data in all_data_parts], axis=0)
    return GaussianData(
        xyz=combined_xyz,
        rot=combined_rot,
        scale=combined_scale,
        opacity=combined_opacity,
        sh=combined_sh,
        nxnynz=combined_nxnynz,
        filter_3Ds=combined_filter_3Ds
    )

def calculate_segment_labels(gaussian_data, indices, apex_idx=None, base_idx=None):

    import numpy as np
    
    if len(indices) == 0:
        return np.array([])
    
    if apex_idx is None or base_idx is None:
        segment_points = gaussian_data.xyz[indices]
        center = np.mean(segment_points, axis=0)
        distances_from_center = np.linalg.norm(segment_points - center, axis=1)
        threshold = np.percentile(distances_from_center, 30)
        labels = (distances_from_center <= threshold).astype(np.float32)
        return labels
    
    segment_points = gaussian_data.xyz[indices]
    apex_point = gaussian_data.xyz[apex_idx]
    base_point = gaussian_data.xyz[base_idx]
    
    distances_to_apex = np.linalg.norm(segment_points - apex_point, axis=1)
    distances_to_base = np.linalg.norm(segment_points - base_point, axis=1)
    
    labels = (distances_to_base < distances_to_apex).astype(np.float32)
    
    print(f"Using endpoint information to calculate labels: {np.sum(labels)} base points, {len(labels) - np.sum(labels)} apex points")
    return labels

def create_segment_from_gaussians(source_segment_data, indices, color=[1.0, 0.0, 0.0], opacity_factor=0.7):
    selected_xyz = source_segment_data.xyz[indices]
    selected_rot = source_segment_data.rot[indices]
    selected_scale = source_segment_data.scale[indices]
    original_opacity = source_segment_data.opacity[indices]
    original_sh = source_segment_data.sh[indices]
    original_nxnynz = source_segment_data.nxnynz[indices]
    original_filter_3Ds = source_segment_data.filter_3Ds[indices]
    
    selected_opacity = np.clip(original_opacity * opacity_factor, 0., 0.9)  
    target_color = np.array(color, dtype=np.float32)
    pure_rgb = np.tile(target_color, ((selected_xyz.shape[0], 1)))
    pure_sh_dc = (pure_rgb - 0.5) / SH_C0
    selected_sh = np.zeros_like(original_sh)
    selected_sh[:, 0:3] = pure_sh_dc 
    
    colored_gs = GaussianData(
        xyz=selected_xyz,
        rot=selected_rot,
        scale=selected_scale,
        opacity=selected_opacity,
        sh=selected_sh,
        nxnynz=original_nxnynz,
        filter_3Ds=original_filter_3Ds
    )
    
    original_gs = GaussianData(
        xyz=selected_xyz,
        rot=selected_rot,
        scale=selected_scale,
        opacity=original_opacity,
        sh=original_sh,
        nxnynz=original_nxnynz,
        filter_3Ds=original_filter_3Ds
    )
    
    return colored_gs, original_gs

def load_gaussian_file(filepath, given_root_idx = None, debug_vis=False):
    global current_gaussians, current_gaussian_path, gaussian_renderer, gaussian_picker_error, current_bvh, seg_path, mesh_path, deform_path, pack_path
    global cached_geodesic_distances, selection_mode, leaf_tip_idx, current_selection
    global sphere_selection_points, sphere_click_count, root_idx, camera
    global temperature_colors, heat_solver, sparse_heat_solver, leaf_tip_spheres, path_spheres
    global sparse_indices, original_to_sparse_mapping, cached_root_distances
    global segments, segments_expanded, original_plant_gaussians, original_sh_backup
    
    # GLOBAL FOR SEGMENTATION
    global segment_masks, found_tips, found_bases, found_geodist_from_tip, found_clusters

    # plant0
    # camera.set_camera_pose(
    #     position=[-0.265204, -1.080569, -1.652240],
    #     target=[0.711835, 0.950079, 1.506258],
    #     up=[0.000000, -1.000000, 0.000000],
    #     yaw=-1.270796, pitch=-0.550796
    # )

    # plant1
    # camera.set_camera_pose(
    #     position=[-0.294631, -0.789973, -1.492229],
    #     target=[0.821255, 1.174150, 1.662333],
    #     up=[0.000000, -1.000000, 0.000000],
    #     yaw=-1.230796, pitch=-0.530796
    # )

    # plant2
    # camera.set_camera_pose(
    #     position=[0.377244, 0.849293, -2.062506],
    #     target=[0.511435, -0.478254, 1.580863],
    #     up=[0.000000, -1.000000, 0.000000],
    #     yaw=4.749204, pitch=0.349204
    # )

    # # plant3
    # camera.set_camera_pose(
    #     position=[0.492578, -1.346570, -0.813009],
    #     target=[-0.262933, 1.718608, 1.442734],
    #     up=[0.000000, -1.000000, 0.000000],
    #     yaw=4.389204, pitch=-0.910796
    # )

    # # plant4
    # camera.set_camera_pose(
    #     position=[-0.062993, -0.242648, -2.161644],
    #     target=[0.823374, 0.829611, 1.460370],
    #     up=[0.000000, -1.000000, 0.000000],
    #     yaw=-1.330796, pitch=-0.280000
    # )

    # # plant5
    # camera.set_camera_pose(
    #     position=[-0.142804, -1.119612, -2.335229],
    #     target=[0.207654, 0.533031, 1.157666],
    #     up=[0.000000, -1.000000, 0.000000],
    #     yaw=-1.470796, pitch=-0.440000
    # )

    # # plant6
    # camera.set_camera_pose(
    #     position=[-0.129254, -2.094610, -1.542254],
    #     target=[0.254997, 0.944699, 0.838787],
    #     up=[0.000000, -1.000000, 0.000000],
    #     yaw=-1.410796, pitch=-0.900000
    # )

    # # plant 7
    # camera.set_camera_pose(
    #     position=[-0.793205, 0.564231, -1.678616],
    #     target=[1.452237, 0.218582, 1.466660],
    #     up=[0.000000, -1.000000, 0.000000],
    #     yaw=-0.950796, pitch=0.089204
    # )

    new_gaussians = load_ply_gaussian(filepath)

    if new_gaussians:
        new_gaussians = center_gaussians(new_gaussians)
        corrected_gaussians, root_idx, heat_solver = fix_plant_root_direction_legacy(
            new_gaussians, opacity_threshold=0.0, given_root_idx=given_root_idx)
        print('corrected_gaussians: ', len(corrected_gaussians.xyz))
        
        new_gaussians = corrected_gaussians
        sparse_indices = fpsample.bucket_fps_kdline_sampling(new_gaussians.xyz, 8192, h=7, start_idx=0)
        downsampled_gaussians = apply_indices_to_gaussian_data(new_gaussians, sparse_indices)
        sparse_heat_solver = pp3d.PointCloudHeatSolver(downsampled_gaussians.xyz, t_coef=1e+8)
        
        kdtree = NearestNeighbors(n_neighbors=1)
        kdtree.fit(downsampled_gaussians.xyz)  
        
        distances, indices = kdtree.kneighbors(new_gaussians.xyz)
        original_to_sparse_mapping = indices.flatten()  
        cached_root_distances = heat_solver.compute_distance(root_idx)
        heat_source_idx = np.where(cached_root_distances <= 0.1)[0]
        print(f"heat_source_idx: len {len(heat_source_idx)}")
        cached_root_distances = heat_solver.compute_distance_multisource(heat_source_idx)

        temperature_colors = None
        leaf_tip_spheres = []
        path_spheres = []
        segments.clear() 
        segments_expanded.clear()  

        current_bvh = gsplat_bvh.build_bvh(
            new_gaussians.xyz,
            new_gaussians.rot, 
            new_gaussians.scale,
            LEAF_SIZE,
            NUM_BIN
        )
        
        current_gaussians = new_gaussians
        original_plant_gaussians = GaussianData(
            xyz=current_gaussians.xyz.copy(),
            rot=current_gaussians.rot.copy(), 
            scale=current_gaussians.scale.copy(),
            opacity=current_gaussians.opacity.copy(),
            sh=current_gaussians.sh.copy(),
            nxnynz=current_gaussians.nxnynz.copy(),
            filter_3Ds=current_gaussians.filter_3Ds.copy()
        )
        # save_gaussian_data_as_ply(f"original_plant_gaussians.ply", original_plant_gaussians)
        ckdtree = cKDTree(original_plant_gaussians.xyz)
        
        current_gaussian_path = filepath
        seg_path = f'/home/cg/my_codes/leaf_to_forest/exp/segs/{current_gaussian_path.split("/")[-1].split(".")[0]}'
        mesh_path = f'/home/cg/my_codes/leaf_to_forest/exp/meshes/{current_gaussian_path.split("/")[-1].split(".")[0]}'
        deform_path = f'/home/cg/my_codes/leaf_to_forest/exp/deforms/{current_gaussian_path.split("/")[-1].split(".")[0]}'
        pack_path = f'/home/cg/my_codes/leaf_to_forest/exp/packs/{current_gaussian_path.split("/")[-1].split(".")[0]}'
        if not os.path.exists(seg_path):
            os.makedirs(seg_path, exist_ok=True)

        if not os.path.exists(mesh_path):
            os.makedirs(mesh_path, exist_ok=True)

        if not os.path.exists(deform_path):
            os.makedirs(deform_path, exist_ok=True)

        if not os.path.exists(pack_path):
            os.makedirs(pack_path, exist_ok=True)

        # GLOBAL FOR SEGMENTATION
        global path_info
        segment_masks, found_tips, found_bases, found_geodist_from_tip, found_clusters, path_info  = get_segment_mask(
                        original_plant_gaussians, 
                        sparse_indices,
                        original_to_sparse_mapping,
                        heat_solver,
                        ckdtree, 
                        heat_source_idx,
                        g_segmentation_method_tables[g_segmentation_method],
                        cached_root_distances=cached_root_distances,
                        debug_vis=debug_vis
                        )
        selection_mode = False
        clear_segment_hover()  
        leaf_tip_idx = None
        current_selection = []
        cached_geodesic_distances = None
        sphere_selection_points = []
        sphere_click_count = 0
        segments.clear()
        segments_expanded.clear()
        
        original_sh_backup = original_plant_gaussians.sh.copy()

        gaussian_renderer.update_gaussian_data(current_gaussians)
        gaussian_renderer.sort_and_update(camera)
        gaussian_picker_error = ""
        return True

    return False


def main():
    global camera, mesh_renderer, gaussian_renderer, impl, take_screenshot
    global g_scale_modifier, g_auto_sort
    
    window = init_glfw()
    
    imgui.create_context()
    impl = GlfwRenderer(window)
    
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_key_callback(window, key_callback)
    glfw.set_window_size_callback(window, window_size_callback)
    
    camera = Camera(window_height, window_width)  
    mesh_renderer = MeshRenderer()
    gaussian_renderer = GaussianRenderer(window_width, window_height)
    gaussian_renderer.update_camera_pose(camera)
    gaussian_renderer.update_camera_intrin(camera)
    
    gl.glEnable(gl.GL_DEPTH_TEST)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()

        if bg_is_white:
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)  
        else:
            gl.glClearColor(0.1, 0.1, 0.1, 1.0) 

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        view_matrix = camera.get_view_matrix()
        projection_matrix = camera.get_project_matrix()
        
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_TRUE)  
        gl.glDisable(gl.GL_BLEND)   

        
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_TRUE) 
        gl.glDisable(gl.GL_BLEND)  
        
        if len(sphere_selection_points) > 0:
            for sphere_point in sphere_selection_points:
                if sphere_point["mesh"] is not None:
                    mesh_renderer.setup_mesh(sphere_point["mesh"])
                    mesh_renderer.render(view_matrix, projection_matrix, camera.position)
        
        if show_temperature_field and leaf_tip_spheres:
            for tip_sphere in leaf_tip_spheres:
                if tip_sphere["mesh"] is not None:
                    mesh_renderer.setup_mesh(tip_sphere["mesh"])
                    mesh_renderer.render(view_matrix, projection_matrix, camera.position)
        
        if show_temperature_field and path_spheres:
            for path_sphere in path_spheres:
                if path_sphere["mesh"] is not None:
                    mesh_renderer.setup_mesh(path_sphere["mesh"])
                    mesh_renderer.render(view_matrix, projection_matrix, camera.position)
        
        if root_idx is not None and current_gaussians is not None and show_root_sphere:
            try:
                root_position = current_gaussians.xyz[root_idx]

                if current_gaussians is not None:
                    scene_scale = np.max(np.std(current_gaussians.xyz, axis=0))
                    sphere_radius = max(0.05, scene_scale * 0.05)  
                else:
                    sphere_radius = 0.05

                global _root_sphere_cache
                root_cache_key = (tuple(root_position), sphere_radius, root_idx)
                if _root_sphere_cache is None or _root_sphere_cache[0] != root_cache_key:
                    root_sphere_mesh = create_sphere_mesh(root_position, sphere_radius, [1.0, 0.87, 0.13])  
                    _root_sphere_cache = (root_cache_key, root_sphere_mesh)

                if _root_sphere_cache is not None:
                    mesh_renderer.setup_mesh(_root_sphere_cache[1])
                    mesh_renderer.render(view_matrix, projection_matrix, camera.position)
            except (IndexError, TypeError) as e:
                pass
        
        for template_mesh in template_meshes:
            if template_mesh.get("visible", True) and template_mesh.get("mesh_data") is not None:
                mesh_data = template_mesh["mesh_data"]
                mesh_renderer.setup_mesh(mesh_data)
                mesh_renderer.render(view_matrix, projection_matrix, camera.position)
        
        for edge_viz in edge_visualizations:
            if edge_viz.get("visible", True) and edge_viz.get("mesh_data") is not None:
                mesh_data = edge_viz["mesh_data"]
                mesh_renderer.setup_mesh(mesh_data)
                mesh_renderer.render(view_matrix, projection_matrix, camera.position)
        
        combined_gaussian_data = create_combined_gaussian_data()
        
        if combined_gaussian_data is not None:
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthMask(gl.GL_FALSE)  
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            
            if camera.is_pose_dirty:
                gaussian_renderer.update_camera_pose(camera)
                camera.is_pose_dirty = False
                
            if camera.is_intrin_dirty:
                gaussian_renderer.update_camera_intrin(camera)
                camera.is_intrin_dirty = False
            
            gaussian_renderer.update_gaussian_data(combined_gaussian_data)
            gaussian_renderer.set_scale_modifier(g_scale_modifier)
            gaussian_renderer.set_render_mod(g_render_mode - 4)
            
            if g_auto_sort or camera.is_pose_dirty:
                gaussian_renderer.sort_and_update(camera)
                
            gaussian_renderer.draw()
        
        if take_screenshot:
            capture_screenshot(window_width, window_height)
            take_screenshot = False  

        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND)

        draw_ui()
        
        draw_segment_hover_tooltip()
        
        imgui.render()
        impl.render(imgui.get_draw_data())
        
        glfw.swap_buffers(window)
    
    impl.shutdown()
    glfw.terminate()

def draw_segments_menu():
    global segments, segments_expanded, show_segments_menu, template_meshes
    global show_temperature_field, temperature_colors, heat_solver, leaf_tip_spheres, path_spheres
    global root_idx, current_gaussians, original_plant_gaussians
    global selection_mode, brush_mode, leaf_tip_idx, current_selection, sphere_selection_points
    global g_segmentation_method
    global sparse_indices, sparse_heat_solver, original_to_sparse_mapping, cached_root_distances
    global current_gaussian_path, seg_path, mesh_path, deform_path, pack_path
    global g_template_segment_index
    
    # GLOBAL FOR SEGMENTATION
    global segment_masks, found_tips, found_bases, found_geodist_from_tip, found_clusters
    
    imgui.set_next_window_size(400, 500, imgui.FIRST_USE_EVER)
    imgui.set_next_window_position(50, 50, imgui.FIRST_USE_EVER)
    
    expanded, opened = imgui.begin("Segments Manager", True)
    if not opened:
        show_segments_menu = False
    
    if expanded:
        s_key_pressed = glfw.get_key(glfw.get_current_context(), glfw.KEY_S) == glfw.PRESS
        if s_key_pressed:
            imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.84, 0.0, 1.0) 
            if hovered_segment_id is not None and hovered_segment_id < len(segments):
                segment_name = segments[hovered_segment_id]["name"]
                imgui.text(f"+ Segment Hover: ACTIVE (Hovering: {segment_name})")
            else:
                imgui.text("+ Segment Hover: ACTIVE (Hold S key)")
            imgui.pop_style_color(1)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Hover over segments to highlight them while holding S key.")
        else:
            imgui.push_style_color(imgui.COLOR_TEXT, 0.6, 0.6, 0.6, 1.0)  
            imgui.text("- Segment Hover: INACTIVE (Hold S to activate)")
            imgui.pop_style_color(1)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Hold S key to enable segment hover highlighting")
        
        imgui.separator()
        
        if imgui.tree_node("Automatic Segmentation", imgui.TREE_NODE_DEFAULT_OPEN):
            
            # Visualize Temperature Field checkbox
            changed, show_temperature_field = imgui.checkbox("Visualize Temperature Field", show_temperature_field)
            if imgui.is_item_hovered():
                if root_idx is not None:
                    imgui.set_tooltip("Show heat diffusion, leaf tips and paths to root")
                else:
                    imgui.set_tooltip("Temperature field (requires root detection first)")
            
            if changed and show_temperature_field and root_idx is not None and current_gaussians is not None:
                try:
                    tree = cKDTree(current_gaussians.xyz)
                    
                    temp_field = get_temperature_field(heat_solver, [root_idx])
                    temperature_colors = vis_temperature_field(temp_field)
                    
                    start_time = time.time()
                    tip_indices = find_local_tips(current_gaussians, 
                                                  sparse_indices, 
                                                  original_to_sparse_mapping, 
                                                  temp_field, 
                                                  tree, 
                                                  k = len(sparse_indices) // 64)
                    end_time = time.time()
                    leaf_tip_spheres = []
                    path_spheres = []
                    path_spheres_indices = []
                    is_path_marks = np.zeros(len(current_gaussians.xyz), dtype=int)
                    
                    scene_scale = np.max(np.std(current_gaussians.xyz, axis=0))
                    tip_sphere_radius = max(0.06, scene_scale * 0.03)
                    path_sphere_radius = max(0.03, scene_scale * 0.015)  
                    pathes = []
                    start_time = time.time()
                    for i, tip_idx in enumerate(tip_indices):
                        tip_position = current_gaussians.xyz[tip_idx]
                        
                        tip_sphere = {
                            "position": tip_position,
                            "mesh": create_sphere_mesh(tip_position, tip_sphere_radius, [0.0, 1.0, 0.0]),  
                            "index": tip_idx
                        }
                        leaf_tip_spheres.append(tip_sphere)
                        
                        path = find_path_from_tip_to_root(current_gaussians, 
                                                            temp_field, 
                                                            tip_idx, 
                                                            root_idx, 
                                                            {
                                                                "method": "euclidean",
                                                                "tree": tree,
                                                                "dense_solver": heat_solver
                                                            },
                                                            is_path_marks, 
                                                            k = len(sparse_indices) // 32)
                        pathes.append(path)

                    end_time = time.time()
                    
                    for path in pathes:
                        for path_idx in path[1:-1]:  
                            if path_idx not in path_spheres_indices:
                                path_position = current_gaussians.xyz[path_idx]
                                if is_path_marks[path_idx] > 2:
                                    color = [0.0, 0.0, 1.0]
                                elif is_path_marks[path_idx] > 1:
                                    color = [1.0, 1.0, 0.0]
                                else:
                                    color = [1.0, 0.0, 0.0]
                                
                                path_sphere = {
                                    "position": path_position,
                                    "mesh": create_sphere_mesh(path_position, path_sphere_radius, color),
                                    "index": path_idx
                                }
                                path_spheres.append(path_sphere)
                                path_spheres_indices.append(path_idx)
                except Exception as e:
                    show_temperature_field = False
                    temperature_colors = None
                    leaf_tip_spheres = []
                    path_spheres = []
            elif changed and not show_temperature_field:
                temperature_colors = None
                leaf_tip_spheres = []
                path_spheres = []
            
            changed, g_segmentation_method = imgui.combo("Segmentation Method", g_segmentation_method, g_segmentation_method_tables)
            
            if imgui.button("Auto Segmentation"):
                if root_idx is not None and original_plant_gaussians is not None:
                    added_count = 0
                    all_segmented_indices = set()
                    for i, mask, apex_idx, base_idx in zip(range(len(segment_masks)), segment_masks, found_tips, found_bases):
                        indices = mask
                        print(f"Segment {i}: {len(indices)} indices, type: {type(indices)}, first few: {indices[:5] if len(indices) > 0 else 'empty'}")
                        all_segmented_indices.update(indices)
                        if len(apex_idx) > 1:
                            apex_idx = apex_idx[0]
                        
                        if len(indices) > 0:
                            if i < len(colors):
                                color = colors[i]
                            else:
                                color = generate_non_gold_color()
                            
                            labels = calculate_segment_labels(original_plant_gaussians, 
                                                                indices, 
                                                                apex_idx, 
                                                                base_idx)
                            
                            colored_segment_data, original_segment_data = create_segment_from_gaussians(
                                original_plant_gaussians,
                                indices, 
                                color,
                                opacity_factor=0.7  
                            )
                            
                            if not os.path.exists(f"{seg_path}/{g_segmentation_method_tables[g_segmentation_method]}"):
                                os.makedirs(f"{seg_path}/{g_segmentation_method_tables[g_segmentation_method]}", exist_ok=True)
                            save_gaussian_data_as_ply(f"{seg_path}/{g_segmentation_method_tables[g_segmentation_method]}/auto_segment_leaf_{i+1}.ply", original_segment_data)
                            save_gaussian_data_as_ply(f"{seg_path}/{g_segmentation_method_tables[g_segmentation_method]}/auto_segment_leaf_{i+1}_colored.ply", colored_segment_data)
                            np.save(f"{seg_path}/{g_segmentation_method_tables[g_segmentation_method]}/auto_segment_leaf_{i+1}.npy", indices)
                            
                            
                            auto_segment = {
                                "name": f"Auto_Leaf_{i+1}",
                                "indices": indices,
                                "color": color,
                                "visible": True,
                                "colored_data": colored_segment_data,  
                                "original_data": original_segment_data,  
                                "labels": labels,  
                                "apex_idx": apex_idx,  
                                "base_idx": base_idx,  
                                "apex_point": original_plant_gaussians.xyz[apex_idx],
                                "base_point": original_plant_gaussians.xyz[base_idx],
                                "is_auto": True,  
                                "source_data": current_gaussians  
                            }
                            segments.append(auto_segment)
                            added_count += 1
                    
                    print(f"Added {added_count} auto-segmented segments to segments list")
                    
                    total_points = len(original_plant_gaussians.xyz)
                    all_indices = np.arange(total_points)
                    remaining_indices = np.setdiff1d(all_indices, np.array(list(all_segmented_indices)))
                    
                    if len(remaining_indices) > 0:
                        print(f"Create Stem segment, containing {len(remaining_indices)} remaining points")
                        
                        colored_stem_segment_data, original_stem_segment_data = create_segment_from_gaussians(
                            original_plant_gaussians,
                            remaining_indices, 
                            [1.0, 1.0, 1.0],  
                            opacity_factor=0.7
                        )
                        
                        stem_segment = {
                            "name": "Stem",
                            "indices": remaining_indices,
                            "color": [1.0, 1.0, 1.0],
                            "visible": True,
                            "colored_data": colored_stem_segment_data,
                            "original_data": original_stem_segment_data,
                            "is_auto": True,
                            "source_data": current_gaussians
                        }
                        
                        if not os.path.exists(f"{seg_path}/{g_segmentation_method_tables[g_segmentation_method]}"):
                            os.makedirs(f"{seg_path}/{g_segmentation_method_tables[g_segmentation_method]}", exist_ok=True)
                        save_gaussian_data_as_ply(f"{seg_path}/{g_segmentation_method_tables[g_segmentation_method]}/auto_segment_stem.ply", original_stem_segment_data)
                        save_gaussian_data_as_ply(f"{seg_path}/{g_segmentation_method_tables[g_segmentation_method]}/auto_segment_stem_colored.ply", colored_stem_segment_data)

                        np.save(f"{seg_path}/{g_segmentation_method_tables[g_segmentation_method]}/auto_segment_stem.npy", remaining_indices)

                        segments.append(stem_segment)
                        save_dict = {}
                        save_dict["gaussians"] = original_plant_gaussians
                        save_dict["stem_indices"] = remaining_indices
                        counter = len(remaining_indices)
                        for ixxx, segment in enumerate(segments):
                            if segment["name"] != "Stem":
                                indices = segment["indices"]
                                save_dict[f"leaf_{ixxx}_indices"] = indices
                                counter += len(indices)
                        np.savez_compressed(f"{seg_path}/{g_segmentation_method_tables[g_segmentation_method]}/plant_{g_segmentation_method_tables[g_segmentation_method]}_segment.npz", save_dict)
            
            if imgui.is_item_hovered():
                imgui.set_tooltip("Automatically segment plant into leaves/branches")
            
            imgui.same_line()
            if imgui.button("Apply Method"):
                if current_gaussian_path:
                    acc_time = 0
                    print(f"Applying method: {g_segmentation_method_tables[g_segmentation_method]}")
                    start_time = time.time()
                    load_gaussian_file(current_gaussian_path, given_root_idx=root_idx, debug_vis=False)
                    if root_idx is not None and original_plant_gaussians is not None:
                        added_count = 0
                        all_segmented_indices = set()
                        for i, mask, apex_idx, base_idx in zip(range(len(segment_masks)), segment_masks, found_tips, found_bases):
                            indices = mask
                            print(f"Segment {i}: {len(indices)} indices, type: {type(indices)}, first few: {indices[:5] if len(indices) > 0 else 'empty'}")
                            all_segmented_indices.update(indices)
                            if len(apex_idx) > 1:
                                apex_idx = apex_idx[0]
                            
                            if len(indices) > 0:
                                if i < len(colors):
                                    color = colors[i]
                                else:
                                    color = generate_non_gold_color()
                                
                                labels = calculate_segment_labels(original_plant_gaussians, 
                                                                    indices, 
                                                                    apex_idx, 
                                                                    base_idx)
                                
                                colored_segment_data, original_segment_data = create_segment_from_gaussians(
                                    original_plant_gaussians,
                                    indices, 
                                    color,
                                    opacity_factor=0.7  
                                )
                                if not os.path.exists(f"{seg_path}/{g_segmentation_method_tables[g_segmentation_method]}"):
                                    os.makedirs(f"{seg_path}/{g_segmentation_method_tables[g_segmentation_method]}", exist_ok=True)
                                save_gaussian_data_as_ply(f"{seg_path}/{g_segmentation_method_tables[g_segmentation_method]}/auto_segment_leaf_{i+1}.ply", original_segment_data)
                                np.save(f"{seg_path}/{g_segmentation_method_tables[g_segmentation_method]}/auto_segment_leaf_{i+1}.npy", indices)
                                
                                auto_segment = {
                                    "name": f"Auto_Leaf_{i+1}",
                                    "indices": indices,
                                    "color": color,
                                    "visible": True,
                                    "colored_data": colored_segment_data,  
                                    "original_data": original_segment_data,  
                                    "labels": labels,  
                                    "apex_idx": apex_idx,  
                                    "base_idx": base_idx,  
                                    "apex_point": original_plant_gaussians.xyz[apex_idx],
                                    "base_point": original_plant_gaussians.xyz[base_idx],
                                    "is_auto": True,  
                                    "source_data": current_gaussians  
                                }
                                segments.append(auto_segment)
                                added_count += 1
                        
                        print(f"Added {added_count} auto-segmented segments to segments list")
                        end_time = time.time()
                        acc_time = (end_time - start_time)
                        total_points = len(original_plant_gaussians.xyz)
                        all_indices = np.arange(total_points)
                        remaining_indices = np.setdiff1d(all_indices, np.array(list(all_segmented_indices)))
                        
                        if len(remaining_indices) > 0:
                            print(f"Create Stem segment, containing {len(remaining_indices)} remaining points")
                            
                            colored_stem_segment_data, original_stem_segment_data = create_segment_from_gaussians(
                                original_plant_gaussians,
                                remaining_indices, 
                                [1.0, 1.0, 1.0],  
                                opacity_factor=0.7
                            )
                            
                            stem_segment = {
                                "name": "Stem",
                                "indices": remaining_indices,
                                "color": [1.0, 1.0, 1.0],
                                "visible": True,
                                "colored_data": colored_stem_segment_data,
                                "original_data": original_stem_segment_data,
                                "is_auto": True,
                                "source_data": current_gaussians
                            }
                            segments.append(stem_segment)
                            if not os.path.exists(f"{seg_path}/{g_segmentation_method_tables[g_segmentation_method]}"):
                                os.makedirs(f"{seg_path}/{g_segmentation_method_tables[g_segmentation_method]}", exist_ok=True)
                            save_gaussian_data_as_ply(f"{seg_path}/{g_segmentation_method_tables[g_segmentation_method]}/auto_segment_stem.ply", original_stem_segment_data)
                            np.save(f"{seg_path}/{g_segmentation_method_tables[g_segmentation_method]}/auto_segment_stem.npy", remaining_indices)

                            save_dict = {}
                            save_dict["gaussians"] = original_plant_gaussians
                            save_dict["stem_indices"] = remaining_indices
                            save_dict["acc_time"] = acc_time
                            counter = len(remaining_indices)
                            for ixxx, segment in enumerate(segments):
                                if segment["name"] != "Stem":
                                    indices = segment["indices"]
                                    save_dict[f"leaf_{ixxx}_indices"] = indices
                                    counter += len(indices)
                            assert counter == len(original_plant_gaussians.xyz), f"counter: {counter} != {len(original_plant_gaussians.xyz)}"
                            np.savez_compressed(f"{seg_path}/{g_segmentation_method_tables[g_segmentation_method]}/plant_{g_segmentation_method_tables[g_segmentation_method]}_segment.npz", save_dict)

                    print(f"Method applied successfully!")

            
            if imgui.is_item_hovered():
                imgui.set_tooltip("Clear all segments and apply selected segmentation method")
            
            if imgui.button("Load Segmentation"):
                segments.clear()
                gt_method = 'gt'
                selected_method = g_segmentation_method_tables[g_segmentation_method]
                gt_npy_indices_files = os.listdir(f"{seg_path}/{gt_method}")
                if selected_method != gt_method:
                    selected_method_npy_indices_files = os.listdir(f"{seg_path}/{selected_method}")
                    selected_method_npy_indices_files = [file for file in selected_method_npy_indices_files if file.endswith('.npy')]
                    selected_method_npy_indices_files.sort()
                    selected_labels = []
                gt_npy_indices_files = [file for file in gt_npy_indices_files if file.endswith('.npy')]
                gt_npy_indices_files.sort()
                # and put the npy contains stem in the first
                # first find which index contains stem
                stem_index = None
                for xxxx, file in enumerate(gt_npy_indices_files):
                    if 'stem' in file:
                        stem_index = xxxx
                        break
                # put it to the first
                gt_npy_indices_files.insert(0, gt_npy_indices_files.pop(stem_index))

                load_colors  = np.array([
                    [0.30, 0.00, 0.00],  # 0 earth brown (fixed)
                    [1.000, 0.000, 0.000],
                    [0.000, 0.000, 1.000],
                    [0.000, 1.000, 1.000],
                    [1.000, 1.000, 0.000],
                    [1.000, 0.000, 1.000],
                    [0.450, 0.900, 0.000],
                    [0.125, 0.750, 0.550],
                    [0.900, 0.675, 0.000],
                    [0.500, 0.250, 1.000],
                    [1.000, 0.500, 0.750],
                    [0.900, 0.338, 0.000],
                    [0.000, 0.625, 1.000],
                    [0.475, 0.900, 0.400],
                    [0.787, 0.900, 0.400],
                    [0.574, 0.000, 1.000],
                    [0.000, 0.900, 0.562],
                    [1.000, 0.000, 0.375],
                    [0.900, 0.169, 0.000],
                    [0.000, 0.812, 1.000],
                    [0.688, 0.000, 1.000],
                    [0.900, 0.844, 0.000],
                    [0.400, 0.672, 0.070],
                    [0.000, 0.900, 0.694],
                    [1.000, 0.000, 0.562],
                    [0.900, 0.506, 0.000],
                    [0.000, 0.438, 1.000],
                    [1.000, 0.000, 0.938],
                    [0.619, 0.900, 0.100],
                    [0.312, 0.000, 1.000],
                ], dtype=np.float32)
                gt_labels = []
                for xxxx, file in enumerate(gt_npy_indices_files):
                    xxindices = np.load(f"{seg_path}/{gt_method}/{file}")
                    print(f"{file} picks color {load_colors[xxxx]}")
                    gt_labels.append(xxindices)
                    
                    if g_segmentation_method == 9:
                        colored_stem_segment_data, original_stem_segment_data = create_segment_from_gaussians(
                            original_plant_gaussians,
                            xxindices, 
                            load_colors[xxxx],  
                            opacity_factor=0.8
                        )
                        segment = {
                            "name": file.split('.')[0],
                            "indices": xxindices,
                            "is_auto": True,
                            "visible": True,
                            "color": load_colors[xxxx],
                            "colored_data": colored_stem_segment_data,
                            "original_data": original_stem_segment_data,
                            "source_data": original_plant_gaussians
                        }
                        segments.append(segment)
                
                if g_segmentation_method != 9:
                    for xxxx, file in enumerate(selected_method_npy_indices_files):
                        xxindices = np.load(f"{seg_path}/{selected_method}/{file}")
                        selected_labels.append(xxindices)
                    # Do the Hungarian Matching
                    # Build IoU matrix
                    from scipy.optimize import linear_sum_assignment

                    def build_iou_matrix(pred_sets, gt_sets):
                        P, G = len(pred_sets), len(gt_sets)
                        iou = np.zeros((P, G), dtype=np.float32)
                        pred_sets_s = [set(p.tolist()) for p in pred_sets]
                        gt_sets_s   = [set(g.tolist()) for g in gt_sets]
                        for i in range(P):
                            Pi = pred_sets_s[i]
                            if not Pi:
                                continue
                            for j in range(G):
                                Gj = gt_sets_s[j]
                                if not Gj:
                                    continue
                                inter = len(Pi & Gj)
                                if inter == 0:
                                    continue
                                union = len(Pi | Gj)
                                iou[i, j] = inter / union if union > 0 else 0.0
                        return iou

                    iou_mat = build_iou_matrix(selected_labels, gt_labels)
                    row_ind, col_ind = linear_sum_assignment(1.0 - iou_mat)
                    iou_thresh = 0.5
                    pred2gt = {int(r): int(c) for r, c in zip(row_ind, col_ind) if iou_mat[r, c] >= iou_thresh}

                    selected_aligned = [None] * len(gt_labels)
                    for p_idx, g_idx in pred2gt.items():
                        selected_aligned[g_idx] = selected_labels[p_idx]

                    unmatched_pred = [i for i in range(len(selected_labels)) if i not in pred2gt]

                    print(f"[Match] matched pairs: {len(pred2gt)} / {len(gt_labels)} "
                        f"(mean IoU={np.mean([iou_mat[p,g] for p,g in pred2gt.items()] or [0]):.3f})")
                    print(f"[Match] unmatched predicted clusters: {len(unmatched_pred)}")

                    for j, gt_file in enumerate(gt_npy_indices_files):
                        pred_idxs = selected_aligned[j]
                        if pred_idxs is None:
                            continue  
                        colored_seg, original_seg = create_segment_from_gaussians(
                            original_plant_gaussians,
                            pred_idxs,
                            load_colors[j],          
                            opacity_factor=0.8
                        )
                        segment = {
                            "name": f"{selected_method}_aligned_to_{gt_file.split('.')[0]}",
                            "indices": pred_idxs,
                            "is_auto": True,
                            "visible": True,
                            "color": load_colors[j],
                            "colored_data": colored_seg,
                            "original_data": original_seg,
                            "source_data": original_plant_gaussians
                        }
                        segments.append(segment)

                    gray = np.array([0.7, 0.7, 0.7], dtype=np.float32)
                    for p in unmatched_pred:
                        pred_idxs = selected_labels[p]
                        colored_seg, original_seg = create_segment_from_gaussians(
                            original_plant_gaussians,
                            pred_idxs,
                            gray,
                            opacity_factor=0.8
                        )
                        segment = {
                            "name": f"{selected_method}_unmatched_{p}",
                            "indices": pred_idxs,
                            "is_auto": True,
                            "visible": True,
                            "color": gray,
                            "colored_data": colored_seg,
                            "original_data": original_seg,
                            "source_data": original_plant_gaussians
                        }
                        segments.append(segment)
                    
            imgui.tree_pop()
        
        imgui.separator()
        
        # Manual Segmentation section
        if imgui.tree_node("Manual Segmentation", imgui.TREE_NODE_DEFAULT_OPEN):
            
            if imgui.tree_node("Drag Selection (CTRL+MOUSE_MID)"):
                imgui.text("Instructions:")
                imgui.bullet_text("Hold CTRL and drag with middle mouse button")
                imgui.bullet_text("Drag from leaf tip towards stem")
                imgui.bullet_text("Release to select path along the leaf")
                imgui.bullet_text("Click 'Create Segment' to finalize")
                
                global drag_selection_radius
                changed, drag_selection_radius = imgui.slider_float("Selection Radius", drag_selection_radius, 0.01, 0.5, "%.3f")
                if changed:
                    print(f"Drag selection radius updated to: {drag_selection_radius:.3f}")
                
                imgui.tree_pop()
            
            if imgui.tree_node("Click Selection (CTRL+MOUSE_LEFT)"):
                imgui.text("Instructions:")
                imgui.bullet_text("Hold CTRL and click with left mouse button")
                imgui.bullet_text("Click two points to define leaf endpoints")
                imgui.bullet_text("System calculates geodesic path between points")
                imgui.bullet_text("Click 'Create Segment from Selection' to finalize")
                
                imgui.text(f"Selected Points: {len(sphere_selection_points)}/2")
                if len(sphere_selection_points) > 0:
                    for i, point in enumerate(sphere_selection_points):
                        color_name = "Red" if i == 0 else "Green"
                        imgui.text(f"  Point {i+1} ({color_name}): Gaussian {point['gaussian_idx']}")
                    
                imgui.tree_pop()
            
            if imgui.tree_node("Brush Selection (SHIFT+MOUSE_MID)"):
                imgui.text("Instructions:")
                imgui.bullet_text("Hold SHIFT and drag with middle mouse button")
                imgui.bullet_text("Brush over areas to cumulatively select points")
                imgui.bullet_text("Points within geodesic radius are added to selection")
                imgui.bullet_text("Previously selected points remain selected")
                
                if brush_mode:
                    imgui.text_colored("<+> Brush Status: Active", 0.0, 1.0, 1.0)
                else:
                    imgui.text_colored("<+> Brush Status: Inactive", 1.0, 0.0, 0.0)
                
                brush_selected_count = len(current_selection) if brush_mode else 0
                imgui.text(f"Brush Selected Points: {brush_selected_count}")

                if brush_mode and len(current_selection) > 0:
                    if imgui.button("Clear Brush Selection"):
                        current_selection = []
                        brush_mode = False
                
                imgui.tree_pop()
            
            imgui.separator()
            imgui.text_colored("Selection Status:", 1.0, 0.5, 0.0)
            
            if selection_mode:
                imgui.text_colored("<+> Drag Selection: Active", 0.0, 1.0, 0.0)
                if leaf_tip_idx is not None:
                    imgui.text(f"  Leaf Tip: Gaussian {leaf_tip_idx}")
            else:
                imgui.text_colored("<+> Drag Selection: Inactive", 1.0, 0.0, 0.0)
            
            if len(sphere_selection_points) == 2:
                imgui.text_colored("<+> Click Selection: Active", 0.0, 1.0, 0.0)
                imgui.text(f"  Points: Red & Green spheres")
            elif len(sphere_selection_points) == 1:
                imgui.text_colored("<+> Click Selection: Partial", 1.0, 0.7, 0.0)
                imgui.text(f"  Points: 1/2 selected")
            else:
                imgui.text_colored("<+> Click Selection: Inactive", 1.0, 0.0, 0.0)
            
            if brush_mode and len(current_selection) > 0:
                imgui.text_colored("<+> Brush Selection: Active", 0.0, 1.0, 1.0)
                imgui.text(f"  Brush Points: {len(current_selection)}")
            else:
                imgui.text_colored("<+> Brush Selection: Inactive", 1.0, 0.0, 0.0)
            
            selected_count = len(current_selection)
            if selected_count > 0:
                imgui.text_colored(f"Selected Gaussians: {selected_count}", 0.0, 0.8, 1.0)
            else:
                imgui.text("Selected Gaussians: 0")
            
            if selected_count > 1:
                if imgui.button("Create Segment"):
                    apex_idx = current_selection[0]
                    base_idx = current_selection[-1]
                    segment_id = len(segments) + 1
                    
                    if len(sphere_selection_points) == 2:
                        segment_name = f"C_Leaf_{segment_id}"
                    else:
                        segment_name = f"D_Leaf_{segment_id}"
                    
                    manual_segment(current_selection, segment_name, apex_idx, base_idx)
                    if not os.path.exists(f"{seg_path}/gt/manual_segments"):
                        os.makedirs(f"{seg_path}/gt/manual_segments", exist_ok=True)
                    save_gaussian_data_as_ply(f"{seg_path}/gt/manual_segments/manual_segment_leaf_{segment_name}.ply", apply_indices_to_gaussian_data(original_plant_gaussians, current_selection))
                    np.save(f"{seg_path}/gt/manual_segments/manual_segment_leaf_{segment_name}.npy", current_selection)
                    clear_all_highlights()
                    clear_segment_hover()  
                    selection_mode = False
                    brush_mode = False
                    leaf_tip_idx = None
                    current_selection = []
                    clear_sphere_selection()
                    print(f"Created segment {segment_name} with {selected_count} gaussians")
            else:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.3, 0.3, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.3, 0.3, 0.3, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.3, 0.3, 0.3, 1.0)
                imgui.button("Create Segment##disabled")
                imgui.pop_style_color(3)
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Need to select more than 1 gaussian to create segment")
            
            imgui.same_line()
            
            has_selections = selected_count > 0 or len(sphere_selection_points) > 0 or selection_mode or brush_mode
            if has_selections:
                if imgui.button("Clear Selections"):
                    clear_all_highlights()
                    clear_segment_hover()  
                    selection_mode = False
                    brush_mode = False
                    leaf_tip_idx = None
                    current_selection = []
                    clear_sphere_selection()
            else:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.3, 0.3, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.3, 0.3, 0.3, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.3, 0.3, 0.3, 1.0)
                imgui.button("Clear Selections##disabled")
                imgui.pop_style_color(3)
                if imgui.is_item_hovered():
                    imgui.set_tooltip("No selections to clear")
            
            segment_count = len(segments)
            if segment_count >= 1:
                if imgui.button("Finish Segmentation"):
                    if original_plant_gaussians is not None:
                        total_points = len(original_plant_gaussians.xyz)
                        manually_segmented_indices = set()
                        
                        for segment in segments:
                            if not segment.get("is_auto", False):  
                                manually_segmented_indices.update(segment["indices"])
                        
                        remaining_indices = []
                        for i in range(total_points):
                            if i not in manually_segmented_indices:
                                remaining_indices.append(i)
                        
                        remaining_indices = np.array(remaining_indices)
                        
                        if len(remaining_indices) > 0:
                            stem_segment_data = create_segment_from_gaussians(
                                original_plant_gaussians,
                                remaining_indices, 
                                [1.0, 1.0, 1.0],  
                                opacity_factor=0.7
                            )
                            if not os.path.exists(f"{seg_path}/gt"):
                                os.makedirs(f"{seg_path}/gt", exist_ok=True)
                            save_gaussian_data_as_ply(f"{seg_path}/gt/manual_segment_stem.ply", apply_indices_to_gaussian_data(original_plant_gaussians, remaining_indices))
                            np.save(f"{seg_path}/gt/manual_segment_stem.npy", remaining_indices)

                            stem_segment = {
                                "name": "Stem",
                                "indices": remaining_indices,
                                "color": [1.0, 1.0, 1.0],
                                "visible": True,
                                "data": stem_segment_data,
                                "is_auto": False,  
                                "source_data": original_plant_gaussians
                            }
                            segments.append(stem_segment)
                            save_dict = {}
                            save_dict["gaussians"] = original_plant_gaussians
                            save_dict["stem_indices"] = remaining_indices
                            counter = len(remaining_indices)
                            for ixxx, segment in enumerate(segments):
                                if segment["name"] != "Stem":
                                    indices = segment["indices"]
                                    save_dict[f"leaf_{ixxx}_indices"] = indices
                                    counter += len(indices)
                            assert counter == len(original_plant_gaussians.xyz), f"counter: {counter} != {len(original_plant_gaussians.xyz)}"
                            np.savez_compressed(f"{seg_path}/gt/plant_gt_segment.npz", save_dict)
  
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Create white stem segment from all remaining points and finish manual segmentation")
            else:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.3, 0.3, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.3, 0.3, 0.3, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.3, 0.3, 0.3, 1.0)
                imgui.button("Finish Segmentation##disabled")
                imgui.pop_style_color(3)
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Need at least 1 segment to finish segmentation")
            
            imgui.tree_pop()
        
        imgui.separator()
        
        manual_count = len([s for s in segments if not s.get("is_auto", False)])
        auto_count = len([s for s in segments if s.get("is_auto", False)])
        
        imgui.text(f"Manual Segments: {manual_count}")
        if auto_count > 0:
            imgui.text(f"Auto Segments: {auto_count}")
        imgui.text(f"Total Segments: {len(segments)}")
        imgui.separator()
        
        if imgui.button("Hide All Segments"):
            for segment in segments:
                segment["visible"] = False
        imgui.same_line()
        if imgui.button("Show All Segments"):
            for segment in segments:
                segment["visible"] = True
        
        if imgui.button("Clear All Segments"):
            edge_count = len(edge_visualizations)
            edge_visualizations.clear()
            
            segments.clear()
            segments_expanded.clear()
            print(f"Cleared all segments and {edge_count} edge visualizations")
        imgui.separator()
        if imgui.button("T-Leaf ALL"):
            for segment in tqdm(segments):
                if segment["name"].lower() == "stem":
                    continue
                if segment.get("leaf") is not None:
                    continue
                # save_gaussian_data_as_ply(f"{deform_path}/{segment['name']}.ply", original_gaussian_data)
                _, new_points_, new_normals_ = mls_denoising(segment["original_data"].xyz, 0.12, expand=1.0)
     
                vertices, triangles, _ = generate_template_leaf(
                    new_points_,
                    new_normals_,
                )

                segment["leaf"] = {"vertices": vertices, "triangles": triangles}
        
        if imgui.button("EDGE ALL"):
            for segment in tqdm(segments):
                if segment["name"].lower() == "stem":
                    continue
                if segment.get("edge_indices") is not None:
                    continue
                edge_points_record = estimate_edge_from_segment(segment)
                edge_indices = edge_points_record["edge_points_indices"]
                segment["edge_indices"] = edge_indices
        segment_names = ["None"] 
        for segment in segments:
            if segment["name"].lower() != "stem":  
                segment_names.append(segment["name"])
        
        changed, g_template_segment_index = imgui.combo("Template Segment", g_template_segment_index, segment_names)
        
        if imgui.button(f"PCA ALL"):
            if g_template_segment_index == 0: 
                source_segments = [seg for seg in segments if seg["name"].lower() != "stem"]
            else:
                selected_segment_name = segment_names[g_template_segment_index]
                source_segments = [seg for seg in segments if seg["name"] == selected_segment_name]
            final_mean_cd = []
            final_mean_cd_org = []
            final_mean_time = []
            for source_segment in source_segments:
                template_mesh = make_template_leaf_from_segment(source_segment)
                cd_list = []
                cd_org_list = []
                times = []
                for segment in segments:
                    if segment['name'] == source_segment["name"]:
                        continue
                    if segment["name"].lower() == "stem":
                        continue
                    if segment.get("leaf") is None:
                        continue
                    start_time = time.time()
                    mesh, gaussians = apply_pca_transformation_to_segment(segment, template_mesh)
                    end_time = time.time()
                    times.append(end_time - start_time)
                    verts = mesh['mesh_data'].vertices
                    gt_verts = segment["leaf"]["vertices"]
                    gt_gaussians = segment["original_data"].xyz
                    cd = chamfer_distance(verts, gt_verts)
                    cd_org = chamfer_distance(gaussians, gt_gaussians)
                    cd_org_list.append(float(cd_org))
                    cd_list.append(float(cd))

                print(f"{source_segment['name']} -> ALL cd_list: {cd_list}")
                print(f"{source_segment['name']} -> ALL cd_org_list: {cd_org_list}")
                print(f"{source_segment['name']} -> ALL times: {times}")
                print(f"{source_segment['name']} -> ALL mean cd: {np.mean(cd_list)}")
                print(f"{source_segment['name']} -> ALL mean cd_org: {np.mean(cd_org_list)}")
                print(f"{source_segment['name']} -> ALL mean time: {np.mean(times)}")
                final_mean_cd.append(str(float(np.mean(cd_list))))
                final_mean_cd_org.append(str(float(np.mean(cd_org_list))))
                final_mean_time.append(str(float(np.mean(times))))  
            
            print(f" ALL (V): {'  '.join(final_mean_cd)}")
            print(f" ALL (G): {'  '.join(final_mean_cd_org)}")
            print(f" ALL (T): {'  '.join(final_mean_time)}")
            
        if imgui.button(f"FPS_MLS ALL"):
            if g_template_segment_index == 0:  
                source_segments = [seg for seg in segments if seg["name"].lower() != "stem"]
            else:
                selected_segment_name = segment_names[g_template_segment_index]
                source_segments = [seg for seg in segments if seg["name"] == selected_segment_name]
            
            final_mean_cd = []
            final_mean_cd_org = []
            final_mean_time = []
            for source_segment in source_segments:
                print(f"Processing template segment: {source_segment['name']}")
                template_mesh = make_template_leaf_from_segment(source_segment)

                cd_list = []
                cd_org_list = []
                times = []
                corr_pairs = []
                print("Segments are :", segments)
                # template_mesh_save_for_gpu = imgui.NONE
                for segment in segments:
                    if segment['name'] == source_segment["name"]:
                        continue
                    if segment["name"].lower() == "stem":
                        continue
                    if segment.get("leaf") is None:
                        continue
                    print(f"  Deforming to segment: {segment['name']}")
                    start_time = time.time()
                    mesh, gaussians, corr_pair = apply_mls_fps_transformation_to_segment(segment, template_mesh)

                    corr_pairs.append(corr_pair)
                    end_time = time.time()
                    times.append(end_time - start_time)
                    verts = mesh['mesh_data'].vertices
                    gt_verts = segment["leaf"]["vertices"]
                    gt_gaussians = segment["original_data"].xyz
                    cd = chamfer_distance(verts, gt_verts)
                    cd_org = chamfer_distance(gaussians, gt_gaussians)
                    cd_org_list.append(float(cd_org))
                    cd_list.append(float(cd))

                print(f"{source_segment['name']} -> ALL cd_list: {cd_list}")
                print(f"{source_segment['name']} -> ALL cd_org_list: {cd_org_list}")
                print(f"{source_segment['name']} -> ALL times: {times}")
                print(f"{source_segment['name']} -> ALL mean cd: {np.mean(cd_list)}")
                print(f"{source_segment['name']} -> ALL mean cd_org: {np.mean(cd_org_list)}")
                print(f"{source_segment['name']} -> ALL mean time: {np.mean(times)}")
                final_mean_cd.append(str(float(np.mean(cd_list))))
                final_mean_cd_org.append(str(float(np.mean(cd_org_list))))
                final_mean_time.append(str(float(np.mean(times))))  
            
            print(f" ALL (V): {'  '.join(final_mean_cd)}")
            print(f" ALL (G): {'  '.join(final_mean_cd_org)}")
            print(f" ALL (T): {'  '.join(final_mean_time)}")
            
        if imgui.button(f"FPS+OPTIM_MLS ALL"):
            if g_template_segment_index == 0:  
                source_segments = [seg for seg in segments if seg["name"].lower() != "stem"]
            else:
                selected_segment_name = segment_names[g_template_segment_index]
                source_segments = [seg for seg in segments if seg["name"] == selected_segment_name]
            
            final_mean_cd = []
            final_mean_cd_org = []
            final_mean_time = []
            for source_segment in source_segments:
                template_mesh = make_template_leaf_from_segment(source_segment)
                cd_list = []
                cd_org_list = []
                times = []
                corr_pairs = []
                for segment in segments:
                    if segment['name'] == source_segment["name"]:
                        continue
                    if segment["name"].lower() == "stem":
                        continue
                    if segment.get("leaf") is None:
                        continue
                    start_time = time.time()
                    mesh, gaussians, corr_pair = apply_mls_fps_optim_transformation_to_segment(segment, template_mesh)
                    corr_pairs.append(corr_pair)
                    end_time = time.time()
                    times.append(end_time - start_time)
                    verts = mesh['mesh_data'].vertices
                    gt_verts = segment["leaf"]["vertices"]
                    gt_gaussians = segment["original_data"].xyz
                    cd = chamfer_distance(verts, gt_verts)
                    cd_org = chamfer_distance(gaussians, gt_gaussians)
                    cd_org_list.append(float(cd_org))
                    cd_list.append(float(cd))
                    save_path = f"{deform_path}/{source_segment['name']}/to_{segment['name']}_fps_mls_optim"
                    if not os.path.exists(save_path):
                        os.makedirs(save_path, exist_ok=True)
                    write_mesh_to_disk(save_path, mesh['mesh_data'])

                print(f"Packing for GPU for {source_segment['name']}")
                gpu_footprint = pack_for_gpu(template_mesh, corr_pairs)
                
                if not os.path.exists(f"{pack_path}/fps_mls_optim/"):
                    os.makedirs(f"{pack_path}/fps_mls_optim/", exist_ok=True)
                np.savez_compressed(f"{pack_path}/fps_mls_optim/{source_segment['name']}.npz", **gpu_footprint)

                gpu_footprint["path_info"] = path_info
                gpu_footprint["tip_point"] = source_segment.get("apex_point", None)
                gpu_footprint["base_point"] = source_segment.get("base_point", None)
                gpu_footprint["original_gaussians"] = {
                    "xyz": source_segment["original_data"].xyz,
                    "rot": source_segment["original_data"].rot,
                    "scale": source_segment["original_data"].scale,
                    "opacity": source_segment["original_data"].opacity,
                    "sh": source_segment["original_data"].sh,
                }
                if not os.path.exists(f"{pack_path}/redundant_optim/"):
                    os.makedirs(f"{pack_path}/redundant_optim/", exist_ok=True)
                np.savez_compressed(f"{pack_path}/redundant_optim/{source_segment['name']}_fps_mls.npz", **gpu_footprint)
                print(f"{source_segment['name']} -> ALL cd_list: {cd_list}")
                print(f"{source_segment['name']} -> ALL cd_org_list: {cd_org_list}")
                print(f"{source_segment['name']} -> ALL times: {times}")
                print(f"{source_segment['name']} -> ALL mean cd: {np.mean(cd_list)}")
                print(f"{source_segment['name']} -> ALL mean cd_org: {np.mean(cd_org_list)}")
                print(f"{source_segment['name']} -> ALL mean time: {np.mean(times)}")
                final_mean_cd.append(str(float(np.mean(cd_list))))
                final_mean_cd_org.append(str(float(np.mean(cd_org_list))))
                final_mean_time.append(str(float(np.mean(times))))  
            
            print(f" ALL (V): {'  '.join(final_mean_cd)}")
            print(f" ALL (G): {'  '.join(final_mean_cd_org)}")
            print(f" ALL (T): {'  '.join(final_mean_time)}")
            
        if imgui.button(f"MVC_MLS ALL"):
            if g_template_segment_index == 0:  
                source_segments = [seg for seg in segments if seg["name"].lower() != "stem"]
            else:
                selected_segment_name = segment_names[g_template_segment_index]
                source_segments = [seg for seg in segments if seg["name"] == selected_segment_name]
            final_mean_cd = []
            final_mean_cd_org = []
            final_mean_time = []
            for source_segment in source_segments:
                if source_segment["name"].lower() == "stem":
                    continue
                template_mesh = make_template_leaf_from_segment(source_segment)
                ss_time = time.time()
                estimate_mvc_from_segment(source_segment, grid_density=mvc_grid_density, boundary_margin=mvc_boundary_margin, max_fps_points=mvc_max_fps_points)
                ss_time = time.time() - ss_time
                cd_list = []
                cd_org_list = []
                times = []
                corr_pairs = []
                source_mvc = source_segment.get("mvc", None)
                source_mvc_weights = source_mvc.get("weights", None)
                for segment in segments:
                    if segment['name'] == source_segment["name"]:
                        continue
                        continue
                    if segment.get("leaf") is None:
                        continue
                    start_time = time.time()
                    estimate_mvc_from_segment(segment, source_weights=source_mvc_weights)
                    mesh, gaussians, corr_pair = apply_mls_mvc_transformation_to_segment(segment, template_mesh)
                    corr_pairs.append(corr_pair)
                    end_time = time.time()
                    times.append(end_time - start_time)
                    verts = mesh['mesh_data'].vertices
                    gt_verts = segment["leaf"]["vertices"]
                    gt_gaussians = segment["original_data"].xyz
                    cd = chamfer_distance(verts, gt_verts)
                    cd_org = chamfer_distance(gaussians, gt_gaussians)
                    cd_org_list.append(float(cd_org))
                    cd_list.append(float(cd))

                print(f"Packing for GPU for {source_segment['name']}")
                gpu_footprint = pack_for_gpu(template_mesh, corr_pairs)
                gpu_footprint['path_info'] = path_info
                np.savez_compressed(f"deform_pack_{source_segment['name']}.npz", **gpu_footprint)

                print(f"{source_segment['name']} -> ALL cd_list: {cd_list}")
                print(f"{source_segment['name']} -> ALL cd_org_list: {cd_org_list}")
                print(f"{source_segment['name']} -> ALL times: {times}")
                print(f"{source_segment['name']} -> ALL mean cd: {np.mean(cd_list)}")
                print(f"{source_segment['name']} -> ALL mean cd_org: {np.mean(cd_org_list)}")
                print(f"{source_segment['name']} -> ALL mean time: {np.mean(times)}")
                final_mean_cd.append(str(float(np.mean(cd_list))))
                final_mean_cd_org.append(str(float(np.mean(cd_org_list))))
                final_mean_time.append(str(float(np.mean(times)) + float(ss_time)))  

            print(f" ALL (V): {'  '.join(final_mean_cd)}")
            print(f" ALL (G): {'  '.join(final_mean_cd_org)}")
            print(f" ALL (T): {'  '.join(final_mean_time)}")

        if imgui.button(f"MVC+OPTIM_MLS ALL"):
            if g_template_segment_index == 0: 
                source_segments = [seg for seg in segments if seg["name"].lower() != "stem"]
            else:
                selected_segment_name = segment_names[g_template_segment_index]
                source_segments = [seg for seg in segments if seg["name"] == selected_segment_name]
            final_mean_cd = []
            final_mean_cd_org = []
            final_mean_time = []
            for source_segment in source_segments:
                if source_segment["name"].lower() == "stem":
                    continue
                template_mesh = make_template_leaf_from_segment(source_segment)
                ss_time = time.time()
                estimate_mvc_from_segment(source_segment, grid_density=mvc_grid_density, boundary_margin=mvc_boundary_margin, max_fps_points=mvc_max_fps_points)
                ss_time = time.time() - ss_time
                cd_list = []
                cd_org_list = []
                times = []
                corr_pairs = []
                source_mvc = source_segment.get("mvc", None)
                source_mvc_weights = source_mvc.get("weights", None)
                for segment in segments:
                    if segment['name'] == source_segment["name"]:
                        continue
                        continue
                    if segment.get("leaf") is None:
                        continue
                    start_time = time.time()
                    estimate_mvc_from_segment(segment, source_weights=source_mvc_weights)
                    mesh, gaussians, corr_pair = apply_mls_mvc_optim_transformation_to_segment(segment, template_mesh)
                    corr_pairs.append(corr_pair)
                    end_time = time.time()
                    times.append(end_time - start_time)
                    verts = mesh['mesh_data'].vertices
                    gt_verts = segment["leaf"]["vertices"]
                    gt_gaussians = segment["original_data"].xyz
                    cd = chamfer_distance(verts, gt_verts)
                    cd_org = chamfer_distance(gaussians, gt_gaussians)
                    cd_org_list.append(float(cd_org))
                    cd_list.append(float(cd))

                print(f"Packing for GPU for {source_segment['name']}")
                gpu_footprint = pack_for_gpu(template_mesh, corr_pairs)
                gpu_footprint['path_info'] = path_info
                np.savez_compressed(f"deform_pack_{source_segment['name']}.npz", **gpu_footprint)

                print(f"{source_segment['name']} -> ALL cd_list: {cd_list}")
                print(f"{source_segment['name']} -> ALL cd_org_list: {cd_org_list}")
                print(f"{source_segment['name']} -> ALL times: {times}")
                print(f"{source_segment['name']} -> ALL mean cd: {np.mean(cd_list)}")
                print(f"{source_segment['name']} -> ALL mean cd_org: {np.mean(cd_org_list)}")
                print(f"{source_segment['name']} -> ALL mean time: {np.mean(times)}")
                final_mean_cd.append(str(float(np.mean(cd_list))))
                final_mean_cd_org.append(str(float(np.mean(cd_org_list))))
                final_mean_time.append(str(float(np.mean(times)) + float(ss_time)))  
            print(f" ALL (V): {'  '.join(final_mean_cd)}")
            print(f" ALL (G): {'  '.join(final_mean_cd_org)}")
            print(f" ALL (T): {'  '.join(final_mean_time)}")

        imgui.separator()
        
        if len(segments) == 0:
            imgui.text("No segments created yet.")
            imgui.text("Use the segmentation above to create segments")
            imgui.text("Gaussian selections.")
        else:
            segments_to_delete = []
            
            for i, segment in enumerate(segments):
                segment_name = segment["name"]
                
                if segment_name not in segments_expanded:
                    segments_expanded[segment_name] = False
                
                imgui.push_id(f"segment_{i}")
                
                changed, segment["visible"] = imgui.checkbox("##visible", segment["visible"])
                imgui.same_line()
                
                point_count = len(segment["indices"])
                
                if segments_expanded.get(segment_name, False):
                    imgui.set_next_item_open(True)
                    segments_expanded[segment_name] = False  
                
                segment_color = segment.get("color", [1.0, 1.0, 1.0]) 
                imgui.push_style_color(imgui.COLOR_TEXT, segment_color[0], segment_color[1], segment_color[2], 1.0)
                
                prefix = "[Auto] " if segment.get("is_auto", False) else ""
                tree_expanded = imgui.tree_node(f"{prefix}{segment_name} ({point_count} points)")
                
                imgui.pop_style_color(1)
                
                if tree_expanded:
                    imgui.indent()
                    
                    imgui.text(f"Points: {point_count}")
                    
                    imgui.text("Color:")
                    changed, new_color = imgui.color_edit3(
                        f"##color_picker_{i}", 
                        *segment["color"],
                        imgui.COLOR_EDIT_PICKER_HUE_WHEEL | imgui.COLOR_EDIT_DISPLAY_RGB
                    )
                    if changed:
                        segment["color"] = list(new_color)
                        if segment.get("original_data") is not None:
                            colored_segment_data, _ = create_segment_from_gaussians(
                                original_plant_gaussians,
                                segment["indices"], 
                                new_color,
                                opacity_factor=0.7
                            )
                            segment["colored_data"] = colored_segment_data
                    
                    imgui.text("Name:")
                    changed, new_name = imgui.input_text(f"##name_{i}", segment["name"], 128)
                    if changed and new_name.strip():
                        if segment_name in segments_expanded:
                            segments_expanded[new_name] = segments_expanded.pop(segment_name)
                        segment["name"] = new_name
                    
                    if segment["name"].lower() != "stem":
                        if imgui.button(f"Make Template Leaf##template_{i}"):
                            make_template_leaf_from_segment(segment, dense=False)
                        if imgui.is_item_hovered():
                            imgui.set_tooltip("Convert this leaf segment into a template mesh")
                        imgui.same_line()
                        
                        if imgui.button(f"Estimate Edge##edge_{i}"):
                            edge_points_record = estimate_edge_from_segment(segment, vis=True)
                            edge_indices = edge_points_record["edge_points_indices"]
                            segment["edge_indices"] = edge_indices
                        if imgui.is_item_hovered():
                            imgui.set_tooltip("Estimate and visualize edge points for this leaf segment")
                        
                    if imgui.button(f"Delete##del_{i}"):
                        segments_to_delete.append(i)
                        
                    
                    imgui.unindent()
                    imgui.tree_pop()
                
                imgui.pop_id()
                imgui.separator()
            
            for i in reversed(segments_to_delete):
                deleted_segment = segments.pop(i)
                if deleted_segment["name"] in segments_expanded:
                    del segments_expanded[deleted_segment["name"]]
                
                segment_name = deleted_segment["name"]
                edge_visualizations_to_remove = []
                for j, edge_viz in enumerate(edge_visualizations):
                    if (edge_viz.get("name", "").startswith(f"{segment_name}_edge_") or 
                        (edge_viz.get("type") == "edge_points_data" and edge_viz.get("segment_name") == segment_name)):
                        edge_visualizations_to_remove.append(j)
                
                for j in reversed(edge_visualizations_to_remove):
                    removed_edge = edge_visualizations.pop(j)
                
                print(f"Deleted segment: {segment_name}")
                if edge_visualizations_to_remove:
                    print(f"  Also deleted {len(edge_visualizations_to_remove)} edge visualizations")
    
    imgui.end()

def draw_mesh_manager():
    global template_meshes, edge_visualizations, show_mesh_manager
    
    imgui.set_next_window_size(400, 600, imgui.FIRST_USE_EVER)
    imgui.set_next_window_position(450, 50, imgui.FIRST_USE_EVER)
    
    expanded, opened = imgui.begin("Mesh Manager", True)
    
    if not opened:
        show_mesh_manager = False
    
    if expanded:
        imgui.text("Template Meshes:")
        imgui.separator()
        
        if len(template_meshes) == 0:
            imgui.text("No template meshes created yet.")
            imgui.text("")
            imgui.text("To create a template mesh:")
            imgui.text("1. Select a leaf segment in Segments window")
            imgui.text("2. Click 'Make Template Leaf' button")
        else:
            meshes_to_delete = []
            
            for i, template_mesh in enumerate(template_meshes):
                imgui.push_id(f"template_{i}")
                
                changed, template_mesh['visible'] = imgui.checkbox("##visible", template_mesh.get('visible', True))
                imgui.same_line()
                
                mesh_name = template_mesh['name']
                
                if imgui.tree_node(mesh_name, imgui.TREE_NODE_DEFAULT_OPEN):
                    imgui.indent()
                    
                    source_segment = template_mesh.get('source_segment', 'N/A')
                    point_count = template_mesh.get('point_count', 0)
                    imgui.text(f"Source: {source_segment}")
                    imgui.text(f"Points: {point_count}")
                    
                    if template_mesh.get("transformation_type") == "PCA":
                        imgui.text(f"Target: {template_mesh.get('target_segment', 'Unknown')}")
                        imgui.text("Type: PCA Transformed")
                    
                    if template_mesh.get('mesh_data'):
                        mesh_data = template_mesh['mesh_data']
                        
                        memory_components = {}
                        total_memory = 0
                        
                        if hasattr(mesh_data, 'vertices') and mesh_data.vertices is not None:
                            verts_mb = mesh_data.vertices.nbytes / 1024 / 1024
                            memory_components['vertices'] = verts_mb
                            total_memory += verts_mb
                        
                        if hasattr(mesh_data, 'faces') and mesh_data.faces is not None:
                            faces_mb = mesh_data.faces.nbytes / 1024 / 1024
                            memory_components['faces'] = faces_mb
                            total_memory += faces_mb
                        
                        if hasattr(mesh_data, 'uvs') and mesh_data.uvs is not None:
                            uvs_mb = mesh_data.uvs.nbytes / 1024 / 1024
                            memory_components['uvs'] = uvs_mb
                            total_memory += uvs_mb
                        
                        if hasattr(mesh_data, 'normals') and mesh_data.normals is not None:
                            normals_mb = mesh_data.normals.nbytes / 1024 / 1024
                            memory_components['normals'] = normals_mb
                            total_memory += normals_mb
                        
                        if hasattr(mesh_data, 'colors') and mesh_data.colors is not None:
                            colors_mb = mesh_data.colors.nbytes / 1024 / 1024
                            memory_components['colors'] = colors_mb
                            total_memory += colors_mb
                        
                        if hasattr(mesh_data, 'texture_data') and mesh_data.texture_data is not None:
                            texture_mb = mesh_data.texture_data.nbytes / 1024 / 1024
                            memory_components['texture'] = texture_mb
                            total_memory += texture_mb
                        
                        imgui.text(f"Total Memory: {total_memory:.2f} MB")
                        
                        if imgui.tree_node("Memory Breakdown"):
                            for component, mb_size in memory_components.items():
                                if total_memory > 0:
                                    percentage = (mb_size / total_memory) * 100
                                else:
                                    percentage = 0
                                
                                component_name = {
                                    'vertices': 'Vertices',
                                    'faces': 'Faces', 
                                    'uvs': 'UV Coords',
                                    'normals': 'Normals',
                                    'colors': 'Colors',
                                    'texture': 'Texture'
                                }.get(component, component.capitalize())
                                
                                imgui.text(f"{component_name}: {mb_size:.2f} MB ({percentage:.1f}%)")
                            imgui.tree_pop()
                    
                    imgui.separator()
                    
                    if imgui.button(f"Export Mesh##export_{i}"):
                        export_template_mesh(template_mesh)
                    imgui.same_line()
                    
                    if imgui.button(f"Delete##del_{i}"):
                        meshes_to_delete.append(i)
                    
                    imgui.unindent()
                    imgui.tree_pop()
                
                imgui.pop_id()
            
            for i in reversed(meshes_to_delete):
                deleted_mesh = template_meshes[i]
                template_name = deleted_mesh['name']
                
                edge_objects_to_remove = []
                for j, mesh in enumerate(template_meshes):
                    if mesh.get("type") == "edge_visualization" and mesh.get("parent_template") == template_name:
                        edge_objects_to_remove.append(j)
                
                for j in reversed(edge_objects_to_remove):
                    template_meshes.pop(j)
                    
                original_index = template_meshes.index(deleted_mesh)
                template_meshes.pop(original_index)
                
                mesh_data = deleted_mesh.get("mesh_data")
                if mesh_data is not None:
                    memory_freed = 0
                    if hasattr(mesh_data, 'texture_data') and mesh_data.texture_data is not None:
                        memory_freed += mesh_data.texture_data.nbytes / 1024 / 1024  # MB
                        mesh_data.texture_data = None
                    if hasattr(mesh_data, 'vertices') and mesh_data.vertices is not None:
                        memory_freed += mesh_data.vertices.nbytes / 1024 / 1024
                        mesh_data.vertices = None
                    if hasattr(mesh_data, 'uvs') and mesh_data.uvs is not None:
                        memory_freed += mesh_data.uvs.nbytes / 1024 / 1024
                        mesh_data.uvs = None
                
                if hasattr(deleted_mesh, '_mesh_renderer'):
                    deleted_mesh['_mesh_renderer'] = None
                
            if meshes_to_delete:
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
        
        if len(template_meshes) > 0:
            total_memory = sum(
                (mesh_data.texture_data.nbytes + mesh_data.vertices.nbytes + mesh_data.uvs.nbytes) / 1024 / 1024
                for template_mesh in template_meshes
                for mesh_data in [template_mesh.get('mesh_data')]
                if mesh_data and hasattr(mesh_data, 'texture_data') and mesh_data.texture_data is not None
            )
            imgui.text(f"Total Memory: ~{total_memory:.1f} MB")
            
            if imgui.button("Hide All Templates"):
                for template_mesh in template_meshes:
                    template_mesh["visible"] = False
            imgui.same_line()
            if imgui.button("Show All Templates"):
                for template_mesh in template_meshes:
                    template_mesh["visible"] = True
            
            if imgui.button("Clear All Templates"):
                for template_mesh in template_meshes:
                    mesh_data = template_mesh.get("mesh_data")
                    if mesh_data is not None:
                        if hasattr(mesh_data, 'texture_data'):
                            mesh_data.texture_data = None
                        if hasattr(mesh_data, 'vertices'):
                            mesh_data.vertices = None
                        if hasattr(mesh_data, 'uvs'):
                            mesh_data.uvs = None
                    if hasattr(template_mesh, '_mesh_renderer'):
                        template_mesh['_mesh_renderer'] = None
                
                template_meshes.clear()
                
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
        
        imgui.separator()
        imgui.text("Edge Visualizations:")
        
        if len(edge_visualizations) == 0:
            imgui.text("No edge visualizations created yet.")
        else:
            edge_groups = {}
            for edge_viz in edge_visualizations:
                segment_name = edge_viz.get("name", "").split("_edge_")[0] if "_edge_" in edge_viz.get("name", "") else "Unknown"
                if segment_name not in edge_groups:
                    edge_groups[segment_name] = {"apex": [], "left": [], "base": [], "right": []}
                
                name_parts = edge_viz.get("name", "").split("_")
                if len(name_parts) >= 3:
                    edge_type = name_parts[-2]  
                    if edge_type in edge_groups[segment_name]:
                        edge_groups[segment_name][edge_type].append(edge_viz)
            
            for segment_name, type_groups in edge_groups.items():
                total_edges = sum(len(edges) for edges in type_groups.values())
                if total_edges == 0:
                    continue
                
                imgui.push_id(f"segment_{segment_name}")
                
                all_visible = all(edge['visible'] for edges in type_groups.values() for edge in edges)
                any_visible = any(edge['visible'] for edges in type_groups.values() for edge in edges)
                
                if all_visible:
                    master_state = True
                else:
                    master_state = False
                
                changed, new_master_state = imgui.checkbox("##show_all_master", master_state)
                if changed:
                    for edges in type_groups.values():
                        for edge in edges:
                            edge['visible'] = new_master_state
                
                imgui.same_line()
                    
                if imgui.tree_node(f"{segment_name} Edges ({total_edges})"):
                    
                    for edge_type, edges in type_groups.items():
                        if len(edges) == 0:
                            continue
                            
                        color_names = {
                            "apex": "Green",
                            "left": "Red", 
                            "base": "Yellow",
                            "right": "Blue"
                        }
                        color_name = color_names.get(edge_type, "Unknown")
                        
                        if imgui.tree_node(f"{edge_type.capitalize()} ({color_name}) - {len(edges)} points"):
                            for i, edge_viz in enumerate(edges):
                                imgui.push_id(f"edge_{edge_type}_{i}")
                                
                                changed, edge_viz['visible'] = imgui.checkbox("##edge_visible", edge_viz.get('visible', True))
                                imgui.same_line()
                                
                                name_parts = edge_viz.get("name", "").split("_")
                                point_index = name_parts[-1] if name_parts else str(i)
                                
                                imgui.text(f"Point {point_index} ({edge_type.capitalize()})")
                                
                                imgui.pop_id()
                            imgui.tree_pop()
                    imgui.tree_pop()
                
                imgui.pop_id()  

    imgui.end()

def draw_ui():
    global show_control_panel, show_simple_gaussian_picker, g_scale_modifier, g_auto_sort
    global edge_sampling_count, mls_num_corr
    global mvc_grid_density, mvc_boundary_margin, mvc_max_fps_points, show_edge_debug
    global simple_file_path, root_idx
    global show_temperature_field, temperature_colors, heat_solver, leaf_tip_spheres, path_spheres
    global current_gaussians, original_plant_gaussians, current_gaussian_path, gaussian_picker_error, simple_gaussian_path
    global camera, gaussian_renderer, g_render_mode, g_render_mode_tables
    global selection_mode, leaf_tip_idx, current_selection, segments
    global sphere_selection_points, sphere_click_count
    global brush_mode, drag_selection_radius
    
    if show_control_panel:
        if imgui.begin("Control Panel", True):
            imgui.text(f"FPS: {imgui.get_io().framerate:.1f}")
            imgui.separator()
            
            imgui.text("Camera:")
            imgui.text(f"Position: ({camera.position[0]:.2f}, {camera.position[1]:.2f}, {camera.position[2]:.2f})")
            imgui.text(f"Target Distance: {camera.target_dist:.2f}")
            
            changed, camera.fovy = imgui.slider_float("FOV", camera.fovy, 0.1, np.pi - 0.1, "%.3f")
            if changed:
                camera.is_intrin_dirty = True
                
            changed, camera.rot_sensitivity = imgui.slider_float("Rotation Speed", camera.rot_sensitivity, 0.001, 0.1, "%.3f")
            changed, camera.trans_sensitivity = imgui.slider_float("Translation Speed", camera.trans_sensitivity, 0.001, 0.05, "%.3f")
            
            imgui.separator()
            
            global bg_is_white
            changed, bg_is_white = imgui.checkbox("White Background", bg_is_white)

            global show_root_sphere
            changed, show_root_sphere = imgui.checkbox("Show Root Sphere", show_root_sphere)

            imgui.separator()

            imgui.text("Gaussian Rendering:")
            changed, g_scale_modifier = imgui.slider_float("Scale Modifier", g_scale_modifier, 0.1, 5.0, "%.2f")
            
            changed, g_render_mode = imgui.combo("Rendering Mode", g_render_mode, g_render_mode_tables)
            
            imgui.separator()
            
            changed, g_auto_sort = imgui.checkbox("Auto Sort Gaussians", g_auto_sort)
            imgui.same_line()
            if imgui.button("Manual Sort"):
                if current_gaussians:
                    gaussian_renderer.update_gaussian_data(current_gaussians)
                    gaussian_renderer.sort_and_update(camera)
            
            imgui.text("Sort Backend:")
            available_backends = get_available_backends()
            current_backend = get_sort_backend_name()
            
            for i, backend in enumerate(available_backends):
                if i > 0:
                    imgui.same_line()
                selected = (backend == current_backend)
                if imgui.radio_button(f"{backend}##backend_{i}", selected):
                    if backend == "CPU (NumPy)":
                        set_sort_backend("cpu")
                    elif backend == "PyTorch CUDA":
                        set_sort_backend("torch")
                        
            imgui.separator()
            
            # Template transformation parameters
            imgui.text("Template Parameters:")
            global edge_sampling_count, mls_num_corr, mvc_grid_density, mvc_boundary_margin, mvc_max_fps_points, show_edge_debug
            changed, edge_sampling_count = imgui.slider_int("Edge Sampling", edge_sampling_count, 5, 50)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Number of samples per edge path for template leaf generation")
            
            changed, mls_num_corr = imgui.slider_int("MLS Correspondences", mls_num_corr, 16, 1024)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Number of correspondence points for MLS transformation")
            
            imgui.text("MVC Estimation:")
            changed, mvc_grid_density = imgui.slider_int("Grid Density", mvc_grid_density, 10, 40)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Density of sampling grid for MVC coordinate estimation")
            
            changed, mvc_boundary_margin = imgui.slider_float("Boundary Margin", mvc_boundary_margin, 0.001, 0.5, "%.3f")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Margin around leaf boundary for MVC grid generation")
                
            changed, mvc_max_fps_points = imgui.slider_int("Max FPS Points", mvc_max_fps_points, 4, 128)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Maximum number of points for Farthest Point Sampling")
            
            imgui.separator()
            imgui.text("Scene Info:")
            if current_gaussians:
                imgui.text(f"[OK] Gaussians: {len(current_gaussians.xyz)} (filtered & aligned)")
                if current_gaussian_path:
                    imgui.text(f"PLY File: {os.path.basename(current_gaussian_path)}")
                if current_bvh:
                    imgui.text(f"[OK] BVH: Ready")
                if root_idx is not None:
                    imgui.text(f"[OK] Root detected: Gaussian {root_idx}")
                else:
                    imgui.text("[WARN] No root detected - using original data")
            else:
                imgui.text("[X] No Gaussians loaded")
            
            imgui.end()
    
    if show_simple_gaussian_picker:
        if not simple_gaussian_path:
            simple_gaussian_path = "/home/cg/my_codes/leaf_to_forest/data"
            
        imgui.set_next_window_size(600, 350, imgui.FIRST_USE_EVER)
        imgui.set_next_window_position(200, 200, imgui.FIRST_USE_EVER)
        
        expanded, opened = imgui.begin("PLY Gaussian Loader", True)
        if not opened:
            show_simple_gaussian_picker = False
        if expanded:
            imgui.text("Select PLY Gaussian file to load:")
            imgui.separator()
            
            imgui.text("File Path: (Click to edit)")
            if imgui.is_window_appearing():
                imgui.set_keyboard_focus_here()
            changed, simple_gaussian_path = imgui.input_text("##gaussianpath", simple_gaussian_path, 512)
            
            imgui.text("Quick Access:")
            if imgui.button("Debug Folder##gauss"):
                simple_gaussian_path = "/home/cg/my_codes/leaf_to_forest/debug/"
            imgui.same_line()
            if imgui.button("Home##gauss"):
                simple_gaussian_path = os.path.expanduser("~/")
            imgui.same_line()
            if imgui.button("Desktop##gauss"):
                simple_gaussian_path = os.path.expanduser("~/Desktop/")
            
            imgui.separator()
            
            if simple_gaussian_path and simple_gaussian_path.strip():
                path = simple_gaussian_path.strip()
                imgui.text(f"Current: {path}")
                
                if os.path.exists(path):
                    if os.path.isfile(path):
                        if path.lower().endswith('.ply'):
                            imgui.text_colored("[OK] Valid PLY file - Ready to load", 0.0, 1.0, 0.0, 1.0)
                        else:
                            imgui.text_colored("[WARN] Not a PLY file", 1.0, 0.5, 0.0, 1.0)
                    elif os.path.isdir(path):
                        imgui.text_colored(" Directory - Enter full file path", 0.0, 0.5, 1.0, 1.0)
                        try:
                            ply_files = [f for f in os.listdir(path) if f.lower().endswith('.ply')]
                            if ply_files:
                                imgui.text("PLY files in this directory:")
                                for i, filename in enumerate(ply_files[:10]):  
                                    if imgui.button(f"{filename}##ply_{i}"):
                                        simple_gaussian_path = os.path.join(path, filename)
                                if len(ply_files) > 10:
                                    imgui.text(f"... and {len(ply_files) - 10} more")
                        except:
                            pass
                else:
                    imgui.text_colored("[X] Path does not exist", 1.0, 0.0, 0.0, 1.0)
            else:
                imgui.text("Enter path to PLY file...")
            
            imgui.separator()
            
            if gaussian_picker_error:
                imgui.text_colored(f"Error: {gaussian_picker_error}", 1.0, 0.0, 0.0, 1.0)
            
            can_load = (simple_gaussian_path and simple_gaussian_path.strip() and 
                       os.path.isfile(simple_gaussian_path.strip()) and 
                       simple_gaussian_path.strip().lower().endswith('.ply'))
            
            if imgui.button("Load PLY"):
                if can_load:
                    if load_gaussian_file(simple_gaussian_path.strip()):
                        show_simple_gaussian_picker = False
            
            imgui.same_line()
            if imgui.button("Cancel##gauss"):
                show_simple_gaussian_picker = False
            
            imgui.separator()
            imgui.text("Tips:")
            imgui.bullet_text("Use quick access buttons for common folders")
            imgui.bullet_text("Click on filenames in directory listings")
            imgui.bullet_text("PLY files should contain Gaussian splatting data")
        
        
        imgui.end()
    
    if show_segments_menu:
        draw_segments_menu()
    
    if show_mesh_manager:
        draw_mesh_manager()

if __name__ == "__main__":
    main()
