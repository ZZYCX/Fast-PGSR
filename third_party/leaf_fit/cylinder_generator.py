"""
Cylinder Generation Module for Plant Stem Visualization

This module handles the generation of tapered cylinders for visualizing plant stem structures
based on path analysis and usage statistics.
"""

import numpy as np
import open3d as o3d
from collections import deque


def simple_noise_3d(x, y, z, scale=1.0):
    """Simple 3D noise function using sine waves for natural variation"""
    return (np.sin(x * scale * 4.7) * np.cos(y * scale * 3.3) * np.sin(z * scale * 5.1) +
            np.cos(x * scale * 2.1) * np.sin(y * scale * 6.7) * np.cos(z * scale * 3.9)) * 0.5


def enforce_monotone_radii(point_graph,
                           point_radii,
                           true_root,
                           transition_factor=0.90,
                           min_radius=0.002,
                           max_radius=0.020,
                           max_iters=4):
    transition_factor = float(transition_factor)
    for _ in range(max_iters):
        changed = False
        q = deque([true_root])
        seen = {true_root}
        while q:
            u = q.popleft()
            ru = point_radii[u]
            for v, _usage in point_graph.get(u, []):
                target = max(min_radius, min(ru * transition_factor, max_radius))
                if point_radii[v] > target + 1e-9:
                    point_radii[v] = target
                    changed = True
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        if not changed:
            break
    return point_radii


def create_tapered_cylinder(coord1, coord2, radius1, radius2,
                            color1=None, color2=None,
                            resolution=12, noise_strength=0.08,
                            debug_colors=False,
                            height_segments=None,             
                            target_segment_length=0.01):      
    """Create a tapered cylinder with different radii at each end (world-space noise, seam-safe)."""
    coord1 = np.asarray(coord1, dtype=float)
    coord2 = np.asarray(coord2, dtype=float)

    direction = coord2 - coord1
    height = float(np.linalg.norm(direction))

    if height < 1e-6:
        return None

    direction_normalized = direction / max(height, 1e-12)
    center = (coord1 + coord2) / 2

    if height_segments is None:
        hs = max(8, int(np.ceil(height / max(1e-6, target_segment_length))))
    else:
        hs = int(max(1, height_segments))

    angles = np.linspace(0, 2*np.pi, resolution, endpoint=False)
    vertices = []
    faces = []

    for h in range(hs + 1):  
        t = h / hs  # 0..1
        current_radius = radius1 * (1 - t) + radius2 * t
        current_z = -height/2 + t * height
        for angle in angles:
            x = current_radius * np.cos(angle)
            y = current_radius * np.sin(angle)
            z = current_z
            vertices.append([x, y, z])

    vertices.append([0, 0, -height/2])  # bottom center
    vertices.append([0, 0,  height/2])  # top center

    for h in range(hs):
        for i in range(resolution):
            ni = (i + 1) % resolution
            cur = h * resolution
            nxt = (h + 1) * resolution
            v1 = cur + i
            v2 = cur + ni
            v3 = nxt + i
            v4 = nxt + ni
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])

    bottom_center_idx = len(vertices) - 2
    for i in range(resolution):
        ni = (i + 1) % resolution
        faces.append([bottom_center_idx, ni, i])

    top_center_idx = len(vertices) - 1
    top_ring_start = hs * resolution
    for i in range(resolution):
        ni = (i + 1) % resolution
        faces.append([top_center_idx, top_ring_start + i, top_ring_start + ni])

    vertices = np.asarray(vertices, dtype=float)

    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    dot_ = float(np.clip(np.dot(z_axis, direction_normalized), -1.0, 1.0))
    if np.isclose(dot_, 1.0, atol=1e-8):
        rotation_matrix = np.eye(3, dtype=float)
    elif np.isclose(dot_, -1.0, atol=1e-8):
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([np.pi, 0.0, 0.0], dtype=float))
    else:
        axis = np.cross(z_axis, direction_normalized)
        axis_norm = np.linalg.norm(axis)
        axis = axis / max(axis_norm, 1e-12)
        angle = np.arccos(dot_)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    verts_world = (vertices[:, :3] @ rotation_matrix.T)
    verts_world += center

    total_ring_vertices = (hs + 1) * resolution
    axis_dir = direction_normalized
    axis_origin = coord1.copy()

    if noise_strength > 0:
        verts_noisy = verts_world.copy()
        for i in range(verts_noisy.shape[0]):
            xw, yw, zw = verts_noisy[i]

            if i < total_ring_vertices:
                ring_idx = i // resolution
                t = ring_idx / hs if hs > 0 else 0.0

                edge = 0.15
                fade_t0 = (t - 0.0) / edge
                fade_t1 = (1.0 - t) / edge
                fade = max(0.0, min(fade_t0, fade_t1, 1.0))

                radius_current = radius1 * (1 - t) + radius2 * t

                n1 = simple_noise_3d(xw, yw, zw, scale=10.0) * noise_strength
                n2 = simple_noise_3d(xw + 100.0, yw + 100.0, zw + 100.0, scale=10.0) * noise_strength
                n3 = simple_noise_3d(xw + 200.0, yw + 200.0, zw + 200.0, scale=10.0) * noise_strength

                vec = np.array([xw, yw, zw], dtype=float) - axis_origin
                axial = axis_dir * np.dot(vec, axis_dir)
                radial_vec = vec - axial
                radial_norm = np.linalg.norm(radial_vec)
                if radial_norm > 1e-12:
                    radial_dir = radial_vec / radial_norm
                    tangential_dir = np.cross(axis_dir, radial_dir)
                    tangential_norm = np.linalg.norm(tangential_dir)
                    if tangential_norm > 1e-12:
                        tangential_dir /= tangential_norm
                    else:
                        tangential_dir = np.zeros(3, dtype=float)

                    radial_amp = (n1 + n2) * 0.5 * (noise_strength * radius_current) * fade
                    tang_amp   = (n2 - n1) * 0.25 * (noise_strength * 0.5 * radius_current) * fade
                    axial_amp  = n3 * (noise_strength * 0.25) * fade

                    verts_noisy[i] += radial_dir * radial_amp
                    verts_noisy[i] += tangential_dir * tang_amp
                    verts_noisy[i] += axis_dir * axial_amp
            else:
                cscale = noise_strength * 0.05
                nx = simple_noise_3d(xw + 300.0, yw + 300.0, zw + 300.0, scale=10.0) * cscale
                ny = simple_noise_3d(xw + 400.0, yw + 400.0, zw + 400.0, scale=10.0) * cscale
                nz = simple_noise_3d(xw + 500.0, yw + 500.0, zw + 500.0, scale=10.0) * cscale
                verts_noisy[i] += np.array([nx, ny, nz], dtype=float)

        verts_world = verts_noisy

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_world)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))

    if color1 is not None and color2 is not None:
        vertex_colors = []
        for i in range(verts_world.shape[0]):
            xw, yw, zw = verts_world[i]
            color_noise_scale = 0.0
            cn_r = simple_noise_3d(xw + 300.0, yw + 300.0, zw + 300.0, scale=5.0) * color_noise_scale
            cn_g = simple_noise_3d(xw + 400.0, yw + 400.0, zw + 400.0, scale=5.0) * color_noise_scale
            cn_b = simple_noise_3d(xw + 500.0, yw + 500.0, zw + 500.0, scale=5.0) * color_noise_scale

            if i < total_ring_vertices:
                ring_idx = i // resolution
                t = ring_idx / hs if hs > 0 else 0.0
                base_color = np.asarray(color1) * (1 - t) + np.asarray(color2) * t
            else:
                base_color = np.asarray(color1) if i == total_ring_vertices else np.asarray(color2)

            noisy_color = np.clip(base_color + np.array([cn_r, cn_g, cn_b], dtype=float), 0.0, 1.0)
            if debug_colors and i < 3:
                print(f"      Vertex {i}: base={base_color}, noise={[cn_r, cn_g, cn_b]}, final={noisy_color}")
            vertex_colors.append(noisy_color)
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(vertex_colors, dtype=float))
    elif color1 is not None:
        base = np.asarray(color1, dtype=float)
        vertex_colors = []
        for i in range(verts_world.shape[0]):
            xw, yw, zw = verts_world[i]
            color_noise_scale = 0.10
            cn = np.array([
                simple_noise_3d(xw + 300.0, yw + 300.0, zw + 300.0, scale=5.0),
                simple_noise_3d(xw + 400.0, yw + 400.0, zw + 400.0, scale=5.0),
                simple_noise_3d(xw + 500.0, yw + 500.0, zw + 500.0, scale=5.0),
            ]) * color_noise_scale
            vertex_colors.append(np.clip(base + cn, 0.0, 1.0))
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(vertex_colors, dtype=float))

    mesh.compute_vertex_normals()
    return mesh


def generate_stem_cylinders(path_info,
                            radius_levels=[0.03, 0.015, 0.005],
                            transition_factor=0.95,
                            branch_connection_radius=0.003,
                            min_radius=0.01,
                            max_radius=0.04,
                            cylinder_resolution=16,
                            noise_strength=0.0,
                            thickness_scale=3,          
                            target_segment_length=0.01):  
    """
    Generate cylinder meshes with gradual radius transitions between usage levels
    and branch-wise linear tapering to min radius at terminal base.
    """
    cylinders = []


    segment_usage = path_info['segment_usage']
    point_usage_counter = path_info['point_usage_counter']
    point_coordinates = path_info['point_coordinates']  # dict[int] -> np.array(3,)
    point_colors = path_info['point_colors']            # dict[int] -> (3,)
    max_segment_usage = max(segment_usage.values()) if segment_usage else 1

    point_graph = {}
    for (p1, p2), usage_count in segment_usage.items():
        point_graph.setdefault(p1, []).append((p2, usage_count))
        point_graph.setdefault(p2, []).append((p1, usage_count))

    point_radii = {}
    for point_idx in point_graph:
        max_connected_usage = max(usage for _, usage in point_graph[point_idx])
        point_own_usage = point_usage_counter.get(point_idx, 1)
        effective_usage = max(max_connected_usage, point_own_usage)
        usage_ratio = effective_usage / max(max_segment_usage, 1e-12)

        if usage_ratio >= 0.9:
            r = radius_levels[0]
        elif usage_ratio >= 0.6:
            r = radius_levels[1]
        elif usage_ratio >= 0.3:
            r = radius_levels[1] * 0.7
        else:
            r = radius_levels[2]
        point_radii[point_idx] = max(min_radius, min(r, max_radius))

    main_stem_points = set()
    for point_idx, r in point_radii.items():
        effective_usage = max(max(usage for _, usage in point_graph[point_idx]),
                              point_usage_counter.get(point_idx, 1))
        usage_ratio = effective_usage / max(max_segment_usage, 1e-12)
        if usage_ratio >= 0.6:
            main_stem_points.add(point_idx)

    for point_idx in point_graph:
        own = point_usage_counter.get(point_idx, 1)
        ratio = own / max(max_segment_usage, 1e-12)
        if ratio < 0.3:
            connected_main = [point_radii[c] for c, _ in point_graph[point_idx] if c in main_stem_points]
            if connected_main:
                inherited_radius = max(connected_main)
                adjusted_radius = inherited_radius * 0.85
                point_radii[point_idx] = max(point_radii[point_idx], adjusted_radius)
                point_radii[point_idx] = max(min_radius, min(point_radii[point_idx], max_radius))
                print(f"    LOW USAGE Point {point_idx}: inherited={inherited_radius:.4f}, "
                      f"adjusted={adjusted_radius:.4f}, final={point_radii[point_idx]:.4f}")

    true_root = max(point_usage_counter.keys(), key=lambda x: point_usage_counter[x])
    point_radii[true_root] = max(min_radius, min(radius_levels[0], max_radius))

    if abs(thickness_scale - 1.0) > 1e-9:
        for k in point_radii:
            point_radii[k] *= thickness_scale
        for k in point_radii:
            point_radii[k] = max(min_radius, min(point_radii[k], max_radius))

    enforce_monotone_radii(point_graph, point_radii, true_root,
                           transition_factor=transition_factor,
                           min_radius=min_radius, max_radius=max_radius, max_iters=4)

  
    leaves = [n for n, nbrs in ((n, point_graph.get(n, [])) for n in point_graph)
              if len(nbrs) == 1 and n not in main_stem_points]

    def trace_to_main_stem(start):
        parent = {start: None}
        q = deque([start])
        hit = None
        while q:
            u = q.popleft()
            if u in main_stem_points:
                hit = u
                break
            for v, _ in point_graph.get(u, []):
                if v not in parent:
                    parent[v] = u
                    q.append(v)
        if hit is None:
            return None 
        path = []
        cur = start
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path  
    for leaf in leaves:
        path = trace_to_main_stem(leaf)
        if not path or len(path) < 2:
            continue
        start_idx = 1 if path[0] in main_stem_points and len(path) > 1 else 0
        if start_idx >= len(path):
            continue
        branch_nodes = path[start_idx:]                 
        N = max(1, len(branch_nodes) - 1)              
        start_node = branch_nodes[0]
        r_start = float(point_radii[start_node])        
        for i, node in enumerate(branch_nodes):
            t = i / N
            target = r_start * (1 - t) + min_radius * t
            point_radii[node] = max(min_radius, min(point_radii[node], target, max_radius))

    sorted_segments = sorted(segment_usage.items(), key=lambda x: x[1], reverse=True)

    for (p1, p2), segment_usage_count in sorted_segments:
        usage1 = point_usage_counter.get(p1, 1)
        usage2 = point_usage_counter.get(p2, 1)
        if usage1 >= usage2:
            root_point, leaf_point = p1, p2
        else:
            root_point, leaf_point = p2, p1

        coord1, coord2 = point_coordinates[root_point], point_coordinates[leaf_point]
        color1, color2 = point_colors[root_point], point_colors[leaf_point]
        radius1, radius2 = point_radii[root_point], point_radii[leaf_point]

        if radius2 > radius1:
            coord1, coord2 = coord2, coord1
            color1, color2 = color2, color1
            radius1, radius2 = radius2, radius1
            root_point, leaf_point = leaf_point, root_point


        debug_colors = (len(cylinders) == 0)
        cylinder = create_tapered_cylinder(
            coord1, coord2,
            radius1, radius2,
            color1, color2,
            resolution=cylinder_resolution,
            noise_strength=noise_strength,
            debug_colors=debug_colors,
            height_segments=None,                    
            target_segment_length=target_segment_length
        )
        if cylinder is not None:
            cylinders.append(cylinder)

    # print(f"Generated {len(cylinders)} cylinder segments")
    return cylinders


def create_test_cylinder():
    """Create a simple test tapered cylinder for verification"""
    vertices = []
    faces = []
    resolution = 8
    height = 1.0
    radius1 = 0.1
    radius2 = 0.05

    angles = np.linspace(0, 2*np.pi, resolution, endpoint=False)

    for angle in angles:
        x = radius1 * np.cos(angle)
        y = radius1 * np.sin(angle)
        z = 0
        vertices.append([x, y, z])

    for angle in angles:
        x = radius2 * np.cos(angle)
        y = radius2 * np.sin(angle)
        z = height
        vertices.append([x, y, z])

    for i in range(resolution):
        ni = (i + 1) % resolution
        faces.append([i, ni, resolution + i])
        faces.append([ni, resolution + ni, resolution + i])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=float))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    mesh.paint_uniform_color([1.0, 0.0, 0.0])
    mesh.compute_vertex_normals()
    return mesh
