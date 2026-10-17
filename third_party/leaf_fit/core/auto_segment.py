from typing import List, Optional, Tuple, Dict, Any
from scipy.spatial import cKDTree
import potpourri3d as pp3d
from tqdm import tqdm
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

from gaussian_utils import GaussianData
from apex_grouping import group_apexes_by_inequality
from petiole_detection import find_base_idx_by_geodesic_density, find_base_idx_by_euclidean_density, \
    find_base_idx_by_euclidean_graph, find_base_idx_by_geodesic_tip_density, \
    find_base_idx_by_geodesic_tip_graph, find_base_idx_by_geometry_feature
    
def fix_plant_root_direction_legacy(gaussians: GaussianData,
                                    opacity_threshold: float = 0.1,
                                    given_root_idx = None):
    if opacity_threshold > 0:
        opacity_mask = (gaussians.opacity > np.percentile(gaussians.opacity, opacity_threshold * 100)).flatten()
        new_gaussians = GaussianData(
            xyz=gaussians.xyz[opacity_mask],
            rot=gaussians.rot[opacity_mask],
            scale=gaussians.scale[opacity_mask],
            opacity=gaussians.opacity[opacity_mask],
            sh=gaussians.sh[opacity_mask],
            nxnynz=gaussians.nxnynz[opacity_mask],
            filter_3Ds=gaussians.filter_3Ds[opacity_mask]
        )
    else:
        new_gaussians = gaussians
    
    points = new_gaussians.xyz
    try:
        solver = pp3d.PointCloudHeatSolver(points, t_coef=1e+8)
    except Exception as e:
        raise e
    
    if given_root_idx is not None:
        root_idx = given_root_idx
    else:
        centroid = points.mean(axis=0)
        
        centered_points = points - centroid
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        asymmetries = []
        axis_info = []
        
        for i, vec in enumerate(eigenvectors.T):
            projections = np.dot(centered_points, vec)
            proj_min, proj_max = projections.min(), projections.max()
            axis_midpoint = (proj_min + proj_max) / 2.0
            
            left_count = np.sum(projections < axis_midpoint)
            right_count = np.sum(projections >= axis_midpoint)
            total = len(projections)
            asymmetry = abs(left_count - right_count) / total
            
            asymmetries.append(asymmetry)
            axis_info.append({
                'vector': vec,
                'projections': projections,
                'midpoint': axis_midpoint,
                'min_proj': proj_min,
                'max_proj': proj_max,
                'left_count': left_count,
                'right_count': right_count,
                'asymmetry': asymmetry
            })
            
            print(f"PC{i+1}: asymmetry={asymmetry:.3f}, left={left_count}({left_count/total*100:.1f}%), right={right_count}({right_count/total*100:.1f}%)")
        
        most_asymmetric_idx = np.argmax(asymmetries)
        most_asymmetric_axis = axis_info[most_asymmetric_idx]
        
        projections = most_asymmetric_axis['projections']
        proj_1st = np.percentile(projections, 1)
        proj_99th = np.percentile(projections, 99)
        
        robust_midpoint = (proj_1st + proj_99th) / 2.0
        
        robust_left_count = np.sum(projections < robust_midpoint)
        robust_right_count = np.sum(projections >= robust_midpoint)
        
        if robust_left_count < robust_right_count:
            sparse_side = "left"
            candidate_threshold = np.percentile(projections, 5)
            candidate_mask = projections <= candidate_threshold
        else:
            sparse_side = "right"
            candidate_threshold = np.percentile(projections, 95)
            candidate_mask = projections >= candidate_threshold
        
        candidate_indices = np.where(candidate_mask)[0]
        candidate_points = points[candidate_indices]
        candidate_projections = projections[candidate_mask]
        
        if sparse_side == "left":
            extreme_proj_idx = np.argmin(candidate_projections)
            extreme_proj_val = np.min(candidate_projections)
        else:
            extreme_proj_idx = np.argmax(candidate_projections)
            extreme_proj_val = np.max(candidate_projections)
        
        extreme_candidate_idx = candidate_indices[extreme_proj_idx]
        
        try:
            geodesic_distances = solver.compute_distance(extreme_candidate_idx)
            candidate_geodesic_dists = geodesic_distances[candidate_indices]
            distance_threshold = np.percentile(candidate_geodesic_dists, 15)
            distance_threshold = max(distance_threshold, geodesic_distances.mean() * 0.05)
            neighborhood_mask = geodesic_distances <= distance_threshold
            neighborhood_indices = np.where(neighborhood_mask)[0]
            neighborhood_projections = projections[neighborhood_indices]
            
        except Exception as e:
            neighborhood_indices = candidate_indices
            neighborhood_projections = projections[neighborhood_indices]
        
        if sparse_side == "left":
            neighborhood_local_idx = np.argmin(neighborhood_projections)
        else:
            neighborhood_local_idx = np.argmax(neighborhood_projections)
        
        root_idx = int(neighborhood_indices[neighborhood_local_idx])

    return new_gaussians, root_idx, solver
    
def get_temperature_field(solver: pp3d.PointCloudHeatSolver, heat_source_idx: list):
    if len(heat_source_idx) == 1:
        geodesic_distance_field = solver.compute_distance(heat_source_idx[0])
    else:
        geodesic_distance_field = solver.compute_distance_multisource(heat_source_idx)
    temperature_field = geodesic_distance_field.max() - geodesic_distance_field
    return temperature_field
    
def find_local_tips(gaussians: GaussianData, 
                    sparse_indices: np.ndarray, 
                    mapping: np.ndarray, 
                    temperature_field: np.ndarray, 
                    tree: cKDTree, 
                    k=512):
    sparse_xyz = gaussians.xyz[sparse_indices]
    found_tips = []
    found_tips_sparse = []
    for i in tqdm(range(sparse_xyz.shape[0])):
        # Get the temperature field of the sparse point
        temp = temperature_field[sparse_indices[i]] # temp of current sparse point
        dist, local_mask = tree.query(sparse_xyz[i], k=k)

        local_mask = mapping[local_mask]
        # Map sparse local_mask back to full indices
        full_local_indices = sparse_indices[local_mask[1:]]
        temp_mask = temperature_field[full_local_indices]
        if np.all(temp_mask >= temp):
            found_tips.append(sparse_indices[i])
            found_tips_sparse.append(i)
        
    return found_tips

def find_path_from_tip_to_root(gaussians: GaussianData, 
                        temperature_field: np.ndarray, 
                        tip_idx: int, 
                        heat_source_idx: int, 
                        distance_param: dict,
                        is_path_marks: np.ndarray,
                        k=128):
    tree = None
    sparse_solver = None
    if distance_param is not None:
        if distance_param['method'] == 'euclidean':
            tree = distance_param['tree']
            dense_solver = distance_param['dense_solver']
            _temperature_field = temperature_field - 0.25 * dense_solver.compute_distance(tip_idx)
        elif distance_param['method'] == 'geodesic':
            sparse_solver = distance_param['sparse_solver']
            dense_solver = distance_param['dense_solver']
            _temperature_field = temperature_field - 0.5 * dense_solver.compute_distance(tip_idx)
            basisX, basisY, basisN = dense_solver.get_tangent_frames()
            sparse_indices = distance_param['sparse_indices']
            mapping = distance_param['mapping']
        debug = distance_param.get('debug', False) 
    xyz = gaussians.xyz
    path = [tip_idx]
    i = tip_idx
    while i != heat_source_idx:
        if tree is not None:
            dists, idx = tree.query(xyz[i], k=k)
        elif sparse_solver is not None:
            # Find sparse position corresponding to dense index i
            sparse_i = np.where(sparse_indices == i)[0]
            if len(sparse_i) == 0:
                raise ValueError(f"Dense index {i} not found in sparse_indices")
            sparse_i = sparse_i[0]
            
            sparse_dists = sparse_solver.compute_distance(sparse_i)
            sparse_idx = np.where(sparse_dists <= sparse_dists.mean() * 0.025)[0]
            # Convert sparse indices back to dense indices
            idx = sparse_indices[sparse_idx]
        else:
            raise ValueError("Invalid distance parameter")
            
        if len(path) > 3:
            # Find marked neighbors
            marked_mask = is_path_marks[idx] >= 1
            if np.any(marked_mask):
                marked_neighbor_indices = idx[marked_mask]
                marked_neighbor_temps = _temperature_field[marked_neighbor_indices]

                best_marked_local_idx = np.argmax(marked_neighbor_temps)
                best_marked_global_idx = marked_neighbor_indices[best_marked_local_idx]
                best_marked_temp = marked_neighbor_temps[best_marked_local_idx]

                if best_marked_temp > _temperature_field[i]:
                    next_idx = best_marked_global_idx
                    max_temp_idx = np.where(idx == next_idx)[0][0]
                else:
                    temp_values = _temperature_field[idx]
                    max_temp_idx = np.argmax(temp_values)
            else:
                temp_values = _temperature_field[idx]
                max_temp_idx = np.argmax(temp_values)
        else:   
            temp_values = _temperature_field[idx]
            max_temp_idx = np.argmax(temp_values)
            
        if idx[max_temp_idx] in path:
            break
        
        is_path_marks[idx[max_temp_idx]] += 1

        path.append(int(idx[max_temp_idx]))
        i = idx[max_temp_idx]

    return path

def find_earliest_intersection_indices(
    paths: List[List[int]],
    forbid_tip: bool = True,
    return_detail: bool = False,
) -> List[Optional[int]] | List[Dict[str, Any]]:

    # Normalize to int to avoid key inconsistencies from types like np.uint64
    P = [[int(x) for x in p] for p in paths]
    n = len(P)

    # Build reverse index: node -> [(path_id, pos), ...]
    occ: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for pid, path in enumerate(P):
        for pos, node in enumerate(path):
            occ[node].append((pid, pos))

    results_idx: List[Optional[int]] = [None] * n
    if return_detail:
        results_detail: List[Dict[str, Any]] = [
            {"index": None, "node": None, "partners": []} for _ in range(n)
        ]

    # For each path, scan from tip to root to find earliest intersection with other paths
    start_offset = 1 if forbid_tip else 0
    for pid, path in enumerate(P):
        # Edge case: empty path or only tip
        if len(path) <= start_offset:
            if return_detail:
                results_detail[pid] = {"index": None, "node": None, "partners": []}
            continue

        chosen_idx: Optional[int] = None
        chosen_node: Optional[int] = None
        partners: List[int] = []

        for pos in range(start_offset, len(path)):
            node = path[pos]
            # Check this node's occurrences in other paths (filtered by forbid_tip)
            entries = occ.get(node, [])
            # Check if at least one valid occurrence exists in other paths
            valid_partners = [
                qid for (qid, qpos) in entries
                if qid != pid and (not forbid_tip or qpos > 0)
            ]
            if valid_partners:
                chosen_idx = pos
                chosen_node = node
                partners = sorted(set(valid_partners))
                break  

        if return_detail:
            results_detail[pid] = {
                "index": chosen_idx,
                "node": chosen_node,
                "partners": partners,
            }
        else:
            results_idx[pid] = chosen_idx

    return results_detail if return_detail else results_idx

def get_segment_mask(gaussians: GaussianData, 
                     sparse_indices: np.ndarray, 
                     mapping: np.ndarray, 
                     solver: pp3d.PointCloudHeatSolver, 
                     tree: cKDTree, 
                     heat_source_idx: list,
                     method: str,
                     cached_root_distances: np.ndarray = None,
                     debug_vis=False):
    
    temperature_field = get_temperature_field(solver, heat_source_idx)
    found_tips = find_local_tips(gaussians, 
                                    sparse_indices, 
                                    mapping, 
                                    temperature_field, 
                                    tree, 
                                    k=len(sparse_indices) // 64
                                    )
    is_path_marks = np.zeros(len(gaussians.xyz), dtype=int)
    
    pathes = []
    for i in found_tips:
        path = find_path_from_tip_to_root(gaussians, 
                                          temperature_field, 
                                          i, 
                                          heat_source_idx[0], 
                                          {
                                                "method": "euclidean",
                                                "tree": tree,
                                                "dense_solver": solver
                                        }, 
                                          is_path_marks, 
                                          k= len(sparse_indices) // 32)
        pathes.append(path)

    cluster_info = group_apexes_by_inequality(
        found_tips,
        pathes,
        overlap_cut=0.8,
        root_cahced_distance=cached_root_distances,
        dense_solver=solver,
    )

    pathes = [ci['path'] for ci in cluster_info]
    lcas = find_earliest_intersection_indices(pathes)
    for ci, lca, path in zip(cluster_info, lcas, pathes):
        ci['lca'] = path[lca] if lca is not None else path[-1]

    final_cluster_results = calculate_cluster_base_indices(cluster_info, 
                                                           sparse_indices,
                                                           mapping, 
                                                           solver, 
                                                           gaussians, 
                                                           temperature_field, 
                                                           method)

    found_tips = []
    found_bases = []
    found_segs  = []
    found_geodist_from_tip = []
    
    for result in final_cluster_results:
        tips = result['tips']
        selected_tip = result['selected_tip']
        base_idx = result['base_idx']
        type = result['type']
        found_bases.append(base_idx)
        
        geodist_from_tip = solver.compute_distance(selected_tip)
        if type == 'single_tip':
            found_tips.append(tips)
            _distance_factor = 1
            distance_factor = 1
        elif type == 'multi_tips':
            found_tips.append([selected_tip] + tips)
            _distance_factor = 1
            distance_factor = 1.25
            
        dist_from_root_to_base = cached_root_distances[base_idx]
        dist_from_tip_to_base = geodist_from_tip[base_idx]
        
        indices_from_root_to_base = np.where(cached_root_distances >= dist_from_root_to_base * _distance_factor)[0]
        indices_from_tip_to_base = np.where(geodist_from_tip <= dist_from_tip_to_base * distance_factor)[0]
        segment_indices = np.intersect1d(indices_from_root_to_base, indices_from_tip_to_base)
        
        # Remove overlapping indices with previously found segments
        if found_segs:
            all_previous_indices = np.concatenate(found_segs)
            overlapping_indices = np.intersect1d(segment_indices, all_previous_indices)
            segment_indices = np.setdiff1d(segment_indices, overlapping_indices)
        found_geodist_from_tip.append(geodist_from_tip)
        found_segs.append(segment_indices)

    # Use paths from root to base for stem generation
    root_to_base_paths = []
    point_usage_counter = {}

    # First pass: collect paths from BASE to ROOT by finding base in existing paths
    for cluster in final_cluster_results:
        base_idx = cluster['base_idx']
        base_to_root_path = None

        # Find which path contains this base_idx
        for path in pathes:
            if base_idx in path:
                base_position = path.index(base_idx)
                # Cut from one point before base (closer to tip) to end (root)
                if base_position > 0:
                    start_position = base_position - 1
                    base_to_root_path = path[start_position:]
                else:
                    base_to_root_path = path[base_position:]
                break

        if base_to_root_path is None:
            continue

        # Reverse to get root -> base path
        root_to_base_path = base_to_root_path[::-1]
        root_to_base_paths.append(root_to_base_path)

    # Rename for consistency with rest of code
    cleaned_found_pathes = root_to_base_paths

    # Second pass: count usage of each point across all root-to-base paths
    for root_to_base_path in cleaned_found_pathes:
        for point_idx in root_to_base_path:
            if point_idx not in point_usage_counter:
                point_usage_counter[point_idx] = 0
            point_usage_counter[point_idx] += 1

    # Categorize points by usage frequency
    high_usage_points = []
    medium_usage_points = []
    low_usage_points = []
    
    if point_usage_counter:
        max_usage = max(point_usage_counter.values())
        for point_idx, usage_count in point_usage_counter.items():
            if usage_count >= max_usage * 0.8:
                high_usage_points.append((point_idx, usage_count))
            elif usage_count >= max_usage * 0.4:
                medium_usage_points.append((point_idx, usage_count))
            else:
                low_usage_points.append((point_idx, usage_count))

    # Calculate segment usage for cylinder generation
    segment_usage = {}
    for full_path in cleaned_found_pathes:
        for i in range(len(full_path) - 1):
            p1, p2 = full_path[i], full_path[i + 1]
            # Use consistent ordering for segment keys
            segment_key = (min(p1, p2), max(p1, p2))
            if segment_key not in segment_usage:
                segment_usage[segment_key] = 0
            segment_usage[segment_key] += 1

    # Save coordinates and colors for all points involved in paths
    point_coordinates = {}
    point_colors = {}
    all_path_points = set()
    for path in cleaned_found_pathes:
        all_path_points.update(path)

    # SH level 0 to RGB conversion constant
    SH_C0 = 0.28209479177387814

    for point_idx in all_path_points:
        point_coordinates[point_idx] = gaussians.xyz[point_idx]

        # Convert SH level 0 to RGB
        if hasattr(gaussians, 'sh') and gaussians.sh is not None and gaussians.sh.shape[1] >= 3:
            sh_coeffs = gaussians.sh[point_idx, :3]
            rgb_color = SH_C0 * sh_coeffs + 0.5
            rgb_color = np.clip(rgb_color, 0.0, 1.0)
            point_colors[point_idx] = rgb_color
        else:
            if hasattr(gaussians, 'colors') and gaussians.colors is not None:
                rgb_color = gaussians.colors[point_idx]
                point_colors[point_idx] = np.clip(rgb_color, 0.0, 1.0)
            else:
                # Use varied default colors based on point index
                hue = (point_idx * 0.618033988749) % 1.0
                if hue < 0.33:
                    base_color = np.array([0.2 + hue, 0.6, 0.2])
                elif hue < 0.66:
                    base_color = np.array([0.6, 0.4 + (hue-0.33)*0.6, 0.2])
                else:
                    base_color = np.array([0.4, 0.6, 0.2 + (hue-0.66)*0.6])
                point_colors[point_idx] = np.clip(base_color, 0.0, 1.0)

   
    path_analysis_data = {
        'cleaned_paths': cleaned_found_pathes,
        'point_usage_counter': point_usage_counter,
        'segment_usage': segment_usage,
        'point_coordinates': point_coordinates,
        'point_colors': point_colors,
        'high_usage_points': high_usage_points if 'high_usage_points' in locals() else [],
        'medium_usage_points': medium_usage_points if 'medium_usage_points' in locals() else [],
        'low_usage_points': low_usage_points if 'low_usage_points' in locals() else [],
    }
    
    return found_segs, found_tips, found_bases, found_geodist_from_tip, final_cluster_results, path_analysis_data

def calculate_cluster_base_indices(cluster_info, 
                                   sparse_indices, 
                                   mapping, 
                                   solver, 
                                   gaussians, 
                                   temperature_field, 
                                   method='euclidean_density'):

    final_cluster_results = []
    nbrs_reusable = NearestNeighbors(radius=0.05, algorithm='ball_tree').fit(gaussians.xyz)

    for cluster_idx, cluster_info_item in enumerate(cluster_info):
        selected_tip = None
        optimal_base_idx = None  

        start_time = time.time()
        if method == 'euclidean_density':
            optimal_base_idx = find_base_idx_by_euclidean_density(
                cluster_info_item, sparse_indices, mapping, gaussians,
                euclidean_radius=0.1, truncate_ratio=0.1
            )
        elif method == 'geodesic_density':
            optimal_base_idx = find_base_idx_by_geodesic_density(
                cluster_info_item, sparse_indices, solver,
                radius_threshold=0.1, density_percentile=75.0, truncate_ratio=0.1
            )
        elif method == 'geodesic_tip_density':
            optimal_base_idx = find_base_idx_by_geodesic_tip_density(
                cluster_info_item, sparse_indices, solver,
                recovery_threshold=0.25, truncate_ratio=0.1
            )
        elif method == 'geometry_feature_euclidean':
            optimal_base_idx = find_base_idx_by_geometry_feature(
                cluster_info_item, sparse_indices, mapping, gaussians,
                neighbor_selection={'method': 'euclidean','radius': 0.1,'ratio_threshold': 5.0}
            )
        elif method == 'geometry_feature_geodesic':
            optimal_base_idx = find_base_idx_by_geometry_feature(
                cluster_info_item, sparse_indices, mapping, gaussians,
                neighbor_selection={'method': 'geodesic','radius': 0.1,'ratio_threshold': 5.0,'dense_solver': solver}
            )
        elif method == 'geometry_feature_geodesic_donuts':
            optimal_base_idx = find_base_idx_by_geometry_feature(
                cluster_info_item, sparse_indices, mapping, gaussians,
                neighbor_selection={'method': 'geodesic_donuts','radius': 0.1,'radius_multiplier': 2.0,'dense_solver': solver}
            )
        elif method == 'geometry_feature_geodesic_tip':
            optimal_base_idx = find_base_idx_by_geometry_feature(
                cluster_info_item, sparse_indices, mapping, gaussians,
                neighbor_selection={'method': 'geodesic_tip','ratio_threshold': 5.0,'dense_solver': solver}
            )
        elif method == 'euclidean_graph':
            optimal_base_idx = find_base_idx_by_euclidean_graph(
                cluster_info_item, gaussians, nbrs_reusable, temperature_field,
                temperature_percentile=50, max_iterations_factor=2,
                path_distance_factor=1.5, protection_period_ratio=0.25
            )
        elif method == 'geodesic_tip_graph':
            optimal_base_idx = find_base_idx_by_geodesic_tip_graph(
                cluster_info_item, gaussians, solver, temperature_field,
                min_distance_threshold=0.05, tolerance_percentage=0.02,
                last_virtual_path_distance_factor=2.5, protection_period_ratio=0.25,
                debug_vis=False
            )
        end_time = time.time()
        selected_tip = cluster_info_item['tips'][0]

        final_cluster_results.append({
            'cluster_idx': cluster_idx + 1,
            'tips': cluster_info_item['tips'],
            'selected_tip': selected_tip,
            'type': cluster_info_item['type'],
            'base_idx': optimal_base_idx,
            'path': cluster_info_item['path']
        })


    return final_cluster_results
