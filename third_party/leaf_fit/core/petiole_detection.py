import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import fpsample
from sklearn.decomposition import PCA
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

def find_base_idx_by_euclidean_density(cluster_info_item, 
                             sparse_indices, mapping, 
                             gaussians, 
                             euclidean_radius=0.1, 
                             truncate_ratio=0.1):

    tips = cluster_info_item['tips'] 
    potential_path = cluster_info_item['potential_base_idx']
    cluster_type = cluster_info_item['type']
    
    if len(potential_path) == 0:
        return tips[0] if tips else None
    
    if cluster_type == 'multi_tip' and len(tips) > 1:
        return potential_path[0] if potential_path else tips[0]
    
    print(f"\n--- find_base_idx_by_density: Processing {cluster_type} cluster ---")
    print(f"Original path length: {len(potential_path)} points")
    
    path_length = len(potential_path)
    start_truncate = int(path_length * truncate_ratio)
    
    if start_truncate >= path_length - 3:
        print("Path too short after truncation, using original path")
        truncated_path = potential_path
    else:
        truncated_path = potential_path[start_truncate:]
        print(f"Truncated path: removed first {start_truncate} points")
        print(f"Working with {len(truncated_path)} points (starting from index {start_truncate})")
    
    potential_path = truncated_path
    
    sparse_1x = sparse_indices
    sparse_pos_1x = gaussians.xyz[sparse_1x]
    
    target_2x = len(sparse_1x) // 2
    sparse_2x_fps = fpsample.bucket_fps_kdline_sampling(sparse_pos_1x, target_2x, h=3)
    sparse_2x = sparse_1x[sparse_2x_fps]
    sparse_pos_2x = sparse_pos_1x[sparse_2x_fps]
    
    target_4x = len(sparse_1x) // 4
    sparse_4x_fps = fpsample.bucket_fps_kdline_sampling(sparse_pos_1x, target_4x, h=3)
    sparse_4x = sparse_1x[sparse_4x_fps]
    sparse_pos_4x = sparse_pos_1x[sparse_4x_fps]
    
    scales = [
        {'name': '1x', 'indices': sparse_1x, 'positions': sparse_pos_1x},
        {'name': '2x', 'indices': sparse_2x, 'positions': sparse_pos_2x},
        {'name': '4x', 'indices': sparse_4x, 'positions': sparse_pos_4x}
    ]
    
    nbrs_scales = {}
    for scale in scales:
        nbrs_scales[scale['name']] = NearestNeighbors(radius=euclidean_radius, algorithm='ball_tree').fit(scale['positions'])
    
    multi_scale_densities = []
    valid_nodes = []
    
    for path_pos, dense_idx in enumerate(potential_path):
        if dense_idx < len(mapping):
            node_coord = gaussians.xyz[dense_idx]
            scale_densities = []
            
            for scale in scales:
                nbrs = nbrs_scales[scale['name']]
                neighbors = nbrs.radius_neighbors([node_coord], return_distance=False)[0]
                density = len(neighbors)
                scale_densities.append(density)
            
            avg_density = np.mean(scale_densities)
            multi_scale_densities.append(scale_densities + [avg_density])
            valid_nodes.append((path_pos, dense_idx, scale_densities, avg_density))
    
    if len(valid_nodes) < 2:
        return potential_path[0] if potential_path else tips[0]
    
    scale_1x_densities = [row[0] for row in multi_scale_densities]  
    mean_density_1x = np.mean(scale_1x_densities)
    
    candidate_indices = []
    
    if cluster_type == 'single_tip':
        for i in range(len(scale_1x_densities)):
            if scale_1x_densities[i] < mean_density_1x:
                candidate_indices.append(i)
    else:
        candidate_indices = list(range(len(scale_1x_densities)))
    
    if not candidate_indices:
        candidate_indices = list(range(len(scale_1x_densities)))
    
    scale_names = ['1x', '2x', '4x', 'avg']
    scale_results = {}
    
    for scale_idx, scale_name in enumerate(scale_names):
        scale_densities = [row[scale_idx] for row in multi_scale_densities]
        
        max_gradient = 0
        gradient_base_idx = valid_nodes[0][1]
        
        for i in candidate_indices:
            if i == 0:
                continue  
            if i-1 not in candidate_indices:
                continue  
                
            gradient = scale_densities[i-1] - scale_densities[i]
            if gradient > max_gradient:
                max_gradient = gradient
                gradient_base_idx = valid_nodes[i][1]
        
        min_density_info = min(enumerate(scale_densities), key=lambda x: x[1])
        min_pos, min_density = min_density_info
        min_node = valid_nodes[min_pos][1]
        
        if max_gradient == 0:
            gradient_base_idx = min_node
        
        scale_results[scale_name] = {
            'gradient_base_idx': gradient_base_idx,
            'min_density_base_idx': min_node,
        }
    
    gradient_bases = [result['gradient_base_idx'] for result in scale_results.values()]
    
    if not gradient_bases:
        return potential_path[0] if potential_path else tips[0]
    
    gradient_vote = Counter(gradient_bases).most_common(1)
    
    if not gradient_vote:
        return potential_path[0] if potential_path else tips[0]
    
    return gradient_vote[0][0]

def find_base_idx_by_geodesic_density(cluster_info_item, 
                                      sparse_indices, 
                                      dense_solver, 
                                      radius_threshold=0.1, 
                                      density_percentile=75.0,
                                      truncate_ratio=0.1):

    tips = cluster_info_item['tips'] 
    potential_path = cluster_info_item['potential_base_idx']
    cluster_type = cluster_info_item['type']
    
    if len(potential_path) == 0:
        return tips[0] if tips else None
    
    if cluster_type == 'multi_tip' and len(tips) > 1:
        return potential_path[0] if potential_path else tips[0]
    
    path_length = len(potential_path)
    start_truncate = int(path_length * truncate_ratio)
    
    if start_truncate >= path_length - 3:
        truncated_path = potential_path
    else:
        truncated_path = potential_path[start_truncate:]
    
    potential_path = truncated_path
    
    point_densities = []
    point_diffs = []
    point_thresholds = []
    point_indices = []
    
    prev_density = None
    
    for _, path_idx in enumerate(potential_path):
        geodesic_distances = dense_solver.compute_distance(path_idx)
        
        sparse_geodesic_distances = geodesic_distances[sparse_indices]
        neighbor_count = np.sum(sparse_geodesic_distances < radius_threshold)
        
        point_densities.append(neighbor_count)
        point_thresholds.append(radius_threshold)
        point_indices.append(path_idx)
        
        if prev_density is not None:
            diff = prev_density - neighbor_count
            point_diffs.append(diff)
        else:
            point_diffs.append(0)
        
        prev_density = neighbor_count
    
    
    if point_densities:
        density_threshold = np.percentile(point_densities, density_percentile)
        
        low_density_mask = np.array(point_densities) <= density_threshold
        
        
        if np.any(low_density_mask):
            filtered_diffs = np.array(point_diffs)[low_density_mask]
            if len(filtered_diffs) > 0:
                max_diff_idx_in_filtered = np.argmax(filtered_diffs)
                filtered_indices = np.where(low_density_mask)[0]
                original_idx = filtered_indices[max_diff_idx_in_filtered]
                
                selected_path_idx = potential_path[original_idx]
                
                if original_idx + 1 < len(potential_path):
                    final_selected_idx = potential_path[original_idx + 1]
                    return final_selected_idx
                else:
                    return selected_path_idx
        
        return potential_path[0]
    else:
        return potential_path[0] if potential_path else tips[0]

def find_base_idx_by_geodesic_tip_density(cluster_info_item, 
                                          sparse_indices, 
                                          dense_solver, 
                                          recovery_threshold=0.25,
                                          truncate_ratio=0.1):

    tips = cluster_info_item['tips']
    potential_path = cluster_info_item['potential_base_idx']
    cluster_type = cluster_info_item['type']
    
    if not potential_path:
        return tips[0]
    
    if cluster_type == 'multi_tip' and len(tips) > 1:
        return potential_path[0] if potential_path else tips[0]
    
    path_length = len(potential_path)
    start_truncate = int(path_length * truncate_ratio)
    
    if start_truncate >= path_length - 3:
        truncated_path = potential_path
    else:
        truncated_path = potential_path[start_truncate:]
    
    potential_path = truncated_path
    
    tip = tips[0]
    single_tip_distances = dense_solver.compute_distance(tip)
    
    point_counts = []
    point_diffs = []
    point_thresholds = []
    point_indices = []
    
    prev_count = None
    for i, path_idx in enumerate(potential_path):
        threshold_geodesic = single_tip_distances[path_idx]
        sparse_geodesic_distances = single_tip_distances[sparse_indices]
        selected_count = np.sum(sparse_geodesic_distances < threshold_geodesic)
        
        point_counts.append(selected_count)
        point_thresholds.append(threshold_geodesic)
        point_indices.append(path_idx)
        
        if prev_count is not None:
            diff = prev_count - selected_count
            point_diffs.append(diff)
        else:
            point_diffs.append(0)
        
        prev_count = selected_count
        
    
    if len(point_diffs) > 1:
        valid_diffs = np.array(point_diffs[1:])  
        valid_indices = list(range(1, len(point_diffs)))  
        
        if len(valid_diffs) > 0:
            first_max_increase_idx = np.argmin(valid_diffs)
            first_max_increase_pos = valid_indices[first_max_increase_idx]
            max_increase_value = valid_diffs[first_max_increase_idx]
            
            peak_end_threshold = max_increase_value * recovery_threshold
            search_start = len(point_diffs)  
            
            for i in range(first_max_increase_pos + 1, len(point_diffs)):
                if point_diffs[i] > peak_end_threshold:
                    search_start = i
                    break
            
            if search_start < len(point_diffs):
                remaining_diffs = np.array(point_diffs[search_start:])
                remaining_indices = list(range(search_start, len(point_diffs)))
                
                if len(remaining_diffs) > 0:
                    second_min_diff_idx_in_remaining = np.argmin(remaining_diffs)
                    second_min_diff_pos = remaining_indices[second_min_diff_idx_in_remaining]
                    
                    if second_min_diff_pos > search_start:
                        midpoint = (search_start + second_min_diff_pos) // 2
                        
                        search_range_diffs = np.array(point_diffs[search_start:midpoint])
                        search_range_indices = list(range(search_start, midpoint))
                        
                        if len(search_range_diffs) > 0:
                            max_diff_value = np.max(search_range_diffs)
                            first_max_idx_in_range = np.argmax(search_range_diffs == max_diff_value)
                            selected_idx = search_range_indices[first_max_idx_in_range]
                            
                            base_idx = max(0, selected_idx - 1)
                            selected_base_idx = point_indices[base_idx] if base_idx < len(point_indices) else point_indices[selected_idx]
                        else:
                            selected_base_idx = point_indices[search_start]
                    else:
                        max_diff_idx_in_remaining = np.argmax(remaining_diffs)
                        selected_idx = remaining_indices[max_diff_idx_in_remaining]
                        base_idx = max(0, selected_idx - 1)
                        selected_base_idx = point_indices[base_idx] if base_idx < len(point_indices) else point_indices[selected_idx]
                else:
                    selected_base_idx = point_indices[first_max_increase_pos]
            else:
                selected_base_idx = point_indices[first_max_increase_pos]
        else:
            selected_base_idx = potential_path[0] if potential_path else tips[0]
    else:
        selected_base_idx = potential_path[0] if potential_path else tips[0]
    
    return selected_base_idx


def get_connected_component_ratio(points_delta, radius=0.01, ignore_size=0):
 
    if len(points_delta) == 0:
        return 0, 0.0, [], []

    tree = cKDTree(points_delta)
    
    dist = cdist(points_delta, points_delta)
    dist = dist[np.triu_indices(dist.shape[0], 1)]
    radius = np.percentile(dist, 25)
    
    pairs = tree.query_pairs(radius)

    G = nx.Graph()
    G.add_nodes_from(range(len(points_delta)))
    G.add_edges_from(pairs)

    components = list(nx.connected_components(G))

    valid_components = [c for c in components if len(c) >= ignore_size]
    if len(valid_components) == 0:
        return 0, 0.0, np.full(len(points_delta), -1, dtype=int), []

    sizes = [len(c) for c in valid_components]
    largest_size = max(sizes)
    total_valid = sum(sizes)
    largest_ratio = largest_size / total_valid

    labels = np.full(len(points_delta), -1, dtype=int)
    for new_label, comp in enumerate(valid_components):
        for idx in comp:
            labels[idx] = new_label

    return len(valid_components), largest_ratio, labels, sizes


def find_base_idx_by_geometry_feature(cluster_info_item, 
                                      sparse_indices,
                                      mapping, 
                                      gaussians, 
                                      truncate_ratio=0.1,
                                      neighbor_selection={'method': 'euclidean', 'radius': 0.1, 'ratio_threshold': 5.0, 
                                                          'inner_radius': 0.05, 'radius_multiplier': 2.0}):
    
    tips = cluster_info_item['tips']
    potential_path = cluster_info_item['potential_base_idx']
    cluster_type = cluster_info_item['type']
    method = neighbor_selection.get('method', 'euclidean')
    radius = neighbor_selection.get('radius', 0.1)
    ratio_threshold = neighbor_selection.get('ratio_threshold', 5.0)
    dense_solver = neighbor_selection.get('dense_solver', None)
    
    if method == 'geodesic_tip':
        true_tip = tips[0]
    
    if len(potential_path) == 0:
        return tips[0] if tips else None
    
    path_length = len(potential_path)
    start_truncate = int(path_length * truncate_ratio)
    
    if start_truncate >= path_length - 3:
        truncated_path = potential_path
    else:
        truncated_path = potential_path[start_truncate:]
    
    potential_path = truncated_path
    
    if cluster_type == 'multi_tip' and len(tips) > 1:
        return potential_path[0] if potential_path else tips[0]
    

    
    donuts_inner_radius = neighbor_selection.get('inner_radius', 0.05)     
    radius_multiplier = neighbor_selection.get('radius_multiplier', 2.0)  
    
    if method not in ['euclidean', 'geodesic', 'geodesic_donuts', 'geodesic_tip']:
        raise ValueError(f"Unsupported neighbor selection method: {method}")
    
    if method in ['geodesic', 'geodesic_donuts', 'geodesic_tip'] and dense_solver is None:
        raise ValueError("dense_solver is required when method='geodesic' or 'geodesic_donuts'")
    
    if method == 'euclidean':
        sparse_positions = gaussians.xyz[sparse_indices]
        nbrs = NearestNeighbors(radius=radius, algorithm='ball_tree').fit(sparse_positions)
    
    linear_planar_ratios = []
    
    if method == 'euclidean' or method == 'geodesic':
        for i, path_idx in enumerate(potential_path):
            if method == 'euclidean':
                path_coord = gaussians.xyz[path_idx]
                neighbor_indices = nbrs.radius_neighbors([path_coord], return_distance=False)[0]
            elif method == 'geodesic':
                geodesic_distances = dense_solver.compute_distance(path_idx)
                sparse_geodesic_distances = geodesic_distances[sparse_indices]
                neighbor_mask = sparse_geodesic_distances < radius
                neighbor_indices = np.where(neighbor_mask)[0]
                
            if len(neighbor_indices) < 3:  
                linear_planar_ratios.append(0.0)  
                continue
            
            sparse_positions = gaussians.xyz[sparse_indices]
            neighbor_coords = sparse_positions[neighbor_indices]
            pca = PCA(n_components=3)
            pca.fit(neighbor_coords)
            
            eigenvalues_ratio = pca.explained_variance_ratio_
            
            linearity = eigenvalues_ratio[0]
            planarity = eigenvalues_ratio[1]
            
            lp_ratio = linearity / planarity if planarity > 1e-10 else float('inf')
            linear_planar_ratios.append(lp_ratio)
            
            if len(linear_planar_ratios) > 1 and \
                lp_ratio > ratio_threshold or \
                    lp_ratio == float('inf'):
                return potential_path[i]
                
        if len(linear_planar_ratios) > 1:
            valid_ratios = linear_planar_ratios[1:]  
            max_ratio_idx = np.argmax(valid_ratios) + 1  
            return potential_path[max_ratio_idx]
        else:
            return potential_path[0]
        
    elif method == 'geodesic_donuts':
        for i, path_idx in enumerate(potential_path):
            geodesic_distances = dense_solver.compute_distance(path_idx)
            
            inner_radius = donuts_inner_radius                    
            outer_radius = donuts_inner_radius * radius_multiplier 
            
            print(f"    Donuts radii: inner={inner_radius:.4f}, outer={outer_radius:.4f} (×{radius_multiplier})")
            
            inner_mask_dense = geodesic_distances < inner_radius
            outer_mask_dense = (geodesic_distances >= inner_radius) & (geodesic_distances < outer_radius)
            
            inner_indices_dense = np.where(inner_mask_dense)[0]
            outer_indices_dense = np.where(outer_mask_dense)[0]
            
            print(f"    Dense donuts: inner={len(inner_indices_dense)} points, outer={len(outer_indices_dense)} points")

            is_connected = False
            component_count = 0
            largest_ratio = 0.0
            
            if len(outer_indices_dense) > 0:
                delta_coords_dense = gaussians.xyz[outer_indices_dense]
                print(f"    Analyzing connectivity on {len(delta_coords_dense)} dense delta points...")

                component_count, largest_ratio, labels, sizes = get_connected_component_ratio(
                    delta_coords_dense, 
                    radius=radius,  
                    ignore_size=0  
                )
                is_connected = (component_count == 1) or (component_count <= 2 and largest_ratio > 0.8)
            
            structure_type = "Leaf" if is_connected else "Stem"
            print(f"Step {i}: Path {path_idx} | Inner={len(inner_indices_dense)}, Delta={len(outer_indices_dense)}, Components={component_count}, LargestRatio={largest_ratio:.3f}, Connected={is_connected} | Type: {structure_type}")
            
            if not is_connected:  # stem-like (disconnected)
                print(f"Found first stem point: Step {i}, Path {path_idx}")
                return potential_path[i]
            
        print(f"No stem point found, selecting last point: {potential_path[-1]}")
        return potential_path[-1]
    elif method == 'geodesic_tip':
        tip_geodesic_distances = dense_solver.compute_distance(true_tip)  
        target_threshold = ratio_threshold
        print(f"\nSelection Strategy: Looking for first L/P ratio < {target_threshold}")
        
        closest_ratio = float('inf')
        closest_idx = 0
        closest_diff = float('inf')
        
        for i in range(len(potential_path) - 1):
            current_path_idx = potential_path[i]
            next_path_idx = potential_path[i + 1]
            
            current_dist = tip_geodesic_distances[current_path_idx]
            next_dist = tip_geodesic_distances[next_path_idx]
            
            current_mask = tip_geodesic_distances <= current_dist
            next_mask = tip_geodesic_distances <= next_dist
            delta_mask = next_mask & (~current_mask)
            delta_indices = np.where(delta_mask)[0]
            
            if len(delta_indices) < 3:  
                continue
                
            delta_coords = gaussians.xyz[delta_indices]
            pca = PCA(n_components=3)
            pca.fit(delta_coords)
            
            eigenvalues_ratio = pca.explained_variance_ratio_
            linearity = eigenvalues_ratio[0]
            planarity = eigenvalues_ratio[1]
            lp_ratio = linearity / planarity if planarity > 1e-10 else float('inf')
            
            print(f"  Step {i}: L/P ratio = {lp_ratio:.6f}")
            
            if lp_ratio < target_threshold:
                print(f"  Found first ratio < {target_threshold}: Step {i}, L/P = {lp_ratio:.6f}")
                return potential_path[i]
            
            diff = abs(lp_ratio - target_threshold)
            if diff < closest_diff:
                closest_diff = diff
                closest_ratio = lp_ratio
                closest_idx = i
        
        print(f"  No ratio < {target_threshold} found, selecting closest to {target_threshold}:")
        print(f"  Step {closest_idx}, L/P = {closest_ratio:.6f} (diff = {closest_diff:.6f})")
        return potential_path[closest_idx]
    
def find_base_idx_by_euclidean_graph(cluster_info_item, 
                                gaussians, 
                                nbrs_reusable,
                                temperature_field, 
                                temperature_percentile=75, 
                                max_iterations_factor=2,
                                path_distance_factor=1.5,
                                protection_period_ratio=0.25):
    
    tips = cluster_info_item['tips']
    original_path = cluster_info_item['potential_base_idx']
    cluster_type = cluster_info_item['type']
    
    if len(tips) == 0 or len(original_path) == 0:
        print("No tips or original path found, cannot generate virtual path")
        return
    
    if cluster_type == 'multi_tip' and len(tips) > 1:
        first_path_point = original_path[0] if len(original_path) > 0 else tips[0]
        print(f"Multi-tip cluster detected with {len(tips)} tips, returning first path point: {first_path_point}")
        return first_path_point
    
    tip_idx = tips[0]
    tip_coord = gaussians.xyz[tip_idx]
    first_step_radius = np.linalg.norm(tip_coord - gaussians.xyz[original_path[0]])

    # find neighboring in radius 
    nbrs = NearestNeighbors(radius=first_step_radius, algorithm='ball_tree').fit(gaussians.xyz)     
    tip_to_neighbor_distances, neighbor_indices = nbrs.radius_neighbors([tip_coord], return_distance=True)
    tip_to_neighbor_distances = tip_to_neighbor_distances[0]
    neighbor_indices = neighbor_indices[0]
    neighbor_coords = gaussians.xyz[neighbor_indices]
    
    if len(original_path) > 1:
        second_path_coord = gaussians.xyz[original_path[1]]
    else:
        second_path_coord = tip_coord  
    next_path_to_neighbor_distances = np.linalg.norm(neighbor_coords - second_path_coord, axis=1)
    
    neighbor_temps = temperature_field[neighbor_indices]
    temp_threshold = np.percentile(neighbor_temps, temperature_percentile)
    high_temp_mask = neighbor_temps > temp_threshold
    
    if np.sum(high_temp_mask) == 0:
        print("Warning: No neighbors above temperature threshold, using all neighbors")
        high_temp_mask = np.ones(len(neighbor_indices), dtype=bool)
    
    total_distances = tip_to_neighbor_distances + (next_path_to_neighbor_distances * path_distance_factor)
    filtered_distances = np.where(high_temp_mask, total_distances, -np.inf)
    farthest_point_idx = np.argmax(filtered_distances)

    selected_point_idx = neighbor_indices[farthest_point_idx]
    
    virtual_path = [tip_idx, selected_point_idx]  
    current_tip = selected_point_idx  
    
    max_iterations = int(len(original_path) * max_iterations_factor)
    protection_period = max(1, int(max_iterations * protection_period_ratio))  
    
    connected_path_idx = None
    
    for iteration in range(1, max_iterations):  
        
        current_tip_coord = gaussians.xyz[current_tip]
        
        if iteration + 1 < len(original_path):
            next_path_idx = original_path[iteration + 1]
            next_path_coord = gaussians.xyz[next_path_idx]
        else:
            next_path_idx = original_path[-1]
            next_path_coord = gaussians.xyz[next_path_idx]
        
        new_tip_distances, new_neighbor_indices = nbrs_reusable.radius_neighbors([current_tip_coord], return_distance=True)
        new_tip_distances = new_tip_distances[0]
        new_neighbor_indices = new_neighbor_indices[0]
        
        if len(new_neighbor_indices) == 0:
            print("No neighbors found within radius, stopping iteration")
            break
        
        if iteration >= protection_period:
            path_points_in_radius = []
            for path_idx in original_path:
                if path_idx in new_neighbor_indices:
                    path_points_in_radius.append(path_idx)
            
            if len(path_points_in_radius) > 0:
                
                path_distances = []
                for path_idx in path_points_in_radius:
                    distance = np.linalg.norm(current_tip_coord - gaussians.xyz[path_idx])
                    path_distances.append((path_idx, distance))
                
                path_distances.sort(key=lambda x: x[1])
                
                if len(path_points_in_radius) == 1:
                    selected_path_idx = path_distances[0][0]
                elif len(path_points_in_radius) == 2:
                    selected_path_idx = path_distances[-1][0]  
                else:
                    selected_path_idx = path_distances[-2][0]  
                
                virtual_path.append(selected_path_idx)
                connected_path_idx = selected_path_idx
                break
 
        new_neighbor_coords = gaussians.xyz[new_neighbor_indices]
        
        next_path_distances = np.linalg.norm(new_neighbor_coords - next_path_coord, axis=1)
        
        virtual_path_distances = np.zeros(len(new_neighbor_indices))
        for virtual_idx in virtual_path:
            virtual_coord = gaussians.xyz[virtual_idx]
            virtual_path_distances += np.linalg.norm(new_neighbor_coords - virtual_coord, axis=1)
        
        new_neighbor_temps = temperature_field[new_neighbor_indices]
        new_temp_threshold = np.percentile(new_neighbor_temps, temperature_percentile)
        new_high_temp_mask = new_neighbor_temps > new_temp_threshold

        if np.sum(new_high_temp_mask) == 0:
            print("  Warning: No neighbors above temperature threshold, using all neighbors")
            new_high_temp_mask = np.ones(len(new_neighbor_indices), dtype=bool)
        
        total_new_distances = new_tip_distances + (next_path_distances * path_distance_factor) + virtual_path_distances
        filtered_new_distances = np.where(new_high_temp_mask, total_new_distances, -np.inf)
        new_farthest_idx = np.argmax(filtered_new_distances)
        
        new_virtual_point_idx = int(new_neighbor_indices[new_farthest_idx])
        
        virtual_path.append(new_virtual_point_idx)
        
        current_tip = new_virtual_point_idx
    
    if connected_path_idx is not None:
        return connected_path_idx
    else:
        return tip_idx

def find_base_idx_by_geodesic_tip_graph(cluster_info_item, 
                                        gaussians,
                                        dense_solver,
                                        temperature_field,
                                        min_distance_threshold=0.05,
                                        tolerance_percentage=0.005,
                                        last_virtual_path_distance_factor=3,
                                        protection_period_ratio=0.25,
                                        debug_vis=False):
    
    tips = cluster_info_item['tips']
    original_path = cluster_info_item['potential_base_idx']

    final_tip = cluster_info_item['final_tip']
    lca = cluster_info_item['lca']

    # find idx of lca in original_path
    if lca in original_path:
        lca_idx_in_path = original_path.index(lca)
    trimmed_path = original_path[:lca_idx_in_path+1] if lca in original_path else original_path
    original_path = trimmed_path

    if len(tips) == 0 or len(original_path) == 0:
        return
    
    tip_idx = int(final_tip)  
    tip_geodesic = dense_solver.compute_distance(tip_idx)
    
    virtual_path = [tip_idx]
    selected_points = []
    selected_dists = []
    connected_path_point = None  
    
    protection_period = max(1, int(len(original_path) * protection_period_ratio))
    prev_selected_idx = tip_idx
    for i, path_idx in enumerate(original_path):
        path_geodesic_dist = tip_geodesic[path_idx]
        
        tolerance = path_geodesic_dist * tolerance_percentage
        mask = np.abs(tip_geodesic - path_geodesic_dist) < tolerance
        candidate_indices = np.where(mask)[0]

        temp_at_path = temperature_field[path_idx]
        temp_tolerance = temp_at_path * tolerance_percentage
        temp_mask = np.abs(temperature_field - temp_at_path) < temp_tolerance 
        temp_candidate_indices = np.where(temp_mask)[0]

        if len(candidate_indices) == 0:
            continue

        if len(temp_candidate_indices) > 0:

            candidate_set = set(candidate_indices)
            temp_candidate_set = set(temp_candidate_indices)
            intersect_indices = list(candidate_set.intersection(temp_candidate_set))
            if len(intersect_indices) > 0:
                candidate_indices = np.array(intersect_indices)
            
        path_coord = gaussians.xyz[path_idx]
        tip_coord = gaussians.xyz[tip_idx]
        candidate_coords = gaussians.xyz[candidate_indices]
        
        distances_from_path = np.linalg.norm(candidate_coords - path_coord, axis=1)
        distances_from_tip = np.linalg.norm(candidate_coords - tip_coord, axis=1)
        distances = distances_from_path + distances_from_tip

        best_idx = np.argmax(distances)
        selected_point_idx = candidate_indices[best_idx]
        selected_distance = distances_from_path[best_idx]

        selected_points.append(int(selected_point_idx))
        virtual_path.append(int(selected_point_idx))
        selected_dists.append(selected_distance)
        
        if i >= protection_period and selected_distance < min_distance_threshold:
            connected_path_point = int(path_idx)  
            break
            
        if i >= protection_period and int(selected_point_idx) in original_path:
            connected_path_point = int(selected_point_idx)
            break

    virtual_path = [int(idx) for idx in virtual_path]
    selected_points = [int(idx) for idx in selected_points]
    

    
    final_base_idx = None
    if connected_path_point is not None:
        final_base_idx = connected_path_point
    elif len(selected_points) > 0:
        min_idx_after_protection = np.argmin(selected_dists[protection_period:]) + protection_period
        final_base_idx = selected_points[min_idx_after_protection]
    else:
        final_base_idx = tip_idx
    
    return final_base_idx   