import numpy as np
import fpsample

def calculate_angle_between_vectors(vec1, vec2, in_degrees=True):
    norm_v1 = np.linalg.norm(vec1)
    norm_v2 = np.linalg.norm(vec2)

    if norm_v1 == 0 or norm_v2 == 0:
        print("Warning: One or both vectors are zero. Angle is undefined.")
        return np.nan # Angle is undefined if one vector is zero

    # Normalize vectors
    unit_vec1 = vec1 / norm_v1
    unit_vec2 = vec2 / norm_v2
    
    # Calculate dot product
    dot_product = np.dot(unit_vec1, unit_vec2)
    
    # Clip the dot_product to the range [-1.0, 1.0] to avoid numerical errors with arccos
    dot_product_clipped = np.clip(dot_product, -1.0, 1.0)
    
    # Calculate angle in radians
    angle_rad = np.arccos(dot_product_clipped)
    
    if in_degrees:
        return np.degrees(angle_rad)
    else:
        return angle_rad

def perform_pca(points):

    if points.shape[0] < points.shape[1]: # Need more points than dimensions for meaningful covariance
        # Or handle as an error/warning
        print(f"Warning: PCA on {points.shape[0]} points in {points.shape[1]}D might be unstable.")

    mean = np.mean(points, axis=0)
    centered_points = points - mean
    
    # np.cov expects variables as rows, observations as columns.
    # If points is (N,3), centered_points.T is (3,N).
    # The covariance matrix will be (3,3).
    covariance_matrix = np.cov(centered_points.T)
    
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # Sort eigenvalues and corresponding eigenvectors in descending order
    idx_sort = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx_sort]
    sorted_eigenvectors = eigenvectors[:, idx_sort]
    
    return mean, sorted_eigenvalues, sorted_eigenvectors

def get_min_max_projection_points_on_axis(points_data, mean_vec, axis_vector):
    centered_points = points_data - mean_vec
    projections = centered_points @ axis_vector # Dot product for each point
    
    idx_min = np.argmin(projections)
    idx_max = np.argmax(projections)
    min_point = points_data[idx_min]
    max_point = points_data[idx_max]

    min_to_mean = min_point - mean_vec
    max_to_mean = max_point - mean_vec
    # check the porjection direction
    # min_to_mean proj to eigenvector should be negative
    # max_to_mean proj to eigenvector should be positive

    proj_min_to_mean_to_axis = np.dot(min_to_mean, axis_vector)
    proj_max_to_mean_to_axis = np.dot(max_to_mean, axis_vector)

    if proj_min_to_mean_to_axis > 0 or proj_max_to_mean_to_axis < 0:
        # If the projections are not in the expected direction, flip the points
        min_point, max_point = max_point, min_point
    
    return min_point, max_point

def orient_pca_axes(mean_point,
                    original_eigenvectors,
                    apex_point,
                    max_horizontal_point):
    oriented_eigvecs = np.zeros_like(original_eigenvectors)

    vec_mean_to_apex = apex_point - mean_point
    norm_vec_mean_to_apex = np.linalg.norm(vec_mean_to_apex)

    if norm_vec_mean_to_apex < 1e-9: 
        p0_new = original_eigenvectors[:, 0].copy()
    else:
        p0_new = vec_mean_to_apex / norm_vec_mean_to_apex
    
    oriented_eigvecs[:, 0] = p0_new

    vec_mean_to_max_h = max_horizontal_point - mean_point
    norm_vec_mean_to_max_h = np.linalg.norm(vec_mean_to_max_h)

    p1_new = np.zeros(3) 

    if norm_vec_mean_to_max_h < 1e-9:
        p1_candidate_fallback = original_eigenvectors[:, 1].copy()
        p1_projected = p1_candidate_fallback - np.dot(p1_candidate_fallback, p0_new) * p0_new
        norm_p1_projected = np.linalg.norm(p1_projected)

        if norm_p1_projected < 1e-9:
            p2_candidate_fallback = original_eigenvectors[:, 2].copy()
            p1_new = np.cross(p2_candidate_fallback, p0_new) 
            if np.linalg.norm(p1_new) < 1e-9: 
                
                if np.abs(p0_new[0]) < 0.9: 
                    temp_axis = np.array([1.0, 0.0, 0.0])
                else: 
                    temp_axis = np.array([0.0, 1.0, 0.0])
                p1_new = np.cross(p0_new, temp_axis)
            p1_new /= np.linalg.norm(p1_new) 
        else:
            p1_new = p1_projected / norm_p1_projected
            p2_original_ref = original_eigenvectors[:, 2]
            if np.dot(np.cross(p0_new, p1_new), p2_original_ref) < 0:
                p1_new *= -1
    else:
        p1_projected_from_max_h = vec_mean_to_max_h - np.dot(vec_mean_to_max_h, p0_new) * p0_new
        norm_p1_projected_from_max_h = np.linalg.norm(p1_projected_from_max_h)

        if norm_p1_projected_from_max_h < 1e-9:
            p1_candidate_fallback = original_eigenvectors[:, 1].copy()
            p1_projected = p1_candidate_fallback - np.dot(p1_candidate_fallback, p0_new) * p0_new
            norm_p1_projected = np.linalg.norm(p1_projected)
            if norm_p1_projected < 1e-9:
                p2_candidate_fallback = original_eigenvectors[:, 2].copy()
                p1_new = np.cross(p2_candidate_fallback, p0_new)
                if np.linalg.norm(p1_new) < 1e-9:
                    if np.abs(p0_new[0]) < 0.9: temp_axis = np.array([1.0, 0.0, 0.0])
                    else: temp_axis = np.array([0.0, 1.0, 0.0])
                    p1_new = np.cross(p0_new, temp_axis)
                p1_new /= np.linalg.norm(p1_new)
            else:
                p1_new = p1_projected / norm_p1_projected
                p2_original_ref = original_eigenvectors[:, 2]
                if np.dot(np.cross(p0_new, p1_new), p2_original_ref) < 0:
                    p1_new *= -1
        else:
            p1_new = p1_projected_from_max_h / norm_p1_projected_from_max_h
            if np.dot(p1_new, vec_mean_to_max_h) < 0: 
                 p1_new *= -1

    oriented_eigvecs[:, 1] = p1_new
    oriented_eigvecs[:, 2] = np.cross(p0_new, p1_new)
    norm_p2 = np.linalg.norm(oriented_eigvecs[:, 2])
    if norm_p2 > 1e-9:
        oriented_eigvecs[:, 2] /= norm_p2

    # Check the P2 should be same direction as z [0, 0, 1] unless revert P1
    dot_p2_z = np.dot(oriented_eigvecs[:, 2], np.array([0, 0, 1]))
    
    if dot_p2_z < 0: # If P2 direction is inconsistent with Z axis
        oriented_eigvecs[:, 1] *= -1
        oriented_eigvecs[:, 2] *= -1

    return oriented_eigvecs

def perform_oriented_pca(
    points_full,
    fps_n_points,
    fps_h_param,
    stem_connected_point_idx=None,
    label=""
):
    if points_full is None or len(points_full) == 0:
        print(f"Error ({label}): Input point cloud is empty.")
        return None, None, None, None, None, None

    # 1. FPS downsampling
    if len(points_full) < fps_n_points:
        print(f"Warning ({label}): Full point cloud has {len(points_full)} points, fewer than FPS requested {fps_n_points}. Using all points for PCA.")
        points_for_pca_sampling = points_full
    else:
        try:
            sampled_indices = fpsample.bucket_fps_kdline_sampling(points_full, fps_n_points, h=fps_h_param, start_idx=0)
            points_for_pca_sampling = points_full[sampled_indices]
        except Exception as e:
            print(f"Error ({label}): FPS sampling failed - {e}. Will attempt PCA with all points.")
            points_for_pca_sampling = points_full

    # 2. PCA analysis (calling global/imported function)
    try:
        mean, eigvals, initial_eigvecs = perform_pca(points_for_pca_sampling)
    except NameError:
        print(f"Error ({label}): Function 'perform_pca' is not defined.")
        raise
    except Exception as e:
        print(f"Error ({label}): PCA analysis failed - {e}")
        return None, None, None, None, None, points_for_pca_sampling

    # 3. Get projection extreme points (calling global/imported function)
    try:
        min_pt_v, max_pt_v = get_min_max_projection_points_on_axis(
            points_for_pca_sampling, mean, initial_eigvecs[:, 0]
        )
        min_pt_h, max_pt_h = get_min_max_projection_points_on_axis(
            points_for_pca_sampling, mean, initial_eigvecs[:, 1]
        )

    except NameError:
        print(f"Error ({label}): Function 'get_min_max_projection_points_on_axis' is not defined.")
        raise
    except Exception as e:
        print(f"Error ({label}): Getting projection extreme points failed - {e}")
        return mean, eigvals, initial_eigvecs, None, None, points_for_pca_sampling

    # 4. Determine apex point - if stem_connected_point provided, use it to determine correct apex direction
    apex_point = None
    if stem_connected_point_idx is not None:
        # Validate index
        if 0 <= stem_connected_point_idx < len(points_full):
            stem_point = points_full[stem_connected_point_idx]
            # Find two extreme points along main PCA axis, then choose the one farther from stem
            # Calculate distances from extreme points to stem point
            dist_min_pt_v = np.linalg.norm(min_pt_v - stem_point)
            dist_max_pt_v = np.linalg.norm(max_pt_v - stem_point)
            
            # Choose the point farther from stem as apex point
            if dist_min_pt_v > dist_max_pt_v:
                apex_point = min_pt_v
            else:
                apex_point = max_pt_v
        else:
            print(f"Warning ({label}): Provided stem connection point index {stem_connected_point_idx} out of range [0, {len(points_full)-1}]. Using heuristic method.")
    
    if apex_point is None:
        vec_minH_to_maxV = max_pt_v - min_pt_h
        vec_maxH_to_maxV = max_pt_v - max_pt_h
        vec_minV_to_minH = min_pt_h - min_pt_v
        vec_minV_to_maxH = max_pt_h - min_pt_v
        
        primary_pca_axis = initial_eigvecs[:, 0]

        try:
            # Ensure calculate_angle_between_vectors is available in this scope
            angle_minH_maxV_vs_PCA1 = calculate_angle_between_vectors(vec_minH_to_maxV, primary_pca_axis)
            angle_maxH_maxV_vs_PCA1 = calculate_angle_between_vectors(vec_maxH_to_maxV, primary_pca_axis)
            
            angle1 = np.nan
            if not (np.isnan(angle_minH_maxV_vs_PCA1) or np.isnan(angle_maxH_maxV_vs_PCA1)):
                angle1 = angle_minH_maxV_vs_PCA1 + angle_maxH_maxV_vs_PCA1

            angle_minV_to_minH_vs_PCA1 = calculate_angle_between_vectors(vec_minV_to_minH, primary_pca_axis)
            angle_minV_to_maxH_vs_PCA1 = calculate_angle_between_vectors(vec_minV_to_maxH, primary_pca_axis)

            angle2 = np.nan
            if not (np.isnan(angle_minV_to_minH_vs_PCA1) or np.isnan(angle_minV_to_maxH_vs_PCA1)):
                angle2 = angle_minV_to_minH_vs_PCA1 + angle_minV_to_maxH_vs_PCA1
        except NameError:
            print(f"Error ({label}): Function 'calculate_angle_between_vectors' is not defined.")
            raise
        except Exception as e: # Catch other possible errors from calculate_angle_between_vectors
            print(f"Error ({label}): Error calculating angles - {e}")
            return mean, eigvals, initial_eigvecs, None, max_pt_h, points_for_pca_sampling

        # Determine apex using heuristic rules
        if np.isnan(angle1) and np.isnan(angle2):
            print(f"Warning ({label}): Both heuristic angles angle1 and angle2 are NaN. Defaulting apex to max_pt_v.")
            apex_point = max_pt_v
        elif np.isnan(angle1):
            print(f"Warning ({label}): Heuristic angle angle1 is NaN. Defaulting apex to max_pt_v.")
            apex_point = max_pt_v
        elif np.isnan(angle2):
            print(f"Warning ({label}): Heuristic angle angle2 is NaN. Defaulting apex to max_pt_v.")
            apex_point = max_pt_v
        elif angle1 > angle2:
            apex_point = min_pt_v
        else:
            apex_point = max_pt_v
        
        distances_to_selected_apex = np.linalg.norm(points_full - apex_point, axis=1)
        closest_apex_idx = np.argmin(distances_to_selected_apex)
        apex_point = points_full[closest_apex_idx]
        
    try:
        oriented_eigvecs = orient_pca_axes(mean, initial_eigvecs, apex_point, max_pt_h)
    except Exception as e:
        print(f"Error ({label}): Orienting PCA axes failed - {e}. Returning unoriented axes.")
        return mean, eigvals, initial_eigvecs, apex_point, max_pt_h, points_for_pca_sampling

    return mean, eigvals, oriented_eigvecs, apex_point, max_pt_h, points_for_pca_sampling

def align_to_xy_plane(points, stem_labels=None):
    # Determine stem connection point from stem labels if provided
    stem_connected_point_idx = None
    if stem_labels is not None:
        assert np.sum(stem_labels) > 0, "No stem points found in stem_labels"
        # Find stem points (where label == 1)
        stem_indices = np.where(stem_labels == 1)[0]
        if len(stem_indices) > 0:
            # Use the centroid of stem points, or just pick the first one
            # For simplicity, we'll use the first stem point as reference
            stem_connected_point_idx = stem_indices[0]
        else:
            print("Warning: No stem points found in stem_labels (no points with value 1)")
    
    # Perform oriented PCA to get the leaf's principal axes
    mean, eigvals, oriented_eigvecs, apex_point, max_pt_h, points_for_pca_sampling = perform_oriented_pca(
        points,
        fps_n_points=min(32, len(points)),
        fps_h_param=3,
        stem_connected_point_idx=stem_connected_point_idx,
        label="xy_plane_alignment"
    )
    
    # Verify that eigenvalues are sorted in descending order (largest to smallest)
    assert eigvals[0] >= eigvals[1] >= eigvals[2], f"Eigenvalues not properly sorted: {eigvals}"
    
    # Verify that eigenvectors are orthogonal
    for i in range(3):
        for j in range(i+1, 3):
            dot_product = np.abs(np.dot(oriented_eigvecs[:, i], oriented_eigvecs[:, j]))
            assert dot_product < 1e-6, f"Eigenvectors {i} and {j} are not orthogonal: dot product = {dot_product}"
    
    # Define target coordinate system:
    # X-axis should align with 1st PCA component (largest eigenvalue)
    # Y-axis should align with 2nd PCA component (middle eigenvalue)  
    # Z-axis should align with 3rd PCA component (smallest eigenvalue)
    target_axes = np.array([
        [1, 0, 0],  # Target X-axis 
        [0, 1, 0],  # Target Y-axis
        [0, 0, 1]   # Target Z-axis
    ]).T  # Shape: (3, 3), columns are target axes
    
    source_axes = oriented_eigvecs  # Shape: (3, 3), columns are source PCA axes
    
    # Calculate rotation matrix using orthogonal Procrustes problem
    # We want to find R such that R @ source_axes ≈ target_axes
    # This is equivalent to: R = target_axes @ source_axes.T
    # Since both source_axes and target_axes are orthonormal matrices
    R = target_axes @ source_axes.T
    
    # Verify that R is a valid rotation matrix
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6), f"Rotation matrix determinant is not 1: {np.linalg.det(R)}"
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), "Rotation matrix is not orthogonal"
    
    # Verify the alignment
    rotated_axes = R @ source_axes
    for i in range(3):
        alignment_error = np.linalg.norm(rotated_axes[:, i] - target_axes[:, i])
        # print(f"- PCA axis {i+1} alignment error: {alignment_error:.2e}")
        assert alignment_error < 1e-6, f"Axis {i} alignment failed: error = {alignment_error}"
    
    # Create 4x4 transformation matrix
    # Center the points at origin after rotation
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ mean  # This centers the rotated points at origin
    
    # Apply transformation
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    aligned_points = (T @ points_homogeneous.T).T[:, :3]
    
    # Verify final alignment by checking the PCA of aligned points
    aligned_mean, aligned_eigvals, aligned_eigvecs = perform_pca(aligned_points)
    
    # Prepare PCA info for return
    pca_info = {
        'mean': mean,
        'eigenvalues': eigvals,
        'eigenvectors': oriented_eigvecs,
        'pca_axes': oriented_eigvecs,  # Add alias for compatibility
        'apex_point': apex_point,
        'max_horizontal_point': max_pt_h,
        'rotation_matrix': R,
        'transformation_matrix': T,
        'sampled_points': points_for_pca_sampling,
        'aligned_eigenvalues': aligned_eigvals,
        'aligned_eigenvectors': aligned_eigvecs
    }
    
    return T, aligned_points, pca_info

def align_to_xy_plane_with_tips(points, tip_point, base_point, root_point=None):
    tip_point  = np.array(tip_point).flatten()
    base_point = np.array(base_point).flatten()

    mean, eigvals, initial_eigvecs = perform_pca(points)

    oriented_eigvecs = np.zeros_like(initial_eigvecs)

    p2_raw = initial_eigvecs[:, 2]
    p2 = p2_raw if np.dot(p2_raw, [0, 0, 1]) > 0 else -p2_raw

    v_bt = tip_point - base_point
    if np.linalg.norm(v_bt) < 1e-9:
        v_bt = initial_eigvecs[:, 0]
    p0_proj = v_bt - np.dot(v_bt, p2) * p2
    p0n = np.linalg.norm(p0_proj)
    if p0n < 1e-9:
        largest = initial_eigvecs[:, 0]
        p0_proj = largest - np.dot(largest, p2) * p2
        p0n = np.linalg.norm(p0_proj)
    p0 = p0_proj / p0n

    p1 = np.cross(p2, p0); p1 /= np.linalg.norm(p1)
    base_x = np.dot(base_point - mean, p0)
    tip_x  = np.dot(tip_point  - mean, p0)
    if base_x > tip_x:
        p0 = -p0
        p1 = np.cross(p2, p0); p1 /= np.linalg.norm(p1)

    if np.dot(np.cross(p0, p1), p2) < 0:
        p1 = -p1

    source_axes = np.column_stack([p0, p1, p2])        # 3x3
    R_tmp = np.eye(3) @ source_axes.T                  # = source_axes^T
    T_tmp = np.eye(4); T_tmp[:3,:3] = R_tmp; T_tmp[:3,3] = -R_tmp @ mean

    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
    aligned_tmp = (T_tmp @ pts_h.T).T[:, :3]           # (N,3)

    oriented_eigvecs[:, 0] = p0
    oriented_eigvecs[:, 1] = p1
    oriented_eigvecs[:, 2] = p2

    target_axes = np.eye(3)  # X, Y, Z
    source_axes = oriented_eigvecs
    R = target_axes @ source_axes.T

    detR = np.linalg.det(R)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), "R not orthogonal"
    assert np.isclose(abs(detR), 1.0, atol=1e-6), f"|det(R)| != 1: {detR}"

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = -R @ mean

    # Apply the final transform to root_point if provided
    # If transformed root has z >= 0, mirror across XY (flip Z only in target space).
    # Start your implementation here, just get the corrected T matrix => T
    if root_point is not None:
        root_point_homo = np.hstack([root_point, 1.0])  # (4,)
        transformed_root = T @ root_point_homo
        transformed_root_3d = transformed_root[:3]  # (3,)
        z_root = transformed_root_3d[2]
        if z_root >= 0.0:
            # Mirror on XY plane in target space (det = -1)
            Rx_pi = np.diag([1.0, -1.0, -1.0])
            R = Rx_pi @ R
            # Rebuild T with corrected R
            T[:3, :3] = R
            T[:3, 3]  = -R @ mean

    aligned_points = (T @ pts_h.T).T[:, :3]

    pca_info = {
        'mean': mean,
        'eigenvalues': eigvals,
        'eigenvectors': oriented_eigvecs,
        'pca_axes': oriented_eigvecs,
        'apex_point': tip_point,
        'max_horizontal_point': base_point,
        'rotation_matrix': R,
        'transformation_matrix': T,
        'tip_point': tip_point,
        'base_point': base_point,
    }
    return T, aligned_points, pca_info
