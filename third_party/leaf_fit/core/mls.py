import numpy as np
from scipy.spatial import cKDTree

def mls_surface_reconstruction(points, search_radius, polynomial_order=2, expand=1.0,
                               orient_k=12, orient_to_centroid=False):
    """
    Smooth and resample point cloud using Moving Least Squares (MLS) method with normal orientation.

    Args:
        points (np.ndarray): Input point cloud (N, 3)
        search_radius (float): Radius for neighborhood search
        polynomial_order (int): 1 or 2
        expand (float): Exponent for global scaling of reconstructed points (scale_factor**expand)
        orient_k (int): Number of k-nearest neighbors for normal orientation
        orient_to_centroid (bool): Whether to orient normals towards/away from global centroid after consistency (default False)

    Returns:
        new_points (N,3), new_normals (N,3), inlier_indices (M,)
    """
    assert points.ndim == 2 and points.shape[1] == 3
    N = len(points)
    eps = 1e-12

    # 1) KD-Tree
    kdtree = cKDTree(points)

    new_points = np.zeros_like(points, dtype=np.float64)
    new_normals = np.zeros_like(points, dtype=np.float64)
    deviations  = np.zeros(N, dtype=np.float64)

    # Minimum number of neighbors
    min_neighbors = 6 if polynomial_order == 2 else 3

    # 2) Iterate through each point
    for i, p in enumerate(points):
        # Neighborhood: radius search
        neighbor_indices = kdtree.query_ball_point(p, r=search_radius)

        if len(neighbor_indices) < min_neighbors:
            # Insufficient neighbors: keep position, set normal to zero (will be skipped in orientation)
            new_points[i]  = p
            new_normals[i] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            deviations[i]  = 0.0
            continue

        neighbors = points[neighbor_indices]

        # a) Gaussian weights
        diff = neighbors - p  # More stable using p as reference (can also use centroid)
        distances_sq = np.sum(diff**2, axis=1)
        weights = np.exp(-distances_sq / (search_radius**2))
        W = np.sqrt(weights + eps)  # For stability

        # b) Calculate weighted centroid (recommended to use weighted centroid)
        centroid = np.average(neighbors, axis=0, weights=weights)

        # c) PCA for local basis (use PCA normal first, then refine with fitting gradient)
        centered = neighbors - centroid
        # Weighted covariance: C = X^T W^2 X
        cov = (centered.T * weights) @ centered
        # eigh returns eigenvalues/eigenvectors in ascending order
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Smallest eigenvector as initial normal
        n = eigenvectors[:, 0]
        # Use largest eigenvector as initial u
        u = eigenvectors[:, 2]

        # Right-hand system correction: v = n x u; then correct u = v x n
        v = np.cross(n, u)
        nv = np.linalg.norm(v) + eps
        v /= nv
        u = np.cross(v, n)
        nu = np.linalg.norm(u) + eps
        u /= nu
        nn = np.linalg.norm(n) + eps
        n /= nn

        # d) Project neighborhood points to (u,v,n) local coordinate system
        u_coords = (centered @ u)
        v_coords = (centered @ v)
        w_coords = (centered @ n)

        # e) Weighted least squares fitting: w = f(u,v)
        if polynomial_order == 1:
            # [1, u, v]
            A = np.vstack([np.ones_like(u_coords), u_coords, v_coords]).T
        elif polynomial_order == 2:
            # [1, u, v, u^2, v^2, uv]
            A = np.vstack([
                np.ones_like(u_coords), u_coords, v_coords,
                u_coords**2, v_coords**2, u_coords * v_coords
            ]).T
        else:
            raise ValueError("polynomial_order only supports 1 or 2")

        # Solve (W*A) c = W*w
        Aw = A * W[:, None]
        ww = w_coords * W
        coeffs, _, _, _ = np.linalg.lstsq(Aw, ww, rcond=None)

        # f) p on fitted surface corresponds to (u,v)=(0,0), w=c0
        projected_height = float(coeffs[0])
        new_point = centroid + projected_height * n  # Project back using height along local plane normal

        # g) Refine normal direction using fitting gradient (more stable: normal at origin of surface)
        # Gradient at (0,0) for first/second order depends only on linear terms
        du = float(coeffs[1]) if len(coeffs) > 1 else 0.0
        dv = float(coeffs[2]) if len(coeffs) > 2 else 0.0
        n_fit = n - du * u - dv * v
        n_fit_norm = np.linalg.norm(n_fit)
        if n_fit_norm > eps:
            n_fit /= n_fit_norm
        else:
            n_fit = n  # Fallback for degenerate case

        deviations[i]  = np.linalg.norm(p - new_point)
        new_points[i]  = new_point
        new_normals[i] = n_fit

    # 3) Compute inliers statistics
    mean_dev = float(np.mean(deviations))
    std_dev  = float(np.std(deviations))
    threshold = mean_dev + 1.75 * std_dev
    inlier_mask = deviations <= threshold
    inlier_indices = np.where(inlier_mask)[0]

    # 4) Normal orientation consistency (kNN graph propagation)
    def orient_normals_consistently(pts, normals, k=12):
        tree = cKDTree(pts)
        Np = len(pts)

        # Normalize and mark valid normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        valid = (norms[:, 0] > 1e-12)
        normals[valid] = normals[valid] / (norms[valid] + eps)

        # Select seed: point with valid normal farthest from global centroid
        cen = pts.mean(axis=0)
        seed_candidates = np.where(valid)[0]
        if len(seed_candidates) == 0:
            return normals  # Return directly if all empty

        d = np.linalg.norm(pts[seed_candidates] - cen, axis=1)
        seed = seed_candidates[int(np.argmax(d))]

        visited = np.zeros(Np, dtype=bool)
        from collections import deque
        q = deque([seed])
        visited[seed] = True

        while q:
            i = q.popleft()
            # Include self, take k+1
            _, idxs = tree.query(pts[i], k=min(k + 1, Np))
            # If returned as scalar (for very small point counts), wrap it
            if np.isscalar(idxs):
                idxs = np.array([idxs])
            # Remove self
            idxs = [j for j in idxs if j != i]

            for j in idxs:
                if not valid[j]:
                    continue
                if np.dot(normals[i], normals[j]) < 0:
                    normals[j] = -normals[j]
                if not visited[j]:
                    visited[j] = True
                    q.append(j)
        return normals

    new_normals = orient_normals_consistently(new_points, new_normals, k=orient_k)

    # 5) (Optional) Finally orient all normals based on global centroid: towards or away
    if orient_to_centroid:
        global_centroid = np.mean(new_points, axis=0)
        for i in range(N):
            n = new_normals[i]
            if np.linalg.norm(n) < eps:
                continue
            p_to_c = global_centroid - new_points[i]
            # Consistent with original logic: flip if angle > 90°
            if np.dot(n, p_to_c) < 0:
                new_normals[i] = -n

    # 6) Position scaling
    org_min = np.min(points, axis=0)
    org_max = np.max(points, axis=0)
    new_min = np.min(new_points, axis=0)
    new_max = np.max(new_points, axis=0)

    # Prevent division by zero
    span_orig = np.maximum(org_max - org_min, eps)
    span_new  = np.maximum(new_max - new_min,  eps)
    scale_factor = float(np.min(span_orig / span_new))

    cen = np.mean(new_points, axis=0)
    new_points = cen + (scale_factor * expand) * (new_points - cen)

    return new_points, new_normals, inlier_indices

def mls_denoising(points, search_radius, polynomial_order=2, percentile=99, expand=1.0):
    points_denoised, normals_denoised, _ = mls_surface_reconstruction(points, search_radius, polynomial_order, expand)

    kdtree = cKDTree(points_denoised)
    dis_list = []
    for i in range(len(points)):
        dis, idx = kdtree.query(points[i], k=1)
        dis_list.append(np.mean(dis))
    threshold = np.percentile(dis_list, percentile)
    return dis_list < threshold, points_denoised, normals_denoised