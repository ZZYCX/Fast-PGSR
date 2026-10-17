import cv2
import numpy as np
import pycolmap
import os
import sys
import shutil

# 正常使用这个
# ============================================================
# 配置
# ============================================================

if len(sys.argv) < 2:
    print("❌ 错误：未提供场景路径")
    sys.exit(1)

PROJECT_DIR = os.path.abspath(sys.argv[1])

# COLMAP image_undistorter 生成的去畸变图片
# ArUco 检测固定使用 images/，与 sparse/ 中的去畸变相机模型保持一致
IMAGE_DIR = os.path.join(PROJECT_DIR, "images")

# 原始COLMAP模型
COLMAP_DIR = os.path.join(PROJECT_DIR, "sparse")

# 对齐结果
OUTPUT_DIR = os.path.join(PROJECT_DIR, "sparse", "aligned")

# ArUco配置
ARUCO_DICT = cv2.aruco.DICT_6X6_250

TARGET_ID_1 = 0
TARGET_ID_2 = 1

# ArUco真实边长映射到最终坐标系中的长度
# 如果实际 ArUco 是 5 cm：
# MARKER_LENGTH = 1.0
# 那么最终 1 个单位 = 5 cm
MARKER_LENGTH = 1.0

# 至少需要多少个COLMAP注册视角
MIN_VIEWS = 2

# ------------------------------------------------------------
# 四角整体拟合质量阈值
#
# RMSE超过边长5%认为拟合质量很差
# 最大单点误差超过边长10%认为拟合质量很差
#
# 正常情况下应该远小于这两个值
# ------------------------------------------------------------
MAX_FIT_RMSE_RATIO = 0.05
MAX_POINT_ERROR_RATIO = 0.10

# ------------------------------------------------------------
# 3D 正方形几何约束
#
# 注意：这些判断全部作用于“三角化后的 3D 四角点”，
# 因此 ArUco 在真实空间中水平放、竖放、斜着放都不会受影响。
# ------------------------------------------------------------
SQUARE_MAX_EDGE_RATIO = 1.20      # 最长边 / 最短边
SQUARE_MAX_EDGE_CV = 0.10         # 四条边长度变异系数
SQUARE_MAX_DIAG_RATIO = 1.15      # 两条对角线长度比
SQUARE_MAX_ANGLE_DEG = 15.0       # 相邻边相对90°的最大偏差
SQUARE_MAX_PLANE_RATIO = 0.03     # 平面RMSE / 平均边长


# ============================================================
# 多视图DLT三角化
# ============================================================

def triangulate_points(proj_matrices, points_2d):
    """
    利用多个视角的投影矩阵和二维角点进行DLT三角化。

    参数
    ----
    proj_matrices:
        N个 3x4 投影矩阵

    points_2d:
        N个二维像素坐标

    返回
    ----
    三维坐标 X = [x, y, z]
    """

    A = []

    for P, pt in zip(proj_matrices, points_2d):

        x = pt[0]
        y = pt[1]

        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])

    A = np.asarray(A, dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)

    X = Vt[-1, :]

    if abs(X[3]) < 1e-12:
        raise RuntimeError(
            "三角化失败：齐次坐标的最后一维接近0"
        )

    return X[:3] / X[3]


# ============================================================
# 获取Marker基本几何信息
#
# 注意：
# 这里不是最终对齐。
# 这里只用于：
#
# 1. 计算Marker中心
# 2. 估计法向量
# 3. 双Marker情况下判断哪个更高
#
# 最终的旋转/尺度/平移完全由
# 四角点 -> 标准正方形整体拟合决定。
# ============================================================

def estimate_marker_info(pts3d):

    pts3d = np.asarray(
        pts3d,
        dtype=np.float64
    )

    if pts3d.shape != (4, 3):
        raise RuntimeError(
            f"ArUco角点尺寸错误：{pts3d.shape}"
        )

    # --------------------------------------------------------
    # 1. 四角共同决定中心
    # --------------------------------------------------------

    center = np.mean(
        pts3d,
        axis=0
    )

    # --------------------------------------------------------
    # 2. 两条横向边共同估计X方向
    #
    # 0 ----> 1
    #
    # 3 ----> 2
    # --------------------------------------------------------

    vx = (
        (pts3d[1] - pts3d[0]) +
        (pts3d[2] - pts3d[3])
    )

    vx_norm = np.linalg.norm(vx)

    if vx_norm < 1e-10:
        raise RuntimeError(
            "无法确定ArUco X方向"
        )

    vx = vx / vx_norm

    # --------------------------------------------------------
    # 3. 两条纵向边共同估计Y方向
    #
    # 0
    # |
    # v
    # 3
    #
    # 1
    # |
    # v
    # 2
    # --------------------------------------------------------

    vy = (
        (pts3d[3] - pts3d[0]) +
        (pts3d[2] - pts3d[1])
    )

    # 与X轴正交化
    vy = vy - np.dot(vy, vx) * vx

    vy_norm = np.linalg.norm(vy)

    if vy_norm < 1e-10:
        raise RuntimeError(
            "无法确定ArUco Y方向"
        )

    vy = vy / vy_norm

    # --------------------------------------------------------
    # 4. 法向量
    # --------------------------------------------------------

    vz = np.cross(vx, vy)

    vz_norm = np.linalg.norm(vz)

    if vz_norm < 1e-10:
        raise RuntimeError(
            "无法确定ArUco法向量"
        )

    vz = vz / vz_norm

    # --------------------------------------------------------
    # 5. 四条边长度
    # --------------------------------------------------------

    edge_lengths = np.array([
        np.linalg.norm(
            pts3d[1] - pts3d[0]
        ),
        np.linalg.norm(
            pts3d[2] - pts3d[1]
        ),
        np.linalg.norm(
            pts3d[3] - pts3d[2]
        ),
        np.linalg.norm(
            pts3d[0] - pts3d[3]
        ),
    ])

    mean_side = np.mean(edge_lengths)

    # --------------------------------------------------------
    # 6. SVD计算平面残差
    #
    # 这里只用于打印质量信息
    # 不参与最终姿态计算
    # --------------------------------------------------------

    centered = pts3d - center

    _, _, Vt = np.linalg.svd(
        centered
    )

    plane_normal = Vt[-1]

    plane_normal /= np.linalg.norm(
        plane_normal
    )

    distances = np.abs(
        centered @ plane_normal
    )

    plane_rmse = np.sqrt(
        np.mean(distances ** 2)
    )

    return {
        "center": center,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "edge_lengths": edge_lengths,
        "mean_side": mean_side,
        "plane_rmse": plane_rmse,
    }


# ============================================================
# 3D 正方形约束检查
# ============================================================

def square_metrics(points3d):
    """
    对三角化后的 4 个 3D 角点判断是否接近正方形。

    检查：
    1. 四边是否近似等长
    2. 两条对角线是否近似等长
    3. 四个内角是否接近 90°
    4. 四点是否近似共面

    这是纯 3D 几何判断，所以 Marker 在真实空间中斜放不会影响结果。
    """

    pts = np.asarray(
        points3d,
        dtype=np.float64
    )

    if pts.shape != (4, 3):
        raise RuntimeError(
            f"正方形检查要求4x3点，当前为：{pts.shape}"
        )

    # 四条边
    edge_vecs = np.array([
        pts[1] - pts[0],
        pts[2] - pts[1],
        pts[3] - pts[2],
        pts[0] - pts[3],
    ])

    edges = np.linalg.norm(
        edge_vecs,
        axis=1
    )

    min_edge = float(np.min(edges))
    max_edge = float(np.max(edges))
    mean_edge = float(np.mean(edges))

    if min_edge < 1e-12 or mean_edge < 1e-12:
        return {
            "valid": False,
            "edges": edges,
            "edge_ratio": float("inf"),
            "edge_cv": float("inf"),
            "diagonals": np.array([np.nan, np.nan]),
            "diag_ratio": float("inf"),
            "angles_deg": np.full(4, np.nan),
            "max_angle_deviation": float("inf"),
            "plane_rmse": float("inf"),
            "plane_ratio": float("inf"),
        }

    edge_ratio = max_edge / min_edge
    edge_cv = float(
        np.std(edges) / mean_edge
    )

    # 两条对角线
    diagonals = np.array([
        np.linalg.norm(
            pts[2] - pts[0]
        ),
        np.linalg.norm(
            pts[3] - pts[1]
        ),
    ])

    min_diag = float(np.min(diagonals))

    if min_diag < 1e-12:
        diag_ratio = float("inf")
    else:
        diag_ratio = float(
            np.max(diagonals) / min_diag
        )

    # 四个内角
    vertex_pairs = [
        (pts[3] - pts[0], pts[1] - pts[0]),
        (pts[0] - pts[1], pts[2] - pts[1]),
        (pts[1] - pts[2], pts[3] - pts[2]),
        (pts[2] - pts[3], pts[0] - pts[3]),
    ]

    angles_deg = []

    for v1, v2 in vertex_pairs:
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 < 1e-12 or n2 < 1e-12:
            angle = np.nan
        else:
            cos_value = np.dot(v1, v2) / (n1 * n2)
            cos_value = np.clip(
                cos_value,
                -1.0,
                1.0
            )

            angle = np.degrees(
                np.arccos(cos_value)
            )

        angles_deg.append(angle)

    angles_deg = np.asarray(
        angles_deg,
        dtype=np.float64
    )

    if np.any(
        ~np.isfinite(angles_deg)
    ):
        max_angle_deviation = float("inf")
    else:
        max_angle_deviation = float(
            np.max(
                np.abs(
                    angles_deg - 90.0
                )
            )
        )

    # 共面性
    center = np.mean(
        pts,
        axis=0
    )

    centered = pts - center

    _, _, Vt = np.linalg.svd(
        centered
    )

    normal = Vt[-1]
    normal_norm = np.linalg.norm(normal)

    if normal_norm < 1e-12:
        plane_rmse = float("inf")
        plane_ratio = float("inf")
    else:
        normal = normal / normal_norm

        distances = np.abs(
            centered @ normal
        )

        plane_rmse = float(
            np.sqrt(
                np.mean(
                    distances ** 2
                )
            )
        )

        plane_ratio = (
            plane_rmse / mean_edge
        )

    valid = (
        edge_ratio <= SQUARE_MAX_EDGE_RATIO
        and
        edge_cv <= SQUARE_MAX_EDGE_CV
        and
        diag_ratio <= SQUARE_MAX_DIAG_RATIO
        and
        max_angle_deviation <= SQUARE_MAX_ANGLE_DEG
        and
        plane_ratio <= SQUARE_MAX_PLANE_RATIO
    )

    return {
        "valid": bool(valid),
        "edges": edges,
        "edge_ratio": float(edge_ratio),
        "edge_cv": float(edge_cv),
        "diagonals": diagonals,
        "diag_ratio": float(diag_ratio),
        "angles_deg": angles_deg,
        "max_angle_deviation": float(
            max_angle_deviation
        ),
        "plane_rmse": float(
            plane_rmse
        ),
        "plane_ratio": float(
            plane_ratio
        ),
    }


def print_square_metrics(name, square):
    """
    打印 3D 正方形检查结果。
    """

    print(
        f"\n📐 {name} 3D正方形约束："
    )

    print(
        f"   最长/最短边比："
        f"{square['edge_ratio']:.6f} "
        f"(要求 <= {SQUARE_MAX_EDGE_RATIO:.2f})"
    )

    print(
        f"   边长CV："
        f"{square['edge_cv']:.6f} "
        f"(要求 <= {SQUARE_MAX_EDGE_CV:.2f})"
    )

    print(
        f"   两条对角线："
        f"{square['diagonals']}"
    )

    print(
        f"   对角线比："
        f"{square['diag_ratio']:.6f} "
        f"(要求 <= {SQUARE_MAX_DIAG_RATIO:.2f})"
    )

    print(
        f"   四个内角："
        f"{square['angles_deg']}"
    )

    print(
        f"   最大90°偏差："
        f"{square['max_angle_deviation']:.6f}° "
        f"(要求 <= {SQUARE_MAX_ANGLE_DEG:.1f}°)"
    )

    print(
        f"   平面误差/平均边长："
        f"{square['plane_ratio']:.6f} "
        f"(要求 <= {SQUARE_MAX_PLANE_RATIO:.2f})"
    )

    print(
        f"   判定："
        f"{'✅ 通过' if square['valid'] else '❌ 不通过'}"
    )


# ============================================================
# 构造标准ArUco正方形
# ============================================================

def make_target_square(marker_length):
    """
    构造最终坐标系中的标准ArUco。

    OpenCV ArUco角点顺序：

        0 -------- 1
        |          |
        |          |
        3 -------- 2

    最终定义为：

        (-L/2,+L/2,0) -------- (+L/2,+L/2,0)
               0                        1

               3                        2
        (-L/2,-L/2,0) -------- (+L/2,-L/2,0)

    这样：
        marker中心 = 原点
        marker边长 = MARKER_LENGTH
        marker平面 = XY平面

    注意：
        0->1 为 +X
        0->3 为 -Y

    因此：
        (0->1) × (0->3) = -Z

    这与原代码最终经过R_x180后得到-Z方向是一致的。
    """

    half = marker_length / 2.0

    target = np.array([
        [-half, +half, 0.0],
        [+half, +half, 0.0],
        [+half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)

    return target


# ============================================================
# 四角点 -> 标准正方形
# Umeyama Similarity Transformation
# ============================================================

def estimate_similarity_umeyama(
        source_points,
        target_points
):
    """
    求解：

        target ~= scale * R * source + t

    使用全部四个角点共同估计：

        scale
        rotation
        translation

    这是整个新方案的核心。

    参数
    ----
    source_points:
        Nx3，COLMAP三角化角点

    target_points:
        Nx3，标准正方形角点

    返回
    ----
    scale
    R
    t
    """

    X = np.asarray(
        source_points,
        dtype=np.float64
    )

    Y = np.asarray(
        target_points,
        dtype=np.float64
    )

    if X.shape != Y.shape:
        raise RuntimeError(
            "Umeyama源点和目标点尺寸不一致"
        )

    if X.ndim != 2 or X.shape[1] != 3:
        raise RuntimeError(
            "Umeyama输入必须为Nx3"
        )

    n = X.shape[0]

    if n < 3:
        raise RuntimeError(
            "Umeyama至少需要3个点"
        )

    # --------------------------------------------------------
    # 1. 质心
    # --------------------------------------------------------

    mean_X = np.mean(
        X,
        axis=0
    )

    mean_Y = np.mean(
        Y,
        axis=0
    )

    Xc = X - mean_X
    Yc = Y - mean_Y

    # --------------------------------------------------------
    # 2. 协方差
    #
    # 我们求：
    #
    # Y = s R X + t
    # --------------------------------------------------------

    covariance = (
        Yc.T @ Xc
    ) / n

    U, singular_values, Vt = np.linalg.svd(
        covariance
    )

    # --------------------------------------------------------
    # 3. 防止得到镜像变换
    #
    # Rotation必须满足：
    #
    # det(R) = +1
    # --------------------------------------------------------

    S = np.eye(3)

    if np.linalg.det(
        U @ Vt
    ) < 0:

        S[2, 2] = -1.0

    R = U @ S @ Vt

    # --------------------------------------------------------
    # 4. 求尺度
    # --------------------------------------------------------

    var_X = np.mean(
        np.sum(
            Xc ** 2,
            axis=1
        )
    )

    if var_X < 1e-15:
        raise RuntimeError(
            "Umeyama失败：源点方差接近0"
        )

    scale = (
        np.sum(
            singular_values *
            np.diag(S)
        )
        / var_X
    )

    if scale <= 0:
        raise RuntimeError(
            f"Umeyama得到异常尺度：{scale}"
        )

    # --------------------------------------------------------
    # 5. 求平移
    # --------------------------------------------------------

    t = (
        mean_Y -
        scale * (R @ mean_X)
    )

    return scale, R, t


# ============================================================
# 应用Sim3到点
# ============================================================

def transform_points(
        points,
        scale,
        R,
        t
):

    points = np.asarray(
        points,
        dtype=np.float64
    )

    return (
        scale *
        (R @ points.T)
    ).T + t


# ============================================================
# 计算四角拟合质量
# ============================================================

def evaluate_square_fit(
        source_points,
        target_points,
        scale,
        R,
        t
):

    aligned = transform_points(
        source_points,
        scale,
        R,
        t
    )

    residuals = (
        aligned -
        target_points
    )

    point_errors = np.linalg.norm(
        residuals,
        axis=1
    )

    rmse = np.sqrt(
        np.mean(
            point_errors ** 2
        )
    )

    max_error = np.max(
        point_errors
    )

    max_abs_z = np.max(
        np.abs(
            aligned[:, 2]
        )
    )

    # 对齐后的4条边
    aligned_edges = np.array([
        np.linalg.norm(
            aligned[1] - aligned[0]
        ),
        np.linalg.norm(
            aligned[2] - aligned[1]
        ),
        np.linalg.norm(
            aligned[3] - aligned[2]
        ),
        np.linalg.norm(
            aligned[0] - aligned[3]
        ),
    ])

    return {
        "aligned": aligned,
        "residuals": residuals,
        "point_errors": point_errors,
        "rmse": rmse,
        "max_error": max_error,
        "max_abs_z": max_abs_z,
        "aligned_edges": aligned_edges,
    }


# ============================================================
# 主程序
# ============================================================

def main():

    print(
        f"--- 正在对齐数据集：{PROJECT_DIR} ---"
    )

    required_files = [
        "cameras.bin",
        "images.bin",
        "points3D.bin",
    ]

    # ========================================================
    # 检查COLMAP模型
    # ========================================================

    for file_name in required_files:

        file_path = os.path.join(
            COLMAP_DIR,
            file_name
        )

        if not os.path.isfile(file_path):

            print(
                f"❌ 错误：缺少COLMAP文件："
                f"{file_path}"
            )

            sys.exit(1)

    # ========================================================
    # ArUco 检测图片目录
    #
    # 固定使用 COLMAP image_undistorter 生成的 images/：
    #   images/      -> 去畸变后的图像
    #   sparse/      -> 与 images/ 对应的去畸变相机模型
    #
    # 不再使用 input/，避免“原始畸变像素 + 去畸变相机模型”混用。
    # ========================================================

    img_dir = IMAGE_DIR

    if not os.path.isdir(img_dir):

        print(
            f"❌ 错误：找不到COLMAP去畸变图片目录：{img_dir}"
        )
        print(
            "   请先完成 COLMAP image_undistorter，确保场景下存在 images/。"
        )

        sys.exit(1)

    # ========================================================
    # 读取COLMAP模型
    # ========================================================

    recon = pycolmap.Reconstruction(
        COLMAP_DIR
    )

    print(
        f"COLMAP注册图片数量："
        f"{len(recon.images)}"
    )

    print(
        f"ArUco检测图片目录："
        f"{img_dir}"
    )

    # 文件名 -> COLMAP Image
    images_by_name = {
        os.path.basename(im.name): im
        for _, im in recon.images.items()
    }

    # ========================================================
    # ArUco检测器
    # ========================================================

    dictionary = (
        cv2.aruco.getPredefinedDictionary(
            ARUCO_DICT
        )
    )

    detector_params = cv2.aruco.DetectorParameters()
    detector_params.adaptiveThreshWinSizeMin = 3
    detector_params.adaptiveThreshWinSizeMax = 65
    detector_params.adaptiveThreshWinSizeStep = 4
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    fallback_detector = cv2.aruco.ArucoDetector(dictionary)

    # ========================================================
    # 保存每一帧的所有 ArUco 检测
    #
    # 重要：这里不再直接 matched[0]。
    # 因为历史数据可能存在两个物理魔方都使用同一个 ArUco ID。
    # 先完整保存，再自动判断：
    #   1) ID0 + ID1：沿用原来的不同 ID 逻辑
    #   2) 同一 ID 出现两个 Marker：启用 A/B 跨帧跟踪
    #   3) 只有一个 Marker：保留单 Marker 逻辑
    # ========================================================

    frame_records = {
        TARGET_ID_1: [],
        TARGET_ID_2: [],
    }

    img_files = sorted([
        file_name
        for file_name in os.listdir(img_dir)
        if file_name.lower().endswith(
            (
                ".png",
                ".jpg",
                ".jpeg",
            )
        )
    ])

    # ========================================================
    # 遍历图片
    # ========================================================

    for frame_index, fname in enumerate(img_files):

        image_obj = images_by_name.get(
            fname
        )

        # 没有被COLMAP注册
        if image_obj is None:
            continue

        img_path = os.path.join(
            img_dir,
            fname
        )

        frame = cv2.imread(
            img_path
        )

        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            # Small/oblique field markers can disappear after undistortion.
            # Convert fallback corners back to original camera pixel coordinates.
            enlarged = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            corners, ids, _ = fallback_detector.detectMarkers(enlarged)
            if ids is not None:
                corners = [corner / 2.0 for corner in corners]

        if ids is None:
            continue

        # ----------------------------------------------------
        # 相机内参
        # ----------------------------------------------------

        K = np.asarray(
            recon.cameras[
                image_obj.camera_id
            ].calibration_matrix(),
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # COLMAP world -> camera
        # ----------------------------------------------------

        pose = image_obj.cam_from_world

        if callable(pose):
            pose = pose()

        Rt = np.asarray(
            pose.matrix(),
            dtype=np.float64,
        )[:3, :4]

        P = K @ Rt

        detected_ids = ids.reshape(-1)

        # ----------------------------------------------------
        # 分别保存 ID0 / ID1 在这一帧中的“全部”检测
        # ----------------------------------------------------

        for target_id in (
            TARGET_ID_1,
            TARGET_ID_2
        ):

            matched = np.flatnonzero(
                detected_ids == target_id
            )

            if len(matched) == 0:
                continue

            detections = []

            for det_index in matched:
                marker_corners = np.asarray(
                    corners[int(det_index)][0],
                    dtype=np.float64,
                )
                detections.append(marker_corners)

            frame_records[target_id].append({
                "frame_index": frame_index,
                "fname": fname,
                "P": P,
                "detections": detections,
            })

    # ========================================================
    # 工具：从“每帧唯一检测”构造普通 ID 观测
    #
    # 如果同一帧出现多个相同 ID，这一帧不参与普通 ID 模式，
    # 避免再次发生 matched[0] 把两个物理魔方混起来。
    # ========================================================

    def build_unique_id_observations(records):

        obs = {
            i: []
            for i in range(4)
        }
        proj = []

        for rec in records:

            if len(rec["detections"]) != 1:
                continue

            marker_corners = rec["detections"][0]
            proj.append(rec["P"])

            for corner_index in range(4):
                obs[corner_index].append(
                    marker_corners[corner_index]
                )

        return proj, obs

    # ========================================================
    # 工具：同 ID 双 Marker 跨帧跟踪
    #
    # 初版策略（故意保守）：
    #   - 只使用“同一帧同时看到两个相同 ID”的帧；
    #   - 如果一帧检测到 >2 个同 ID，取面积最大的两个；
    #   - 第一帧按 x 坐标临时命名为 A / B；
    #   - 后续利用上一帧位置 + 简单速度预测做 2x2 最小代价匹配；
    #   - 遇到很长的双检缺口时，只取最长连续片段，避免跨大缺口串轨迹。
    #
    # A/B 只是程序内部身份，不代表左/右，也不代表真实编号。
    # ========================================================

    SAME_ID_MAX_GAP = 8

    def quad_area(marker_corners):
        pts = np.asarray(
            marker_corners,
            dtype=np.float64
        )
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * abs(
            np.dot(x, np.roll(y, -1)) -
            np.dot(y, np.roll(x, -1))
        )

    def track_two_same_id(records, marker_id):

        dual_records = []

        for rec in records:

            if len(rec["detections"]) < 2:
                continue

            # 若异常检测出 3 个以上同 ID，优先保留图像面积最大的两个。
            dets = sorted(
                rec["detections"],
                key=quad_area,
                reverse=True,
            )[:2]

            dual_records.append({
                **rec,
                "detections": dets,
            })

        if len(dual_records) < MIN_VIEWS:
            return None

        # ----------------------------------------------------
        # 按“连续可双检”切片，取最长片段。
        # 这样宁可少用一些帧，也不在长时间不可见后强行续轨。
        # ----------------------------------------------------

        segments = []
        current_segment = []
        last_index = None

        for rec in dual_records:

            if (
                last_index is None or
                rec["frame_index"] - last_index <= SAME_ID_MAX_GAP
            ):
                current_segment.append(rec)
            else:
                if current_segment:
                    segments.append(current_segment)
                current_segment = [rec]

            last_index = rec["frame_index"]

        if current_segment:
            segments.append(current_segment)

        segment = max(
            segments,
            key=len
        )

        if len(segment) < MIN_VIEWS:
            return None

        print(
            f"\n[同ID跟踪] ID={marker_id}："
            f"双Marker帧总数={len(dual_records)}，"
            f"采用最长连续片段={len(segment)}帧"
        )

        tracks = {
            "A": {
                "proj": [],
                "obs": {i: [] for i in range(4)},
                "last_center": None,
                "velocity": np.zeros(2, dtype=np.float64),
                "last_frame": None,
            },
            "B": {
                "proj": [],
                "obs": {i: [] for i in range(4)},
                "last_center": None,
                "velocity": np.zeros(2, dtype=np.float64),
                "last_frame": None,
            },
        }

        for rec_index, rec in enumerate(segment):

            dets = rec["detections"]
            centers = [
                np.mean(det, axis=0)
                for det in dets
            ]

            if rec_index == 0:

                # 仅第一帧用左右顺序初始化 A/B。
                # 后续不再假设 A 永远在左、B 永远在右。
                if centers[0][0] <= centers[1][0]:
                    assignment = {
                        "A": 0,
                        "B": 1,
                    }
                else:
                    assignment = {
                        "A": 1,
                        "B": 0,
                    }

            else:

                predicted = {}

                for name in ("A", "B"):
                    tr = tracks[name]
                    gap = max(
                        1,
                        rec["frame_index"] - tr["last_frame"]
                    )
                    predicted[name] = (
                        tr["last_center"] +
                        tr["velocity"] * gap
                    )

                # 两种可能的 A/B 分配，选总运动代价更小的。
                cost_ab = (
                    np.linalg.norm(centers[0] - predicted["A"]) +
                    np.linalg.norm(centers[1] - predicted["B"])
                )

                cost_ba = (
                    np.linalg.norm(centers[1] - predicted["A"]) +
                    np.linalg.norm(centers[0] - predicted["B"])
                )

                if cost_ab <= cost_ba:
                    assignment = {
                        "A": 0,
                        "B": 1,
                    }
                else:
                    assignment = {
                        "A": 1,
                        "B": 0,
                    }

            for name in ("A", "B"):

                det_index = assignment[name]
                marker_corners = dets[det_index]
                center = centers[det_index]
                tr = tracks[name]

                if tr["last_center"] is not None:
                    gap = max(
                        1,
                        rec["frame_index"] - tr["last_frame"]
                    )
                    instant_velocity = (
                        center - tr["last_center"]
                    ) / gap

                    # 速度做一点平滑，降低单帧抖动对下一帧匹配的影响。
                    tr["velocity"] = (
                        0.6 * tr["velocity"] +
                        0.4 * instant_velocity
                    )

                tr["last_center"] = center
                tr["last_frame"] = rec["frame_index"]
                tr["proj"].append(rec["P"])

                for corner_index in range(4):
                    tr["obs"][corner_index].append(
                        marker_corners[corner_index]
                    )

        return tracks

    # ========================================================
    # 自动模式判断
    # ========================================================

    unique_proj = {}
    unique_obs = {}
    unique_count = {}
    multi_frame_count = {}

    for marker_id in (
        TARGET_ID_1,
        TARGET_ID_2
    ):

        proj, obs = build_unique_id_observations(
            frame_records[marker_id]
        )

        unique_proj[marker_id] = proj
        unique_obs[marker_id] = obs
        unique_count[marker_id] = len(proj)

        multi_frame_count[marker_id] = sum(
            1
            for rec in frame_records[marker_id]
            if len(rec["detections"]) >= 2
        )

        print(
            f"ID={marker_id}："
            f"唯一检测帧={unique_count[marker_id]}，"
            f"同帧多检测帧={multi_frame_count[marker_id]}"
        )

    id0_valid = unique_count[TARGET_ID_1] >= MIN_VIEWS
    id1_valid = unique_count[TARGET_ID_2] >= MIN_VIEWS

    observation_sets = {}
    mode_name = None

    # --------------------------------------------------------
    # 模式 1：两个不同 ID 都足够可用 -> 完全保留原逻辑
    # --------------------------------------------------------

    if id0_valid and id1_valid:

        mode_name = "DIFFERENT_IDS"

        print(
            "\n✅ 自动模式：检测到两个不同 ArUco ID，"
            "沿用原 ID0 + ID1 逻辑"
        )

        for marker_id in (
            TARGET_ID_1,
            TARGET_ID_2
        ):
            observation_sets[marker_id] = {
                "proj": unique_proj[marker_id],
                "obs": unique_obs[marker_id],
                "display_name": f"ID={marker_id}",
            }

    else:

        # ----------------------------------------------------
        # 模式 2：某个 ID 在多帧里同时出现两个 -> 同 ID 双魔方
        # ----------------------------------------------------

        duplicate_candidates = [
            marker_id
            for marker_id in (
                TARGET_ID_1,
                TARGET_ID_2
            )
            if multi_frame_count[marker_id] >= MIN_VIEWS
        ]

        if duplicate_candidates:

            # 如果极端情况下两个 ID 都重复，优先选择双检帧更多的那个。
            duplicate_id = max(
                duplicate_candidates,
                key=lambda mid: multi_frame_count[mid]
            )

            tracks = track_two_same_id(
                frame_records[duplicate_id],
                duplicate_id
            )

            if tracks is None:
                print(
                    f"❌ ID={duplicate_id} 虽检测到重复 Marker，"
                    "但无法形成足够稳定的双Marker轨迹"
                )
                sys.exit(1)

            mode_name = "SAME_ID_TRACKING"

            print(
                f"✅ 自动模式：发现两个物理 Marker 共用 ID={duplicate_id}，"
                "启用 A/B 跨帧跟踪"
            )

            for name in ("A", "B"):
                key = f"ID{duplicate_id}-{name}"
                observation_sets[key] = {
                    "proj": tracks[name]["proj"],
                    "obs": tracks[name]["obs"],
                    "display_name": key,
                }

        else:

            # ------------------------------------------------
            # 模式 3：只有一个 ID，且没有重复证据 -> 单 Marker
            # ------------------------------------------------

            valid_single_ids = [
                marker_id
                for marker_id in (
                    TARGET_ID_1,
                    TARGET_ID_2
                )
                if unique_count[marker_id] >= MIN_VIEWS
            ]

            if len(valid_single_ids) == 0:

                # 如果只出现过一次同ID双检，也不能安全地当成单Marker。
                has_any_duplicate = any(
                    multi_frame_count[mid] > 0
                    for mid in (
                        TARGET_ID_1,
                        TARGET_ID_2
                    )
                )

                if has_any_duplicate:
                    print(
                        "❌ 检测到同 ID 多 Marker 的迹象，"
                        "但双检帧不足，无法安全区分 A/B；"
                        "为避免混合两个物理魔方，本次停止对齐。"
                    )
                else:
                    print(
                        "❌ 两个 ArUco ID 都没有足够的唯一观测，"
                        f"至少需要 {MIN_VIEWS} 个注册视角"
                    )

                sys.exit(1)

            single_id = valid_single_ids[0]
            mode_name = "SINGLE_ID"

            print(
                f"\n⚠️ 自动模式：仅 ID={single_id} 可用，"
                "按单 Marker 处理"
            )

            observation_sets[single_id] = {
                "proj": unique_proj[single_id],
                "obs": unique_obs[single_id],
                "display_name": f"ID={single_id}",
            }

    # ========================================================
    # 三角化每一个“物理 Marker”的四个角点
    # ========================================================

    markers_data = {}

    for marker_key, obs_set in observation_sets.items():

        proj = obs_set["proj"]
        obs = obs_set["obs"]

        if len(proj) < MIN_VIEWS:
            continue

        points_3d = []

        for corner_index in range(4):

            point_3d = triangulate_points(
                proj,
                obs[corner_index],
            )

            points_3d.append(
                point_3d
            )

        points_3d = np.asarray(
            points_3d,
            dtype=np.float64
        )

        marker_info = estimate_marker_info(
            points_3d
        )

        square = square_metrics(
            points_3d
        )

        markers_data[marker_key] = {
            "pts3d": points_3d,
            "display_name": obs_set["display_name"],
            "square": square,
            **marker_info,
        }

        print(
            f"\n📌 {obs_set['display_name']} 三角化质量："
        )

        print(
            f"   使用视角数：{len(proj)}"
        )

        print(
            f"   四条边：{marker_info['edge_lengths']}"
        )

        print(
            f"   平均边长：{marker_info['mean_side']:.6f}"
        )

        print(
            f"   平面RMSE：{marker_info['plane_rmse']:.6f}"
        )

        print_square_metrics(
            obs_set["display_name"],
            square
        )

    # ========================================================
    # 3D 正方形约束过滤
    #
    # 只有三角化后的四个 3D 角点满足正方形几何约束的 Marker，
    # 才允许参与后面的基准 Marker 选择和 Sim3。
    # ========================================================

    invalid_marker_keys = [
        key
        for key, data in markers_data.items()
        if not data["square"]["valid"]
    ]

    for key in invalid_marker_keys:
        print(
            f"\n⚠️ {markers_data[key]['display_name']} "
            "不满足3D正方形约束，已排除"
        )

        del markers_data[key]

    if len(markers_data) == 0:
        print(
            "❌ 没有满足3D正方形约束的可用 Marker"
        )
        sys.exit(1)

    # ========================================================
    # 选择基准 Marker
    #
    # 保留原来的基本思想：
    #   - 只有一个物理 Marker：直接使用；
    #   - 有两个物理 Marker：沿共同法向选择“更高”的那个。
    #
    # 这里不再假设 key 一定是整数 0 / 1，
    # 因为同 ID 跟踪模式下 key 会是 ID0-A / ID0-B。
    # ========================================================

    marker_keys = list(
        markers_data.keys()
    )

    if len(marker_keys) == 1:

        chosen_id = marker_keys[0]
        chosen_name = markers_data[
            chosen_id
        ]["display_name"]

        print(
            f"\n⚠️ 仅 {chosen_name} 具有足够观测，"
            "将其作为基准"
        )

    else:

        # 当前流程最多使用两个物理 Marker。
        key1 = marker_keys[0]
        key2 = marker_keys[1]

        name1 = markers_data[
            key1
        ]["display_name"]

        name2 = markers_data[
            key2
        ]["display_name"]

        vz1 = markers_data[
            key1
        ]["vz"]

        vz2 = markers_data[
            key2
        ]["vz"]

        # 如果两个法向方向相反，
        # 将第二个翻转到与第一个一致。
        if np.dot(
            vz1,
            vz2
        ) < 0:
            vz2 = -vz2

        up_dir = (
            vz1 + vz2
        ) / 2.0

        up_norm = np.linalg.norm(
            up_dir
        )

        if up_norm < 1e-10:

            print(
                "❌ 两个标记法向量无法确定统一高度方向"
            )

            sys.exit(1)

        up_dir = (
            up_dir /
            up_norm
        )

        center1 = markers_data[
            key1
        ]["center"]

        center2 = markers_data[
            key2
        ]["center"]

        h1 = np.dot(
            center1,
            up_dir
        )

        h2 = np.dot(
            center2,
            up_dir
        )

        if h1 >= h2:
            chosen_id = key1
        else:
            chosen_id = key2

        chosen_name = markers_data[
            chosen_id
        ]["display_name"]

        print(
            f"\n✅ 两个物理 Marker 均可用，"
            f"选择高度较高的 {chosen_name}"
        )

        print(
            f"   {name1}: h={h1:.6f}"
        )

        print(
            f"   {name2}: h={h2:.6f}"
        )

    # ========================================================
    # 获取选中的四个三维角点
    # ========================================================

    source_corners = markers_data[
        chosen_id
    ]["pts3d"]

    chosen_square = markers_data[
        chosen_id
    ]["square"]

    print(
        f"\n📍 基准标记：{markers_data[chosen_id]['display_name']}"
    )

    print_square_metrics(
        markers_data[chosen_id][
            "display_name"
        ],
        chosen_square
    )

    print(
        "   原始四个三维角点："
    )

    for i, p in enumerate(
        source_corners
    ):
        print(
            f"   corner{i}: {p}"
        )

    # ========================================================
    # 构造标准正方形
    # ========================================================

    target_corners = (
        make_target_square(
            MARKER_LENGTH
        )
    )

    print(
        "\n🎯 目标标准正方形："
    )

    for i, p in enumerate(
        target_corners
    ):
        print(
            f"   corner{i}: {p}"
        )

    # ========================================================
    # 核心：
    #
    # 四个角点 -> 标准正方形
    #
    # 一次性求：
    #
    # scale
    # R
    # t
    #
    # 不再单独使用：
    #
    # 0->1计算scale
    # 0/1/3计算姿态
    # R_x180
    # ========================================================

    scale, R_final, t_final = (
        estimate_similarity_umeyama(
            source_corners,
            target_corners,
        )
    )

    print(
        "\n=============================="
    )

    print(
        "四角整体Sim3拟合结果"
    )

    print(
        "=============================="
    )

    print(
        f"缩放因子：{scale:.8f}"
    )

    print(
        "旋转矩阵："
    )

    print(
        R_final
    )

    print(
        "平移向量："
    )

    print(
        t_final
    )

    print(
        f"det(R) = "
        f"{np.linalg.det(R_final):.8f}"
    )

    # ========================================================
    # 拟合质量检查
    # ========================================================

    fit = evaluate_square_fit(
        source_corners,
        target_corners,
        scale,
        R_final,
        t_final,
    )

    aligned_corners = fit[
        "aligned"
    ]

    print(
        "\n=============================="
    )

    print(
        "🔍 四角整体拟合质量"
    )

    print(
        "=============================="
    )

    for i in range(4):

        print(
            f"corner{i}:"
        )

        print(
            f"   变换后："
            f"{aligned_corners[i]}"
        )

        print(
            f"   目标值："
            f"{target_corners[i]}"
        )

        print(
            f"   误差："
            f"{fit['point_errors'][i]:.6f}"
        )

    print(
        f"\n四角拟合RMSE："
        f"{fit['rmse']:.6f}"
    )

    print(
        f"最大角点误差："
        f"{fit['max_error']:.6f}"
    )

    print(
        f"最大|Z|："
        f"{fit['max_abs_z']:.6f}"
    )

    print(
        "变换后的四条边长："
    )

    print(
        fit[
            "aligned_edges"
        ]
    )

    print(
        f"目标边长："
        f"{MARKER_LENGTH:.6f}"
    )

    # ========================================================
    # 中心验证
    # ========================================================

    source_center = np.mean(
        source_corners,
        axis=0
    )

    center_after = (
        scale *
        (R_final @ source_center)
        +
        t_final
    )

    print(
        f"\n🔍 Marker中心变换后："
        f"{center_after}"
    )

    print(
        f"   与原点距离："
        f"{np.linalg.norm(center_after):.8f}"
    )

    # ========================================================
    # 法向验证
    #
    # 目标正方形的法向应该是-Z
    # ========================================================

    x_dir = (
        aligned_corners[1] -
        aligned_corners[0]
    )

    y_dir = (
        aligned_corners[3] -
        aligned_corners[0]
    )

    normal_after = np.cross(
        x_dir,
        y_dir
    )

    normal_after /= np.linalg.norm(
        normal_after
    )

    print(
        f"🔍 Marker法向量："
        f"{normal_after}"
    )

    print(
        f"   Z方向分量："
        f"{normal_after[2]:.8f}"
    )

    # ========================================================
    # 新的质量判断
    #
    # 不再采用：
    #
    # max|Z| > 0.01
    #
    # 就直接失败。
    #
    # 而是判断整体正方形拟合误差。
    # ========================================================

    max_rmse = (
        MARKER_LENGTH *
        MAX_FIT_RMSE_RATIO
    )

    max_point_error = (
        MARKER_LENGTH *
        MAX_POINT_ERROR_RATIO
    )

    if fit["rmse"] > max_rmse:

        print(
            "\n❌ ArUco四角整体拟合误差过大"
        )

        print(
            f"   RMSE = "
            f"{fit['rmse']:.6f}"
        )

        print(
            f"   允许最大RMSE = "
            f"{max_rmse:.6f}"
        )

        sys.exit(1)

    if fit[
        "max_error"
    ] > max_point_error:

        print(
            "\n❌ 存在严重异常的ArUco角点"
        )

        print(
            f"   最大角点误差 = "
            f"{fit['max_error']:.6f}"
        )

        print(
            f"   允许最大误差 = "
            f"{max_point_error:.6f}"
        )

        sys.exit(1)

    print(
        "\n✅ ArUco四角整体拟合质量正常"
    )

    # ========================================================
    # 构造pycolmap Sim3
    # ========================================================

    sim3 = pycolmap.Sim3d(
        scale,
        pycolmap.Rotation3d(
            R_final
        ),
        t_final,
    )

    # ========================================================
    # 对整个COLMAP模型进行Sim3变换
    # ========================================================

    recon.transform(
        sim3
    )

    # ========================================================
    # 保存
    # ========================================================

    # 删除旧aligned
    if os.path.exists(
        OUTPUT_DIR
    ):

        shutil.rmtree(
            OUTPUT_DIR
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    recon.write(
        OUTPUT_DIR
    )

    # ========================================================
    # 检查输出
    # ========================================================

    for file_name in required_files:

        output_file = os.path.join(
            OUTPUT_DIR,
            file_name
        )

        if not os.path.isfile(
            output_file
        ):

            print(
                f"❌ 错误：对齐结果缺少："
                f"{output_file}"
            )

            sys.exit(1)

    print(
        "\n🎉 对齐完成！"
    )

    print(
        f"对齐后的COLMAP模型已保存至："
        f"{OUTPUT_DIR}"
    )

    print(
        "输出结构："
    )

    print(
        "sparse/aligned/"
        "{cameras.bin, images.bin, points3D.bin}"
    )


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    main()
