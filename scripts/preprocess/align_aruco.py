import cv2
import numpy as np
import pycolmap
import os
import sys
import shutil
# --- 配置 ---
if len(sys.argv) < 2:
    print("❌ 错误: 未提供路径")
    sys.exit(1)

PROJECT_DIR = sys.argv[1]
IMAGE_DIR = os.path.join(PROJECT_DIR, "images")

# Fast-PGSR最终读取的COLMAP模型
COLMAP_DIR = os.path.join(PROJECT_DIR, "sparse", "0")

# 原始未对齐模型备份
RAW_COLMAP_DIR = os.path.join(PROJECT_DIR, "sparse", "0_raw")

# 临时保存对齐后的模型
OUTPUT_DIR = os.path.join(PROJECT_DIR, "sparse", "0_aligned_tmp")

ARUCO_DICT = cv2.aruco.DICT_6X6_250
# 两个标记的 ID（请根据实际使用的 ID 修改）
TARGET_ID_1 = 0
TARGET_ID_2 = 1
MARKER_LENGTH = 1.0   # 目标单位：边长（5 cm） -> 1.0

def triangulate_points(proj_matrices, points_2d):
    A = []
    for P, pt in zip(proj_matrices, points_2d):
        A.append(pt[0] * P[2, :] - P[0, :])
        A.append(pt[1] * P[2, :] - P[1, :])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1, :]
    return X[:3] / X[3]

def estimate_marker_pose(pts3d):
    """从4个角点计算标记中心、边长、局部坐标系 (vx, vy, vz)"""
    center = np.mean(pts3d, axis=0)
    vx = pts3d[1] - pts3d[0]
    side_len = np.linalg.norm(vx)
    vx /= side_len
    vy = pts3d[3] - pts3d[0]
    vy -= np.dot(vy, vx) * vx
    vy /= np.linalg.norm(vy)
    vz = np.cross(vx, vy)
    vz /= np.linalg.norm(vz)
    return center, side_len, vx, vy, vz

def main():
    print(f"--- 正在对齐数据集: {PROJECT_DIR} ---")

    if not os.path.exists(COLMAP_DIR):
        print(f"❌ 错误: 找不到 COLMAP 模型 {COLMAP_DIR}")
        sys.exit(1)

    if not os.path.exists(IMAGE_DIR):
        print(f"❌ 错误: 找不到去畸变图片目录 {IMAGE_DIR}")
        sys.exit(1)

    # 第一次运行时，备份未经对齐的COLMAP模型
    if not os.path.exists(RAW_COLMAP_DIR):
        shutil.copytree(COLMAP_DIR, RAW_COLMAP_DIR)
        print(f"✅ 已备份原始COLMAP模型至: {RAW_COLMAP_DIR}")

    # 始终从原始模型开始对齐，避免重复运行后累计旋转和缩放
    recon = pycolmap.Reconstruction(RAW_COLMAP_DIR)

    img_dir = IMAGE_DIR

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT))

    # 分别存储两个标记的观测 {标记ID: {角点索引: [像素坐标列表]}}
    obs_2d = {TARGET_ID_1: {i: [] for i in range(4)},
              TARGET_ID_2: {i: [] for i in range(4)}}
    proj_matrices = {TARGET_ID_1: [], TARGET_ID_2: []}  # 每个标记的投影矩阵列表

    # 遍历图像
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for fname in img_files:
        img_path = os.path.join(img_dir, fname)
        # 查找对应的 COLMAP 图像对象
        image_obj = None
        for im_id, im in recon.images.items():
            if im.name == fname:
                image_obj = im
                break
        if image_obj is None:
            continue

        frame = cv2.imread(img_path)
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is None:
            continue

        # 获取相机投影矩阵
        K = recon.cameras[image_obj.camera_id].calibration_matrix()
        pose = image_obj.cam_from_world() if callable(image_obj.cam_from_world) else image_obj.cam_from_world
        Rt = pose.matrix()[:3, :4]
        P = K @ Rt

        # 对每个感兴趣的标记 ID 收集数据
        for target_id in (TARGET_ID_1, TARGET_ID_2):
            if target_id in ids:
                idx = np.where(ids == target_id)[0][0]
                m_corners = corners[idx][0]
                proj_matrices[target_id].append(P)
                for i in range(4):
                    obs_2d[target_id][i].append(m_corners[i])

    # 检查至少有一个标记有足够观测
    has_valid = {tid: len(proj_matrices[tid]) >= 2 for tid in (TARGET_ID_1, TARGET_ID_2)}
    if not any(has_valid.values()):
        print("❌ 识别失败: 两个 Aruco 码均未找到足够视角（至少需要2个视角）！")
        sys.exit(1)

    chosen_id = None
    pts3d_chosen = None
    center_chosen = None
    side_len_chosen = None
    vx_chosen, vy_chosen, vz_chosen = None, None, None

    # 先分别三角化两个标记（如果有足够视角）
    markers_data = {}
    for tid in (TARGET_ID_1, TARGET_ID_2):
        if not has_valid[tid]:
            continue
        # 三角化四个角点
        pts3d = np.array([triangulate_points(proj_matrices[tid], obs_2d[tid][i]) for i in range(4)])
        center, side_len, vx, vy, vz = estimate_marker_pose(pts3d)
        markers_data[tid] = {
            'pts3d': pts3d,
            'center': center,
            'side_len': side_len,
            'vx': vx, 'vy': vy, 'vz': vz
        }

    if len(markers_data) == 1:
        # 只有一个标记可用，直接用它
        chosen_id = list(markers_data.keys())[0]
        print(f"⚠️ 仅检测到 ID={chosen_id} 的标记，将其作为基准。")
    else:
        # 两个标记都可用，比较 Z 轴高度
        # 使用两个标记法向量的平均方向作为“上方向”
        vz1 = markers_data[TARGET_ID_1]['vz']
        vz2 = markers_data[TARGET_ID_2]['vz']
        up_dir = (vz1 + vz2) / 2.0
        up_dir /= np.linalg.norm(up_dir)

        c1 = markers_data[TARGET_ID_1]['center']
        c2 = markers_data[TARGET_ID_2]['center']
        h1 = np.dot(c1, up_dir)
        h2 = np.dot(c2, up_dir)

        if h1 >= h2:
            chosen_id = TARGET_ID_1
        else:
            chosen_id = TARGET_ID_2
        print(f"✅ 两个标记均可用，选择 Z 高度较高的 ID={chosen_id} (h1={h1:.3f}, h2={h2:.3f})")

    # 取出选中的标记数据
    md = markers_data[chosen_id]
    pts3d_chosen = md['pts3d']
    center_chosen = md['center']
    side_len_chosen = md['side_len']
    vx_chosen, vy_chosen, vz_chosen = md['vx'], md['vy'], md['vz']

    # 计算缩放因子（边长映射到 1 个单位）
    scale = MARKER_LENGTH / side_len_chosen

    # 确保标记平面平行于 XY 平面（法向量 vz 指向 +Z 方向）
    # 计算当前 vz 与世界 Z 轴的夹角
    world_z = np.array([0, 0, 1])
    vz_dot_z = np.dot(vz_chosen, world_z)
    
    # 如果法向量指向下方，翻转 vy 和 vz
    if vz_dot_z < 0:
        vz_chosen = -vz_chosen
        vy_chosen = -vy_chosen
        vz_dot_z = -vz_dot_z

    # 构造旋转矩阵：R_align 的列是标记坐标系的轴在世界坐标系中的表示
    # R_align 表示从标记坐标系到世界坐标系的旋转
    # Sim3d 的变换公式: X_new = scale * R * X_old + t
    # 我们要将标记中心移到原点，标记平面对齐 XY 平面
    # 如果标记上一点的局部坐标是 p_local，原始世界坐标是: X_old = center_chosen + R_align @ p_local
    # 我们希望变换后: X_new = scale * p_local
    # 代入得: scale * p_local = scale * R_transform @ (center_chosen + R_align @ p_local) + t
    # 要成立必须: R_transform = R_align.T, t = -scale * R_align.T @ center_chosen
    R_align = np.column_stack((vx_chosen, vy_chosen, vz_chosen))
    R_transform = R_align.T
    
    # 平移向量：将标记中心移到原点
    t_align = -scale * (R_transform @ center_chosen)

    # 输出对齐信息
    print(f"📍 基准标记 ID={chosen_id}")
    print(f"   原始边长: {side_len_chosen:.4f}, 缩放因子: {scale:.4f}")
    print(f"   中心坐标: {center_chosen} -> 将移至原点")
    print(f"   标记法向量与 Z 轴夹角余弦: {vz_dot_z:.4f}")

    # 添加绕 X 轴旋转 180 度的变换
    # 旋转矩阵：绕X轴转180度 -> Y和Z坐标取反
    R_x180 = np.array([[1, 0, 0],
                       [0, -1, 0],
                       [0, 0, -1]])
    
    # 组合变换：先对齐到标记坐标系，再绕X轴旋转180度
    R_final = R_x180 @ R_transform
    t_final = R_x180 @ t_align
    
    # 执行变换
    sim3 = pycolmap.Sim3d(scale, pycolmap.Rotation3d(R_final), t_final)
    recon.transform(sim3)

    # 验证：变换后标记中心应在原点，法向量指向+Z方向（考虑绕X轴旋转180度）
    # 验证1：标记中心变换后是否在原点
    center_after = scale * R_final @ center_chosen + t_final
    print(f"🔍 验证1 - 标记中心变换后坐标: {center_after}")
    print(f"   与原点的距离: {np.linalg.norm(center_after):.6f}")
    
    # 验证2：标记法向量变换后是否指向+Z方向（绕X轴旋转180度后法向量方向不变）
    vz_after = scale * R_final @ (center_chosen + vz_chosen) + t_final - center_after
    vz_after /= np.linalg.norm(vz_after)
    print(f"🔍 验证2 - 标记法向量变换后方向: {vz_after}")
    print(f"   与+Z轴的夹角余弦: {vz_after[2]:.6f}")
    
    # 验证3：四个角点变换后的坐标（应该在 XY 平面上）
    print("🔍 验证3 - 四个角点变换后坐标:")
    for i, pt in enumerate(pts3d_chosen):
        pt_after = scale * R_final @ pt + t_final
        print(f"   角点{i}: {pt_after}")
    
    # 验证4：左上角点相对坐标（绕X轴旋转180度后变为 (0.5, 0.5, 0)）
    test_pt = scale * R_x180 @ R_transform @ (pts3d_chosen[0] - center_chosen)
    print(f"🔍 验证4 - 左上角点相对坐标（绕X轴旋转180度后）: {test_pt}")
    
    # 检查验证结果
    if np.linalg.norm(center_after) > 0.01:
        print("❌ 警告: 标记中心未准确移到原点！")
    if vz_after[2] < 0.99:
        print("❌ 警告: 标记法向量未准确指向+Z方向！")
    if np.linalg.norm(test_pt) > 2.0:
        print("❌ 警告: 坐标对齐计算可能有误！")
        sys.exit(1)

    # 清理上次可能遗留的临时目录
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 先写入临时目录
    recon.write(OUTPUT_DIR)

    # 用对齐后的模型替换sparse/0
    # Fast-PGSR训练时会直接读取sparse/0
    if os.path.exists(COLMAP_DIR):
        shutil.rmtree(COLMAP_DIR)

    shutil.move(OUTPUT_DIR, COLMAP_DIR)

    print(f"🎉 对齐后的COLMAP模型已保存至: {COLMAP_DIR}")
    print(f"📦 原始未对齐模型保存在: {RAW_COLMAP_DIR}")

if __name__ == "__main__":
    main()