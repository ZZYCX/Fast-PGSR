#!/usr/bin/env python
#
# PGSR Normal Visualizer
# ----------------------
# 3D 可视化训练/导出的高斯模型的每个三维点的法向量。
#
# 说明：
#   Fast-PGSR 保存的 point_cloud.ply 中 nx/ny/nz 恒为 0（save_ply 里 normals=zeros），
#   真正的法向量按 PGSR 的定义直接从高斯属性重建，与训练时的 get_smallest_axis() 一致：
#     normal = 最小尺度轴 = 四元数旋转矩阵中对应 argmin(scale) 的那一列
#   本脚本纯 CPU 离线计算，不需要 CUDA，也不改动任何训练/导出文件。
#
# 两种渲染后端（--backend，默认 auto 自动选择）：
#   * open3d     交互窗口（本地桌面或 ssh -X/-Y 转发时使用；可 --screenshot 存图）
#   * matplotlib 离屏渲染存 PNG，无窗口，纯 SSH 无显示环境也能用（推荐服务器用）
#
# 依赖（fast-pgsr 环境已具备）：plyfile, numpy, open3d, matplotlib
#
# 用法示例（SSH 服务器，无显示）：
#   python scripts/visualize_normals.py \
#       --ply output/mungbean13_semantic_v4/target_exports/plant_only/point_cloud.ply \
#       --sidecar output/mungbean13_semantic_v4/target_exports/plant_only/semantic_state.npz \
#       --color class --orient outward --max_arrows 20000 \
#       --backend matplotlib --screenshot /tmp/plant_normals.png
#
#   本地桌面交互查看：
#       python scripts/visualize_normals.py \
#           --ply output/mungbean13_semantic_v4/target_exports/plant_only/point_cloud.ply \
#           --color normal --orient outward
#

import argparse
import os

import numpy as np
from plyfile import PlyData

C0 = 0.28209479177387814  # SH 直流系数转基色，与 utils/sh_utils.py 的 SH2RGB 一致

CLASS_COLORS = {
    -1: (1.0, 1.0, 0.0),        # unknown -> 黄
    0: (0.5, 0.5, 0.5),         # background -> 灰
    1: (0.0, 0.78, 0.0),        # plant -> 绿
    2: (0.86, 0.08, 0.24),      # rubik_1 -> 红
    3: (1.0, 0.84, 0.0),        # rubik_2 -> 金
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize PGSR Gaussian normals (smallest-axis normals)")
    p.add_argument("--ply", type=str, required=True, help="输入高斯模型 point_cloud.ply")
    p.add_argument("--sidecar", type=str, default=None,
                   help="可选 semantic_state.npz，用于按类别着色（object_id）")
    p.add_argument("--color", type=str, choices=["sh", "class", "normal"], default="sh",
                   help="点云着色方式：sh=真实基色，class=按语义类别，normal=按法向方向")
    p.add_argument("--orient", type=str, choices=["outward", "origin", "none"], default="outward",
                   help="法向符号对齐：outward=朝物体质心外侧（推荐），origin=朝原点外侧，none=保持原始±")
    p.add_argument("--arrow_scale", type=float, default=0.05,
                   help="箭头长度 = arrow_scale × 包围盒对角线（相对尺度，推荐 0.01~0.1）")
    p.add_argument("--max_arrows", type=int, default=0,
                   help="最多绘制多少条法向箭头（0=全部；点云太大时建议 ≤20000）")
    p.add_argument("--subsample", type=int, default=1,
                   help="每 N 个点取 1 个（点云与箭头都按此抽稀，默认 1=全量）")
    p.add_argument("--max_points_plot", type=int, default=20000,
                   help="matplotlib 后端最多绘制的点数（超出按步长抽稀）")
    p.add_argument("--point_size", type=float, default=1.0, help="点云渲染点大小")
    p.add_argument("--no_arrows", action="store_true", help="不画法向箭头（只看点云着色）")
    p.add_argument("--backend", type=str, choices=["auto", "open3d", "matplotlib"], default="auto",
                   help="auto=有 DISPLAY 用 open3d，否则用 matplotlib；也可强制指定")
    p.add_argument("--screenshot", type=str, default=None,
                   help="保存渲染结果为该 PNG（matplotlib 后端默认 normals_preview.png；"
                        "open3d 后端需有 DISPLAY 才能离屏截图）")
    p.add_argument("--elev", type=float, default=25.0, help="matplotlib 视角仰角（度）")
    p.add_argument("--azim", type=float, default=45.0, help="matplotlib 视角方位角（度）")
    return p.parse_args()


def load_ply(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"PLY 不存在: {path}")
    ply = PlyData.read(path)
    v = ply["vertex"]
    n = v.count
    prop_names = {prop.name for prop in v.properties}
    required = ["x", "y", "z", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3"]
    missing = [r for r in required if r not in prop_names]
    if missing:
        raise ValueError(f"PLY 缺少属性 {missing}，无法重建 PGSR 法向")

    def col(name, dtype):
        return np.asarray(v[name], dtype=dtype)

    xyz = np.stack([col("x", np.float32), col("y", np.float32), col("z", np.float32)], axis=1)
    scale = np.stack([col("scale_0", np.float32), col("scale_1", np.float32),
                      col("scale_2", np.float32)], axis=1)          # 对数尺度(log-scale)
    rot = np.stack([col("rot_0", np.float32), col("rot_1", np.float32),
                    col("rot_2", np.float32), col("rot_3", np.float32)], axis=1)  # [w,x,y,z]
    f_dc = np.stack([col("f_dc_0", np.float32), col("f_dc_1", np.float32),
                     col("f_dc_2", np.float32)], axis=1) if "f_dc_0" in prop_names else None
    return xyz, scale, rot, f_dc


def quaternion_to_matrix(rot):
    """[N,4] 四元数 [w,x,y,z] -> [N,3,3] 旋转矩阵，公式与 pytorch3d.transforms 一致。"""
    w, x, y, z = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    norm = np.where(norm > 1e-8, norm, 1.0)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    col0 = np.stack([1 - 2 * (y * y + z * z),
                     2 * (x * y + w * z),
                     2 * (x * z - w * y)], axis=-1)
    col1 = np.stack([2 * (x * y - w * z),
                     1 - 2 * (x * x + z * z),
                     2 * (y * z + w * x)], axis=-1)
    col2 = np.stack([2 * (x * z + w * y),
                     2 * (y * z - w * x),
                     1 - 2 * (x * x + y * y)], axis=-1)
    return np.stack([col0, col1, col2], axis=-1)  # [N,3,3]，最后一维=列


def compute_pgsr_normals(xyz, scale, rot, orient):
    """重建 PGSR 最小尺度轴法向，与 GaussianModel.get_smallest_axis() 一致。"""
    # scale 是 log-scale，exp 单调不减，argmin 等价于对激活后尺度取最小
    k = np.argmin(scale, axis=1)                      # [N]，每个点最小尺度对应的轴索引
    R = quaternion_to_matrix(rot)                     # [N,3,3]
    normals = np.take_along_axis(R, k[:, None, None].repeat(3, axis=1), axis=2)
    normals = normals.squeeze(2)                      # [N,3]
    # 单位化
    nrm = np.linalg.norm(normals, axis=1, keepdims=True)
    nrm = np.where(nrm > 1e-8, nrm, 1.0)
    normals = normals / nrm

    if orient == "none":
        return normals
    if orient == "outward":
        ref = xyz - xyz.mean(axis=0)                  # 指向物体质心外侧
    else:  # origin
        ref = xyz
    flip = (normals * ref).sum(-1) < 0.0
    normals[flip] *= -1.0
    return normals


def sh_base_color(f_dc):
    """SH 直流系数 -> 基色 RGB，SH2RGB(f_dc) = f_dc*C0 + 0.5，截断到 [0,1]。"""
    rgb = f_dc * C0 + 0.5
    return np.clip(rgb, 0.0, 1.0)


def class_colors(oid):
    oid = np.asarray(oid)
    cols = np.zeros((oid.shape[0], 3), dtype=np.float64)
    for cid, rgb in CLASS_COLORS.items():
        cols[oid == cid] = rgb
    return cols


def normal_rgb(normals):
    """把法向方向映射到 RGB：(n+1)/2。"""
    return (normals + 1.0) / 2.0


def select_display_indices(n, subsample, max_points_plot):
    """选点云显示用的索引（步长抽稀 + 上限抽稀），确定性。"""
    sel = np.arange(0, n, subsample)
    if max_points_plot > 0 and sel.size > max_points_plot:
        stride = max(1, sel.size // max_points_plot)
        sel = sel[::stride]
    return sel


def select_arrow_indices(sel, max_arrows):
    """从已选点里挑箭头的索引（确定性随机抽到上限）。"""
    if max_arrows > 0 and sel.size > max_arrows:
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(sel, size=max_arrows, replace=False))
    return sel


# --------------------------------------------------------------------------- #
# open3d 后端
# --------------------------------------------------------------------------- #
def render_open3d(xyz_s, normals_s, colors, arrow_len, arrow_sel, args, screenshot):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_s)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geoms = [pcd]
    if not args.no_arrows:
        tips = xyz_s[arrow_sel] + normals_s[arrow_sel] * arrow_len
        pts = np.repeat(xyz_s[arrow_sel], 2, axis=0)
        pts[1::2] = tips
        lines = np.arange(pts.shape[0]).reshape(-1, 2)
        cols = np.repeat(normal_rgb(normals_s[arrow_sel]), 2, axis=0)
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(cols)
        geoms.append(ls)

    bbox_diag = np.linalg.norm(xyz_s.max(0) - xyz_s.min(0))
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05 * bbox_diag))

    if not os.environ.get("DISPLAY"):
        raise RuntimeError("当前无 DISPLAY（SSH 未做图形转发）。请用 --backend matplotlib，"
                           "或用 ssh -X/-Y 转发后运行。")

    vis = o3d.visualization.Visualizer()
    ok = vis.create_window(window_name="PGSR Normals", width=1600, height=1200,
                           visible=not bool(screenshot))
    if not ok:
        raise RuntimeError("Open3D 窗口初始化失败（可能是无图形环境）。请改用 --backend matplotlib。")
    for g in geoms:
        vis.add_geometry(g)
    ro = vis.get_render_option()
    if ro is not None:
        ro.point_size = args.point_size
        ro.background_color = np.array([0.1, 0.1, 0.1])
    if screenshot:
        for _ in range(5):  # 多轮刷新保证离屏渲染完成
            vis.poll_events()
            vis.update_renderer()
        vis.capture_screen_image(screenshot)
        vis.destroy_window()
        print(f"[NormalVis] 截图已保存: {screenshot}")
    else:
        vis.run()
        vis.destroy_window()


# --------------------------------------------------------------------------- #
# matplotlib 后端（离屏存 PNG，SSH 无显示可用）
# --------------------------------------------------------------------------- #
def render_matplotlib(xyz_s, normals_s, colors, arrow_len, arrow_sel, args, screenshot):
    import matplotlib
    matplotlib.use("Agg")  # 强制无界面后端
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(xyz_s[:, 0], xyz_s[:, 1], xyz_s[:, 2],
               c=colors, s=max(0.2, args.point_size * 1.0), depthshade=False)

    if not args.no_arrows:
        a = arrow_sel
        ax.quiver(xyz_s[a, 0], xyz_s[a, 1], xyz_s[a, 2],
                  normals_s[a, 0], normals_s[a, 1], normals_s[a, 2],
                  length=arrow_len, normalize=True, arrow_length_ratio=0.35,
                  color=(0.6, 0.1, 0.1, 0.7), linewidth=0.5)

    # 等比例坐标，避免形状被拉伸失真
    mn = xyz_s.min(0)
    mx = xyz_s.max(0)
    ctr = (mn + mx) / 2
    half = 0.55 * np.max(mx - mn)
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=args.elev, azim=args.azim)

    fig.tight_layout()
    fig.savefig(screenshot, dpi=150)
    plt.close(fig)
    print(f"[NormalVis] 截图已保存: {screenshot}（可改 --elev/--azim 换视角）")


def main():
    args = parse_args()
    xyz, scale, rot, f_dc = load_ply(args.ply)
    n = xyz.shape[0]
    normals = compute_pgsr_normals(xyz, scale, rot, args.orient)

    # 显示点（步长+上限抽稀），箭头是其子集
    sel = select_display_indices(n, args.subsample, args.max_points_plot)
    xyz_s = xyz[sel]
    normals_s = normals[sel]
    arrow_sel = select_arrow_indices(sel, args.max_arrows)

    # 着色
    if args.color == "class":
        if not args.sidecar or not os.path.exists(args.sidecar):
            raise FileNotFoundError("--color class 需要 --sidecar 指向存在的 semantic_state.npz")
        oid = np.asarray(np.load(args.sidecar)["object_id"])[sel]
        if oid.shape[0] != xyz_s.shape[0]:
            raise ValueError("sidecar 点数与 PLY 不一致")
        colors = class_colors(oid)
    elif args.color == "normal":
        colors = normal_rgb(normals_s)
    else:  # sh
        if f_dc is None:
            raise ValueError("该 PLY 不含 f_dc，无法按 sh 着色，请用 --color class/normal")
        colors = sh_base_color(f_dc[sel])

    bbox_diag = np.linalg.norm(xyz.max(0) - xyz.min(0))
    arrow_len = args.arrow_scale * bbox_diag

    # 后端选择：auto 按是否有 DISPLAY 决定
    backend = args.backend
    if backend == "auto":
        backend = "open3d" if os.environ.get("DISPLAY") else "matplotlib"

    print(f"[NormalVis] {n} 个高斯点，显示 {xyz_s.shape[0]} 个，箭头 {arrow_sel.size} 条，"
          f"箭头长度 {arrow_len:.4f}，法向来源=PGSR 最小尺度轴，后端={backend}")

    if backend == "matplotlib":
        screenshot = args.screenshot or "normals_preview.png"
        render_matplotlib(xyz_s, normals_s, colors, arrow_len, arrow_sel, args, screenshot)
    else:
        render_open3d(xyz_s, normals_s, colors, arrow_len, arrow_sel, args, args.screenshot)


if __name__ == "__main__":
    main()
