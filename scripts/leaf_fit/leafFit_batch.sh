#!/usr/bin/env bash
# ============================================================
# 批处理：对 output 下每个场景的 plant_only 点云执行 LeafFit
#
# 每个场景等价执行：
#   python scripts/leaf_fit/leafFit_segment_root_path.py \
#       --input <scene>/plant_only/point_cloud.ply \
#       --leaf-fit-root third_party/leaf_fit \
#       --disable-apex-cluster --pick-root --tip-k 128 --path-k 64
#
# LEAF_ARGS 中带 --pick-root 时每个场景会弹 Open3D 窗口手动选 Root：
#   Shift+左键 选点，Shift+右键 撤销最后选点，按 Q 结束（无需再在命令行按 y）。
# 已运行过的场景也会被覆盖重跑：leafFit_segment_root_path.py 启动时会先清空
# 旧的 leaf_fit_segments/，所以重复运行是安全的。
#
# 用法:
#   ./leafFit_batch.sh [output 根目录]   # 第一个参数覆盖下方 OUTPUT_DIR
# ============================================================
set -uo pipefail

# ---------------- 配置区（需要调整时只改这里） ----------------
CONDA_ENV="${CONDA_ENV:-fast-pgsr}"   # leaf_fit 需要 open3d/fpsample/plyfile/potpourri3d，fast-pgsr 环境已装；设为空字符串则跳过激活
OUTPUT_DIR="${1:-output}"             # 输出根目录：其下每个子目录是一个 scene
PYTHON="${PYTHON:-python}"            # python 解释器（默认取 conda 环境里的）

# 传给 leafFit_segment_root_path.py 的参数
LEAF_FIT_ROOT="third_party/leaf_fit"  # 相对项目根或绝对路径
LEAF_ARGS=(--disable-apex-cluster  --tip-k 32 --path-k 64)

# ---------------- 自动推导（一般不用改） ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"   # 脚本在 scripts/leaf_fit/，项目根往上二级
RUNNER="$SCRIPT_DIR/leafFit_segment_root_path.py"
[[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
[[ "$LEAF_FIT_ROOT" != /* ]] && LEAF_FIT_ROOT="$PROJECT_ROOT/$LEAF_FIT_ROOT"

# ---------------- 前置检查 ----------------
[[ -d "$OUTPUT_DIR" ]] || { echo "[错误] 输出目录不存在: $OUTPUT_DIR" >&2; exit 1; }
[[ -f "$RUNNER" ]] || { echo "[错误] 找不到脚本: $RUNNER" >&2; exit 1; }
[[ -d "$LEAF_FIT_ROOT" ]] || { echo "[错误] 找不到 leaf_fit 目录: $LEAF_FIT_ROOT" >&2; exit 1; }

# ---------------- conda 环境 ----------------
if [[ -n "$CONDA_ENV" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        CONDA_BASE="${CONDA_BASE:-}"
        [[ -n "$CONDA_BASE" ]] || for base in "$HOME/miniconda3" "$HOME/anaconda3" \
                                              "$HOME/miniforge3" "$HOME/miniconda" /opt/conda; do
            if [[ -d "$base" ]]; then CONDA_BASE="$base"; break; fi
        done
        if [[ -n "$CONDA_BASE" ]]; then
            # shellcheck disable=SC1090
            source "$CONDA_BASE/etc/profile.d/conda.sh"
        else
            echo "[错误] 找不到 conda，无法激活 $CONDA_ENV（可设 CONDA_BASE，或用 CONDA_ENV='' 跳过）" >&2
            exit 1
        fi
    fi
    conda activate "$CONDA_ENV"
    echo "==> conda 环境: $CONDA_ENV"
fi

echo "==> 输出根目录: $OUTPUT_DIR"
echo "==> leaf_fit:   $LEAF_FIT_ROOT"
echo

# ---------------- 工具函数 ----------------
find_input() {
    # 优先取场景根下的 plant_only，其次取 target_exports/plant_only
    if [[ -f "$1/plant_only/point_cloud.ply" ]]; then
        echo "$1/plant_only/point_cloud.ply"
    elif [[ -f "$1/target_exports/plant_only/point_cloud.ply" ]]; then
        echo "$1/target_exports/plant_only/point_cloud.ply"
    else
        echo ""
    fi
}

# ---------------- 主体：遍历每个 scene ----------------
mapfile -t scenes < <(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
ok=0; skipped=0; failed=0
for scene in "${scenes[@]}"; do
    name="$(basename "$scene")"
    input_ply="$(find_input "$scene")"

    if [[ -z "$input_ply" ]]; then
        echo "[跳过] $name : 没有找到 plant_only/point_cloud.ply"
        ((skipped++)); continue
    fi

    plant_dir="$(dirname "$input_ply")"
    echo "============================================================"
    echo "[处理] $name"
    echo "      input : $input_ply"
    echo "      output: $plant_dir/leaf_fit_segments"
    echo "============================================================"

    echo "  [leaf_fit] 开始执行 ...（若已运行过将覆盖重跑）"
    if "$PYTHON" "$RUNNER" \
        --input "$input_ply" \
        --leaf-fit-root "$LEAF_FIT_ROOT" \
        "${LEAF_ARGS[@]}"; then
        echo "  [leaf_fit] 完成"; ((ok++))
    else
        echo "  [leaf_fit] 失败（退出码 $?）"; ((failed++))
    fi
    echo
done

# ---------------- 结果汇总 ----------------
echo "============================================================"
echo "批处理结束: 成功 $ok, 跳过 $skipped, 失败 $failed"
[[ $failed -eq 0 ]]
