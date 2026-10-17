#!/usr/bin/env bash
# ============================================================
# 批处理：对数据目录下每个 scene 依次执行
#   1. mask  : run_rubik_scene_auto.py        -> 生成 object_mask/ + color_mask/
#   2. label : label_colmap_points_viewer.py  -> 生成 semantic_initialization/
#
# 每阶段执行前先判断是否已生成，已生成则跳过（幂等，可反复运行）。
#
# 用法:
#   ./batch_run_rubik.sh [数据根目录]      # 第一个参数覆盖下方 DATA_DIR
# ============================================================
set -uo pipefail

# ---------------- 配置区（需要调整时只改这里） ----------------
CONDA_ENV="${CONDA_ENV:-ObjectGS_GSAM2}"   # 运行前激活的 conda 环境；设为空字符串则跳过
DATA_DIR="${1:-data}"                      # 数据根目录：其下每个子目录是一个 scene
STEPS="${STEPS:-mask,label}"               # 执行哪些阶段：mask / label / mask,label
PYTHON="${PYTHON:-python}"                 # python 解释器（默认取 conda 环境里的）
FORCE=0                                    # =1 时不检查是否已完成，全部重跑

# mask 阶段参数（传给 run_rubik_scene_auto.py）
MASK_ARGS=(--fill-hole-area 0 --overwrite)
# label 阶段参数（传给 label_colmap_points_viewer.py）
# 批处理默认 --no-visualize：避免每个 scene 弹三维窗口阻塞；想交互检查时改回 --visualize
LABEL_ARGS=(--patch-size 5 --target-dilation 5 --min-views 3 \
            --target-confidence 0.70 --background-confidence 0.90 \
            --no-visualize --overwrite)

# 各阶段"已完成"的判定条件（满足即跳过）：
#   mask : color_mask/ 目录里有文件
#   label: semantic_initialization/point_semantics.npz 已存在

# ---------------- 自动推导（一般不用改） ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # 脚本在 bash_tasks/，项目根往上三级
MASK_RUNNER="$SCRIPT_DIR/../run_rubik_scene_auto.py"
LABEL_RUNNER="$SCRIPT_DIR/../label_colmap_points_viewer.py"
[[ "$DATA_DIR" != /* ]] && DATA_DIR="$PROJECT_ROOT/$DATA_DIR"

# ---------------- 前置检查 ----------------
[[ -d "$DATA_DIR" ]] || { echo "[错误] 数据目录不存在: $DATA_DIR" >&2; exit 1; }
IFS=',' read -r -a stages <<< "$STEPS"
for s in "${stages[@]}"; do
    case "$s" in
        mask|label) ;;
        *) echo "[错误] 未知阶段: $s（可选 mask / label / mask,label）" >&2; exit 1 ;;
    esac
done
for s in "${stages[@]}"; do
    runner=""
    [[ "$s" == "mask"  ]] && runner="$MASK_RUNNER"
    [[ "$s" == "label" ]] && runner="$LABEL_RUNNER"
    [[ -f "$runner" ]] || { echo "[错误] 找不到脚本: $runner" >&2; exit 1; }
done

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

echo "==> 数据根目录: $DATA_DIR"
echo "==> 执行阶段:   $STEPS"
echo

# ---------------- 工具函数 ----------------
mask_done()  { [[ -n "$(find "$1/color_mask" -maxdepth 1 -type f 2>/dev/null | head -1)" ]]; }
label_done() { [[ -f "$1/semantic_initialization/point_semantics.npz" ]]; }

run_one() {
    local stage="$1" scene_rel="$2"
    case "$stage" in
        mask)  "$PYTHON" "$MASK_RUNNER"  --scene "$scene_rel" "${MASK_ARGS[@]}" ;;
        label) "$PYTHON" "$LABEL_RUNNER" --scene "$scene_rel" "${LABEL_ARGS[@]}" ;;
        *)     echo "[错误] 未知阶段: $stage" >&2; return 1 ;;
    esac
}

# ---------------- 主体：遍历每个 scene，逐阶段执行 ----------------
mapfile -t scenes < <(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
ok=0; skipped=0; failed=0
for scene in "${scenes[@]}"; do
    name="$(basename "$scene")"
    if [[ ! -d "$scene/images" ]]; then
        echo "[跳过] $name : 缺少 images 子目录"
        ((skipped++)); continue
    fi
    rel="${scene#"$PROJECT_ROOT"/}"   # 相对项目根路径，处理脚本据此解析场景
    echo "============================================================"
    echo "[处理] $rel"
    echo "============================================================"
    for stage in "${stages[@]}"; do
        if (( FORCE == 0 )) && "${stage}_done" "$scene"; then
            echo "  [$stage] 已完成，跳过"
            ((skipped++)); continue
        fi
        echo "  [$stage] 开始执行 ..."
        if run_one "$stage" "$rel"; then
            echo "  [$stage] 完成"; ((ok++))
        else
            echo "  [$stage] 失败（退出码 $?）"; ((failed++))
        fi
    done
    echo
done

# ---------------- 结果汇总 ----------------
echo "============================================================"
echo "批处理结束: 成功 $ok, 跳过 $skipped, 失败 $failed"
[[ $failed -eq 0 ]]
