#!/usr/bin/env bash

# ==============================================================================
# Fast-PGSR：Grounded-SAM2 + SAM2 批量场景处理脚本
#
# 功能：
#   1. 不修改原有 Python 核心代码；
#   2. 批量调用 run_rubik_scene_auto.py；
#   3. 自动遍历 data/ 下包含 images/ 的场景；
#   4. 单个场景失败后继续处理后续场景；
#   5. 将成功、失败和跳过场景分别写入：
#        log/mask_generation/success_list.txt
#        log/mask_generation/fail_list.txt
#        log/mask_generation/skip_list.txt
#   6. 每个场景保存独立运行日志；
#   7. Python 返回成功后，再检查 object_mask 和 color_mask 是否完整。
# ==============================================================================

# 不使用 set -e：
# 某一个场景失败时，批处理不能整体退出，必须继续处理后面的场景。
set -uo pipefail


# ==============================================================================
# 1. 路径配置
# ==============================================================================

# 当前脚本应放在：
#   Fast-PGSR/scripts/mask_generation/run_batch_rubik_scene.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_ROOT="$PROJECT_ROOT/data"
PY_SCRIPT="$SCRIPT_DIR/run_rubik_scene_auto.py"

# 按用户要求，日志统一保存在：
#   Fast-PGSR/log/mask_generation/
LOG_DIR="$PROJECT_ROOT/log/mask_generation"
SCENE_LOG_DIR="$LOG_DIR/scene_logs"
HISTORY_DIR="$LOG_DIR/history"

SUCCESS_LIST="$LOG_DIR/success_list.txt"
FAIL_LIST="$LOG_DIR/fail_list.txt"
FAIL_DETAIL="$LOG_DIR/fail_detail.txt"
SKIP_LIST="$LOG_DIR/skip_list.txt"
SUMMARY_FILE="$LOG_DIR/batch_summary.txt"

RUN_ID="$(date '+%Y%m%d_%H%M%S')"
RUN_START_TEXT="$(date '+%Y-%m-%d %H:%M:%S')"


# ==============================================================================
# 2. Grounded-SAM2 / SAM2 运行参数
# ==============================================================================

PROMPT="single green leafy plant. rubik's cube with attached ArUco marker. plot."

# 每隔多少张图片扫描一次关键帧候选。
SCAN_STEP=5

# 快速评分后，使用 SAM2 精细评价的候选帧数量。
REFINE_TOP_K=5

# Grounding DINO 检测框阈值。
BOX_THRESHOLD=0.30

# Grounding DINO 文本匹配阈值。
TEXT_THRESHOLD=0.25

# 同类别重复检测框的 NMS IoU 阈值。
NMS_THRESHOLD=0.72

# both：从关键帧向前、向后传播，生成完整序列 Mask。
# forward：只向序列末尾传播。
DIRECTION="both"

# true：运行时增加 --overwrite，重新生成该场景结果。
# false：输出已存在时，由 Python 脚本决定是否报错。
OVERWRITE=true

# true：如果 object_mask 和 color_mask 已经完整，则直接跳过该场景。
# false：不预先跳过；配合 OVERWRITE=true 时会重新处理。
SKIP_COMPLETED=false

# true：把 SAM2 视频状态转移到 CPU，降低显存占用，但速度会变慢。
# RTX 3090 Ti 24GB 通常保持 false。
OFFLOAD_STATE_TO_CPU=false

# true：控制台实时显示 Python 内部阶段和进度条，同时写入日志。
# false：控制台只显示批处理级别信息，Python 输出只写日志。
SHOW_LIVE_OUTPUT=true


# ==============================================================================
# 3. 工具函数
# ==============================================================================

mkdir -p "$LOG_DIR" "$SCENE_LOG_DIR" "$HISTORY_DIR"

# 每次运行先清空“本次运行”的结果列表。
# 历史结果会在最后复制到 history/ 中，不会丢失。
: > "$SUCCESS_LIST"
: > "$FAIL_LIST"
: > "$FAIL_DETAIL"
: > "$SKIP_LIST"
: > "$SUMMARY_FILE"


count_source_images() {
    local image_dir="$1"

    find "$image_dir" -maxdepth 1 -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
        -printf '.' 2>/dev/null | wc -c
}


count_png_files() {
    local directory="$1"

    if [[ ! -d "$directory" ]]; then
        echo 0
        return
    fi

    find "$directory" -maxdepth 1 -type f -iname '*.png' \
        -printf '.' 2>/dev/null | wc -c
}


append_success() {
    local scene="$1"
    printf '%s\n' "$scene" >> "$SUCCESS_LIST"
}


append_failure() {
    local scene="$1"
    local reason="$2"

    # fail_list.txt 只保存场景名，方便后续直接批量重跑。
    printf '%s\n' "$scene" >> "$FAIL_LIST"

    # fail_detail.txt 保存失败原因。
    printf '%s\t%s\n' "$scene" "$reason" >> "$FAIL_DETAIL"
}


append_skip() {
    local scene="$1"
    local reason="$2"

    printf '%s\t%s\n' "$scene" "$reason" >> "$SKIP_LIST"
}


format_duration() {
    local total_seconds="$1"
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))

    printf '%02d:%02d:%02d' "$hours" "$minutes" "$seconds"
}


validate_scene_outputs() {
    local scene_dir="$1"
    local expected_count="$2"

    local object_dir="$scene_dir/object_mask"
    local color_dir="$scene_dir/color_mask"

    if [[ ! -d "$object_dir" ]]; then
        VALIDATION_REASON="缺少 object_mask 目录"
        return 1
    fi

    if [[ ! -d "$color_dir" ]]; then
        VALIDATION_REASON="缺少 color_mask 目录"
        return 1
    fi

    local object_count
    local color_count
    object_count="$(count_png_files "$object_dir")"
    color_count="$(count_png_files "$color_dir")"

    if [[ "$object_count" -eq 0 ]]; then
        VALIDATION_REASON="object_mask 中没有 PNG 文件"
        return 1
    fi

    if [[ "$color_count" -eq 0 ]]; then
        VALIDATION_REASON="color_mask 中没有 PNG 文件"
        return 1
    fi

    if [[ "$DIRECTION" == "both" ]]; then
        if [[ "$object_count" -ne "$expected_count" ]]; then
            VALIDATION_REASON="object_mask 数量不完整：${object_count}/${expected_count}"
            return 1
        fi

        if [[ "$color_count" -ne "$expected_count" ]]; then
            VALIDATION_REASON="color_mask 数量不完整：${color_count}/${expected_count}"
            return 1
        fi
    fi

    if [[ "$object_count" -ne "$color_count" ]]; then
        VALIDATION_REASON="object_mask 与 color_mask 数量不一致：${object_count}/${color_count}"
        return 1
    fi

    VALIDATION_REASON=""
    return 0
}


write_summary() {
    local elapsed_seconds="$1"
    local end_text
    end_text="$(date '+%Y-%m-%d %H:%M:%S')"

    {
        echo "Fast-PGSR 批量 Mask 生成结果"
        echo "========================================"
        echo "运行编号:       $RUN_ID"
        echo "开始时间:       $RUN_START_TEXT"
        echo "结束时间:       $end_text"
        echo "总耗时:         $(format_duration "$elapsed_seconds")"
        echo "扫描场景数:     $TOTAL"
        echo "成功场景数:     $SUCCESS"
        echo "失败场景数:     $FAIL"
        echo "跳过场景数:     $SKIP"
        echo
        echo "参数："
        echo "  scan_step:             $SCAN_STEP"
        echo "  refine_top_k:          $REFINE_TOP_K"
        echo "  box_threshold:         $BOX_THRESHOLD"
        echo "  text_threshold:        $TEXT_THRESHOLD"
        echo "  nms_threshold:         $NMS_THRESHOLD"
        echo "  direction:             $DIRECTION"
        echo "  overwrite:             $OVERWRITE"
        echo "  skip_completed:        $SKIP_COMPLETED"
        echo "  offload_state_to_cpu:  $OFFLOAD_STATE_TO_CPU"
        echo
        echo "结果文件："
        echo "  success_list: $SUCCESS_LIST"
        echo "  fail_list:    $FAIL_LIST"
        echo "  fail_detail:  $FAIL_DETAIL"
        echo "  skip_list:    $SKIP_LIST"
        echo "  scene_logs:   $SCENE_LOG_DIR"
    } > "$SUMMARY_FILE"

    # 保存一份带时间戳的历史记录。
    cp "$SUCCESS_LIST" "$HISTORY_DIR/${RUN_ID}_success_list.txt"
    cp "$FAIL_LIST" "$HISTORY_DIR/${RUN_ID}_fail_list.txt"
    cp "$FAIL_DETAIL" "$HISTORY_DIR/${RUN_ID}_fail_detail.txt"
    cp "$SKIP_LIST" "$HISTORY_DIR/${RUN_ID}_skip_list.txt"
    cp "$SUMMARY_FILE" "$HISTORY_DIR/${RUN_ID}_batch_summary.txt"
}


print_final_summary() {
    local elapsed_seconds="$1"

    echo
    echo "============================================================"
    echo "批量处理结束"
    echo "============================================================"
    echo "总场景数:   $TOTAL"
    echo "成功场景数: $SUCCESS"
    echo "失败场景数: $FAIL"
    echo "跳过场景数: $SKIP"
    echo "总耗时:     $(format_duration "$elapsed_seconds")"
    echo
    echo "成功列表:   $SUCCESS_LIST"
    echo "失败列表:   $FAIL_LIST"
    echo "失败详情:   $FAIL_DETAIL"
    echo "跳过列表:   $SKIP_LIST"
    echo "批次汇总:   $SUMMARY_FILE"
    echo "场景日志:   $SCENE_LOG_DIR"
    echo "============================================================"
}


INTERRUPTED=false

handle_interrupt() {
    INTERRUPTED=true
    echo
    echo "[中断] 收到终止信号，正在保存当前批次结果……"

    local elapsed_seconds=$((SECONDS))
    write_summary "$elapsed_seconds"
    print_final_summary "$elapsed_seconds"

    exit 130
}

trap handle_interrupt INT TERM


# ==============================================================================
# 4. 运行前检查
# ==============================================================================

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "[错误] 数据目录不存在：$DATA_ROOT" >&2
    exit 1
fi

if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "[错误] Python 脚本不存在：$PY_SCRIPT" >&2
    exit 1
fi

if ! command -v python >/dev/null 2>&1; then
    echo "[错误] 当前环境找不到 python，请先激活 ObjectGS_GSAM2。" >&2
    exit 1
fi


# ==============================================================================
# 5. 收集待处理场景
# ==============================================================================

declare -a SCENES=()

# 给 shell 脚本传入场景名时，只处理指定场景：
#   bash run_batch_rubik_scene.sh scene_a scene_b
if [[ "$#" -gt 0 ]]; then
    SCENES=("$@")
else
    # 不传参数时，自动遍历 data/ 下所有一级目录。
    shopt -s nullglob
    for scene_path in "$DATA_ROOT"/*; do
        [[ -d "$scene_path" ]] || continue
        SCENES+=("$(basename "$scene_path")")
    done
    shopt -u nullglob
fi

TOTAL="${#SCENES[@]}"
SUCCESS=0
FAIL=0
SKIP=0

if [[ "$TOTAL" -eq 0 ]]; then
    echo "[错误] 没有找到待处理场景。" >&2
    exit 1
fi


# ==============================================================================
# 6. 批量执行
# ==============================================================================

echo "============================================================"
echo "Fast-PGSR 批量 Grounded-SAM2 + SAM2 处理"
echo "============================================================"
echo "项目目录:       $PROJECT_ROOT"
echo "数据目录:       $DATA_ROOT"
echo "待处理场景数:   $TOTAL"
echo "运行编号:       $RUN_ID"
echo "日志目录:       $LOG_DIR"
echo "============================================================"

BATCH_START_SECONDS=$SECONDS

for ((scene_pos = 0; scene_pos < TOTAL; scene_pos++)); do
    SCENE_NAME="${SCENES[$scene_pos]}"
    SCENE_DIR="$DATA_ROOT/$SCENE_NAME"
    IMAGES_DIR="$SCENE_DIR/images"

    # 避免日志文件名中出现斜杠或空格。
    SAFE_SCENE_NAME="${SCENE_NAME//\//_}"
    SAFE_SCENE_NAME="${SAFE_SCENE_NAME// /_}"
    SCENE_LOG="$SCENE_LOG_DIR/${RUN_ID}_${SAFE_SCENE_NAME}.log"

    echo
    echo "------------------------------------------------------------"
    echo "[$((scene_pos + 1))/$TOTAL] 场景：$SCENE_NAME"
    echo "------------------------------------------------------------"

    if [[ ! -d "$SCENE_DIR" ]]; then
        REASON="场景目录不存在：$SCENE_DIR"
        echo "[跳过] $REASON"
        append_skip "$SCENE_NAME" "$REASON"
        SKIP=$((SKIP + 1))
        continue
    fi

    if [[ ! -d "$IMAGES_DIR" ]]; then
        REASON="缺少 images 目录"
        echo "[跳过] $REASON"
        append_skip "$SCENE_NAME" "$REASON"
        SKIP=$((SKIP + 1))
        continue
    fi

    IMAGE_COUNT="$(count_source_images "$IMAGES_DIR")"
    if [[ "$IMAGE_COUNT" -eq 0 ]]; then
        REASON="images 目录中没有 JPG/JPEG/PNG"
        echo "[跳过] $REASON"
        append_skip "$SCENE_NAME" "$REASON"
        SKIP=$((SKIP + 1))
        continue
    fi

    if [[ "$SKIP_COMPLETED" == true ]]; then
        if validate_scene_outputs "$SCENE_DIR" "$IMAGE_COUNT"; then
            REASON="object_mask 和 color_mask 已完整，按配置跳过"
            echo "[跳过] $REASON"
            append_skip "$SCENE_NAME" "$REASON"
            SKIP=$((SKIP + 1))
            continue
        fi
    fi

    echo "[开始] 图片数量：$IMAGE_COUNT"
    echo "[日志] $SCENE_LOG"

    COMMAND=(
        python "$PY_SCRIPT"
        --scene "$SCENE_NAME"
        --prompt "$PROMPT"
        --scan-step "$SCAN_STEP"
        --refine-top-k "$REFINE_TOP_K"
        --box-threshold "$BOX_THRESHOLD"
        --text-threshold "$TEXT_THRESHOLD"
        --nms-threshold "$NMS_THRESHOLD"
        --direction "$DIRECTION"
    )

    if [[ "$OVERWRITE" == true ]]; then
        COMMAND+=(--overwrite)
    fi

    if [[ "$OFFLOAD_STATE_TO_CPU" == true ]]; then
        COMMAND+=(--offload-state-to-cpu)
    fi

    {
        echo "运行编号: $RUN_ID"
        echo "场景: $SCENE_NAME"
        echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "图片数量: $IMAGE_COUNT"
        echo "命令:"
        printf '  %q' "${COMMAND[@]}"
        echo
        echo "============================================================"
    } > "$SCENE_LOG"

    SCENE_START_SECONDS=$SECONDS
    PYTHON_OK=false

    if [[ "$SHOW_LIVE_OUTPUT" == true ]]; then
        # PIPESTATUS[0] 是 Python 命令的退出码，不能只看 tee 的退出码。
        "${COMMAND[@]}" 2>&1 | tee -a "$SCENE_LOG"
        PYTHON_EXIT_CODE="${PIPESTATUS[0]}"

        if [[ "$PYTHON_EXIT_CODE" -eq 0 ]]; then
            PYTHON_OK=true
        fi
    else
        if "${COMMAND[@]}" >> "$SCENE_LOG" 2>&1; then
            PYTHON_OK=true
        fi
    fi

    SCENE_ELAPSED=$((SECONDS - SCENE_START_SECONDS))

    if [[ "$PYTHON_OK" != true ]]; then
        REASON="Python 运行失败，详见日志：$SCENE_LOG"
        echo "[失败] $SCENE_NAME，耗时 $(format_duration "$SCENE_ELAPSED")"
        append_failure "$SCENE_NAME" "$REASON"
        FAIL=$((FAIL + 1))

        {
            echo
            echo "结果: FAILED"
            echo "原因: $REASON"
            echo "耗时: $(format_duration "$SCENE_ELAPSED")"
        } >> "$SCENE_LOG"

        continue
    fi

    # Python 返回 0 后继续检查输出文件，避免出现“程序成功但结果不完整”。
    if ! validate_scene_outputs "$SCENE_DIR" "$IMAGE_COUNT"; then
        REASON="输出校验失败：$VALIDATION_REASON"
        echo "[失败] $REASON"
        append_failure "$SCENE_NAME" "$REASON"
        FAIL=$((FAIL + 1))

        {
            echo
            echo "结果: FAILED"
            echo "原因: $REASON"
            echo "耗时: $(format_duration "$SCENE_ELAPSED")"
        } >> "$SCENE_LOG"

        continue
    fi

    append_success "$SCENE_NAME"
    SUCCESS=$((SUCCESS + 1))

    echo "[成功] $SCENE_NAME，耗时 $(format_duration "$SCENE_ELAPSED")"

    {
        echo
        echo "结果: SUCCESS"
        echo "耗时: $(format_duration "$SCENE_ELAPSED")"
        echo "object_mask: $SCENE_DIR/object_mask"
        echo "color_mask:  $SCENE_DIR/color_mask"
    } >> "$SCENE_LOG"
done


# ==============================================================================
# 7. 保存并显示批次汇总
# ==============================================================================

BATCH_ELAPSED=$((SECONDS - BATCH_START_SECONDS))
write_summary "$BATCH_ELAPSED"
print_final_summary "$BATCH_ELAPSED"

# 只要有失败场景，脚本整体返回非零状态。
# 这样在 nohup、CI 或外层任务系统中也能识别本批次是否完全成功。
if [[ "$FAIL" -gt 0 ]]; then
    exit 2
fi

exit 0


# ==============================================================================
# 使用说明与脚本总结
# ==============================================================================
#
# 【这个脚本做什么】
#
#   它是 run_rubik_scene_auto.py 的外层批处理器，不修改任何 Python
#   关键帧选择、候选 Mask 选择或 SAM2 传播逻辑。它只负责逐个场景调用、
#   记录日志、继续处理失败后的后续场景，以及检查最终输出是否完整。
#
#
# 【脚本应该放在哪里】
#
#   Fast-PGSR/scripts/mask_generation/run_batch_rubik_scene.sh
#
#   同一目录中应存在：
#
#   Fast-PGSR/scripts/mask_generation/run_rubik_scene_auto.py
#   Fast-PGSR/scripts/mask_generation/run_grounded_sam2_scene.py
#
#
# 【首次使用】
#
#   conda activate ObjectGS_GSAM2
#   cd /media/sx639-2/D/LWL/codes/Train/Fast-PGSR
#   chmod +x scripts/mask_generation/run_batch_rubik_scene.sh
#
#
# 【处理 data/ 下全部场景】
#
#   bash scripts/mask_generation/run_batch_rubik_scene.sh
#
#
# 【只处理指定场景】
#
#   bash scripts/mask_generation/run_batch_rubik_scene.sh \
#       mungbean10_0730 \
#       mungbean11_0730
#
#
# 【后台运行】
#
#   nohup bash scripts/mask_generation/run_batch_rubik_scene.sh \
#       > log/mask_generation/nohup_batch.log 2>&1 &
#
#   查看后台输出：
#
#   tail -f log/mask_generation/nohup_batch.log
#
#
# 【结果目录】
#
#   每个场景最终应生成：
#
#   data/<scene>/object_mask/
#       单通道 uint8 标签 Mask，像素值为 0、1、2、3。
#
#   data/<scene>/color_mask/
#       彩色可视化 Mask。
#
#   data/<scene>/grounded_sam2_scene/
#       初始化 Mask、逐帧预览、质量报告和 summary.json。
#
#
# 【批处理日志】
#
#   log/mask_generation/success_list.txt
#       本次运行成功的场景名，每行一个。
#
#   log/mask_generation/fail_list.txt
#       本次运行失败的场景名，每行一个，方便直接重跑。
#
#   log/mask_generation/fail_detail.txt
#       失败场景和对应原因。
#
#   log/mask_generation/skip_list.txt
#       因目录缺失、图片为空或已完成而跳过的场景。
#
#   log/mask_generation/batch_summary.txt
#       本次批处理的参数、时间和数量汇总。
#
#   log/mask_generation/scene_logs/
#       每个场景的完整独立日志。
#
#   log/mask_generation/history/
#       每次批处理的成功、失败、跳过列表及汇总历史。
#
#
# 【只重跑上一次失败场景】
#
#   mapfile -t FAILED_SCENES \
#       < log/mask_generation/fail_list.txt
#
#   bash scripts/mask_generation/run_batch_rubik_scene.sh \
#       "${FAILED_SCENES[@]}"
#
#
# 【常用配置】
#
#   SCAN_STEP=5
#       每 5 帧扫描一次关键帧候选。增大可提速，减小可提高搜索密度。
#
#   REFINE_TOP_K=5
#       对快速评分最高的 5 张候选帧使用 SAM2 精细评价。
#
#   OVERWRITE=true
#       重新生成 object_mask、color_mask 和 grounded_sam2_scene。
#
#   SKIP_COMPLETED=false
#       不跳过已有完整结果。改成 true 后，只处理未完成场景。
#
#   SHOW_LIVE_OUTPUT=true
#       控制台显示每个 Python 脚本的阶段和进度条，并同时写入独立日志。
#
#
# 【退出码】
#
#   0：所有待运行场景均成功，或只有跳过场景；
#   1：批处理脚本自身配置或路径错误；
#   2：至少有一个场景处理失败；
#   130：用户按 Ctrl+C 或收到终止信号。
#
# ==============================================================================