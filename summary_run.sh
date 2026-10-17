#!/bin/bash
set -o pipefail

project_root="/media/ubuntu2/A/LWL/code/Train/Fast-PGSR"
video_root="/media/ubuntu2/My Passport/LWL/Videos/盆栽_2026_06_25"
data_root="${project_root}/data"

# Fast-PGSR 最终输出根目录
output_root="/media/ubuntu2/My Passport/LWL/PLYOutput/启林北盆栽2026.6.28/大豆"

# 输出日期中的年份固定为 2026；月日从 soybean*_MMDD 中提取
OUTPUT_YEAR="2026"

log_root="${project_root}/logs/batch_pipeline"
conda_base="/home/ubuntu2/下载/enter"

mkdir -p "${data_root}" "${log_root}"
cd "${project_root}" || exit 1

RUN_EXTRACT=0
RUN_MASK=0
RUN_COLMAP=0
RUN_ARUCO=0
RUN_LABEL=0
RUN_TRAIN=0
RUN_RENDER=0
RUN_EXPORT=1

colmap_env="fast-pgsr-old"
aruco_env="Fast-Sam2"
mask_env="Fast-Sam2"
train_env="fast-pgsr-old"

colmap_script="${project_root}/scripts/preprocess/convert.py"
mask_script="${project_root}/scripts/mask_generation/run_rubik_scene_auto.py"
aruco_script="${project_root}/align_aruco.py"
label_script="${project_root}/scripts/mask_generation/label_colmap_points_viewer.py"
train_script="${project_root}/train.py"
render_script="${project_root}/render.py"
export_script="${project_root}/scripts/export_target_gaussians.py"

COLMAP_CAMERA_MODEL="OPENCV"

MASK_SCAN_STEP=5
MASK_REFINE_TOP_K=5
MASK_BOX_THRESHOLD=0.30
MASK_TEXT_THRESHOLD=0.25
MASK_NMS_THRESHOLD=0.72
MASK_DIRECTION="both"
MASK_FILL_HOLE_AREA=0

LABEL_PATCH_SIZE=5
LABEL_TARGET_DILATION=5
LABEL_MIN_VIEWS=3
LABEL_TARGET_CONFIDENCE=0.70
LABEL_BACKGROUND_CONFIDENCE=0.90

GPU_LIST=(0 1 2)
GPU_POLL_INTERVAL=1

ITERATIONS=30000
TRAIN_RESOLUTION=1
TRAIN_NCC_SCALE=0.5
TRAIN_DENSIFICATION_INTERVAL=500
TRAIN_OPACITY_CULL_THRESHOLD=0.05

RENDER_MAX_DEPTH=10.0
RENDER_VOXEL_SIZE=0.01
EXPORT_UNKNOWN_TARGET_THRESHOLD=0.90

if [ ! -f "${conda_base}/etc/profile.d/conda.sh" ]; then
    echo "[FATAL] conda.sh 不存在：${conda_base}/etc/profile.d/conda.sh"
    exit 1
fi

source "${conda_base}/etc/profile.d/conda.sh"

stage_activate()
{
    local env_name="$1"
    echo "[CONDA] activate ${env_name}"
    conda activate "${env_name}" || exit 1
}

stage_deactivate()
{
    conda deactivate >/dev/null 2>&1 || true
}

SKIP_IDS=(2 4 7 8 17 21 22 23 25 26 27 28 37 38 39 40 55 57 58 59)

is_skip_id()
{
    local id="$1" x
    for x in "${SKIP_IDS[@]}"; do
        [ "${id}" -eq "${x}" ] && return 0
    done
    return 1
}

get_soybean_id()
{
    local name raw
    name=$(basename "$1")
    raw=$(echo "${name}" | sed -n 's/^soybean\([0-9][0-9]*\)_.*/\1/p')
    [ -z "${raw}" ] && { echo ""; return; }
    echo $((10#${raw}))
}

scene_from_video()
{
    local video="$1" filename scene_name date
    filename=$(basename "${video}")
    scene_name="${filename%.mp4}"
    date=$(basename "$(dirname "$(dirname "${video}")")")
    echo "${data_root}/${date}_${scene_name}"
}

output_from_video()
{
    local video="$1"
    local filename scene_name date_code month day output_date

    # 例如：soybean13_0810.mp4 -> soybean13_0810
    filename=$(basename "${video}")
    scene_name="${filename%.mp4}"

    # 取场景名最后一个下划线后的 MMDD，例如 0810
    date_code="${scene_name##*_}"
    month="${date_code:0:2}"
    day="${date_code:2:2}"

    # 年份固定，例如：2026.08.10
    output_date="${OUTPUT_YEAR}.${month}.${day}"

    # 最终：PLYOutput/.../大豆/2026.08.10/soybean13_0810
    echo "${output_root}/${output_date}/${scene_name}"
}

VIDEOS=()
SCENES=()
OUTPUTS=()

while IFS= read -r -d '' video; do
    id=$(get_soybean_id "${video}")
    [ -z "${id}" ] && continue
    is_skip_id "${id}" && continue

    VIDEOS+=("${video}")
    SCENES+=("$(scene_from_video "${video}")")
    OUTPUTS+=("$(output_from_video "${video}")")
done < <(
    find "${video_root}" -type f -path "*/大豆/soybean*.mp4" -print0 | sort -z
)

SCENE_COUNT=${#SCENES[@]}

echo "=================================================="
echo "Fast-PGSR Batch Pipeline"
echo "=================================================="
echo "Valid videos : ${#VIDEOS[@]}"
echo "Valid scenes : ${SCENE_COUNT}"
echo "=================================================="

[ "${SCENE_COUNT}" -eq 0 ] && { echo "[FATAL] 没有找到有效视频"; exit 1; }

stage_header()
{
    echo ""
    echo "=================================================="
    echo "$1"
    echo "=================================================="
}

stage_summary()
{
    echo ""
    echo "[$1 SUMMARY] OK=$2 FAIL=$3 SKIP=$4"
}

has_aligned_model()
{
    local scene="$1"
    [ -f "${scene}/sparse/aligned/cameras.bin" ] &&
    [ -f "${scene}/sparse/aligned/images.bin" ] &&
    [ -f "${scene}/sparse/aligned/points3D.bin" ]
}

has_label_result()
{
    local scene="$1"
    [ -f "${scene}/semantic_initialization/point_semantics.npz" ] &&
    [ -f "${scene}/semantic_initialization/points3D_semantic.ply" ]
}

has_train_result()
{
    local output="$1"
    [ -f "${output}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply" ]
}

has_render_result()
{
    local output="$1"
    [ -f "${output}/mesh/mesh_color.ply" ]
}

stage_extract()
{
    stage_header "[1/8] EXTRACT"
    local ok=0 fail=0 i video scene

    for i in "${!VIDEOS[@]}"; do
        video="${VIDEOS[$i]}"
        scene="${SCENES[$i]}"
        mkdir -p "${scene}/input"

        echo "[EXTRACT $((i+1))/${SCENE_COUNT}] $(basename "${video}")"

        ffmpeg -y -i "${video}" -vf "fps=2.5" -q:v 1 \
            "${scene}/input/frame_%06d.jpg" </dev/null

        if [ $? -eq 0 ]; then
            ok=$((ok+1))
        else
            fail=$((fail+1))
            echo "[FAIL] 抽帧失败，跳过：${video}"
        fi
    done

    stage_summary "EXTRACT" "${ok}" "${fail}" 0
}

stage_colmap()
{
    stage_header "[3/8] COLMAP"
    stage_activate "${colmap_env}"

    python "${colmap_script}" \
        --data_path "${data_root}" \
        --camera "${COLMAP_CAMERA_MODEL}"

    ret=$?
    stage_deactivate

    [ "${ret}" -eq 0 ] && echo "[COLMAP OK]" || echo "[COLMAP WARN] 返回非 0，继续后续阶段"
}

stage_mask()
{
    stage_header "[2/8] MASK"
    stage_activate "${mask_env}"

    local ok=0 fail=0 skip=0 i scene rel_scene

    for i in "${!SCENES[@]}"; do
        scene="${SCENES[$i]}"
        rel_scene="${scene#${project_root}/}"

        echo "[MASK $((i+1))/${SCENE_COUNT}] ${rel_scene}"

        if [ ! -d "${scene}/input" ] && [ ! -d "${scene}/images" ]; then
            skip=$((skip+1))
            echo "[SKIP] 缺少 input/images：${scene}"
            continue
        fi

        python "${mask_script}" \
            --scene "${rel_scene}" \
            --scan-step "${MASK_SCAN_STEP}" \
            --refine-top-k "${MASK_REFINE_TOP_K}" \
            --box-threshold "${MASK_BOX_THRESHOLD}" \
            --text-threshold "${MASK_TEXT_THRESHOLD}" \
            --nms-threshold "${MASK_NMS_THRESHOLD}" \
            --direction "${MASK_DIRECTION}" \
            --fill-hole-area "${MASK_FILL_HOLE_AREA}" \
            --overwrite

        if [ $? -eq 0 ]; then
            ok=$((ok+1))
        else
            fail=$((fail+1))
            echo "[FAIL] Mask 失败，跳过：${scene}"
        fi
    done

    stage_deactivate
    stage_summary "MASK" "${ok}" "${fail}" "${skip}"
}

stage_aruco()
{
    stage_header "[4/8] ARUCO ALIGN"
    stage_activate "${aruco_env}"

    local ok=0 fail=0 skip=0 i scene

    for i in "${!SCENES[@]}"; do
        scene="${SCENES[$i]}"
        echo "[ARUCO $((i+1))/${SCENE_COUNT}] ${scene}"

        if [ ! -f "${scene}/sparse/cameras.bin" ] ||
           [ ! -f "${scene}/sparse/images.bin" ] ||
           [ ! -f "${scene}/sparse/points3D.bin" ]; then
            skip=$((skip+1))
            echo "[SKIP] 缺少原始 COLMAP sparse 模型：${scene}"
            continue
        fi

        python "${aruco_script}" "${scene}"

        if [ $? -eq 0 ] && has_aligned_model "${scene}"; then
            ok=$((ok+1))
        else
            fail=$((fail+1))
            echo "[FAIL] ArUco 对齐失败，跳过：${scene}"
        fi
    done

    stage_deactivate
    stage_summary "ARUCO" "${ok}" "${fail}" "${skip}"
}

stage_label()
{
    stage_header "[5/8] LABEL"
    stage_activate "${mask_env}"

    local ok=0 fail=0 skip=0 i scene aligned_dir sparse0_dir

    for i in "${!SCENES[@]}"; do
        scene="${SCENES[$i]}"
        aligned_dir="${scene}/sparse/aligned"
        sparse0_dir="${scene}/sparse/0"

        echo ""
        echo "[LABEL $((i+1))/${SCENE_COUNT}] ${scene}"

        if ! has_aligned_model "${scene}"; then
            skip=$((skip+1))
            echo "[SKIP] 缺少 sparse/aligned：${scene}"
            continue
        fi

        rm -rf "${sparse0_dir}"
        mkdir -p "${sparse0_dir}"

        cp \
            "${aligned_dir}/cameras.bin" \
            "${aligned_dir}/images.bin" \
            "${aligned_dir}/points3D.bin" \
            "${sparse0_dir}/"

        if [ $? -ne 0 ]; then
            fail=$((fail+1))
            echo "[FAIL] aligned -> sparse/0 复制失败：${scene}"
            continue
        fi

        echo "[COPY OK] sparse/aligned -> sparse/0"

        python "${label_script}" \
            --scene "${scene}" \
            --patch-size "${LABEL_PATCH_SIZE}" \
            --target-dilation "${LABEL_TARGET_DILATION}" \
            --min-views "${LABEL_MIN_VIEWS}" \
            --target-confidence "${LABEL_TARGET_CONFIDENCE}" \
            --background-confidence "${LABEL_BACKGROUND_CONFIDENCE}" \
            --no-visualize \
            --overwrite

        if [ $? -eq 0 ] && has_label_result "${scene}"; then
            ok=$((ok+1))
            echo "[LABEL OK] $((i+1))/${SCENE_COUNT}"
        else
            fail=$((fail+1))
            echo "[FAIL] Label 失败，跳过：${scene}"
        fi
    done

    stage_deactivate
    stage_summary "LABEL" "${ok}" "${fail}" "${skip}"
}

# ================================================================
# Bash 5.0 兼容 GPU 动态调度
#
# TRAIN 时：
#   GPU0 / GPU1 / GPU2 各显示一条实时进度条。
#   进度从 train_*.log 中解析 "当前迭代/30000"。
#
# 不使用 Bash 5.1 的 wait -n -p。
# ================================================================

process_finished()
{
    local pid="$1" state

    if ! ps -p "${pid}" >/dev/null 2>&1; then
        return 0
    fi

    state=$(ps -o stat= -p "${pid}" 2>/dev/null | awk '{print $1}')

    [ -z "${state}" ] && return 0

    case "${state}" in
        Z*) return 0 ;;
    esac

    return 1
}


# ================================================================
# 读取某个训练场景当前迭代
#
# Fast-PGSR/train.py 通常由 tqdm 输出类似：
#     12340/30000
#
# train.py 的 stdout/stderr 仍保存到日志，
# 这里仅从日志尾部提取最新迭代数。
# ================================================================

get_train_iteration()
{
    local idx="$1"
    local scene="${SCENES[$idx]}"
    local name
    local log_file
    local current

    name=$(basename "${scene}")
    log_file="${log_root}/train_${name}.log"

    if [ ! -f "${log_file}" ]; then
        echo 0
        return
    fi

    current=$(
        tail -c 200000 "${log_file}" 2>/dev/null \
        | tr '\r' '\n' \
        | grep -aoE "[0-9]+/${ITERATIONS}" \
        | tail -n 1 \
        | cut -d/ -f1
    )

    if ! [[ "${current}" =~ ^[0-9]+$ ]]; then
        current=0
    fi

    if [ "${current}" -gt "${ITERATIONS}" ]; then
        current="${ITERATIONS}"
    fi

    echo "${current}"
}


# ================================================================
# 生成单条文本进度条
# ================================================================

make_progress_bar()
{
    local current="$1"
    local total="$2"
    local width=30

    local percent=0
    local filled=0
    local empty=0

    local left=""
    local right=""

    if [ "${total}" -gt 0 ]; then
        percent=$(( current * 100 / total ))
        filled=$(( current * width / total ))
    fi

    [ "${filled}" -gt "${width}" ] && filled="${width}"

    empty=$(( width - filled ))

    if [ "${filled}" -gt 0 ]; then
        printf -v left "%${filled}s" ""
        left="${left// /#}"
    fi

    if [ "${empty}" -gt 0 ]; then
        printf -v right "%${empty}s" ""
        right="${right// /-}"
    fi

    printf "[%s%s] %3d%% %5d/%d" \
        "${left}" \
        "${right}" \
        "${percent}" \
        "${current}" \
        "${total}"
}


gpu_schedule()
{
    local stage_name="$1"
    local job_func="$2"
    shift 2

    local TASKS=("$@")
    local task_count=${#TASKS[@]}
    local gpu_count=${#GPU_LIST[@]}

    [ "${task_count}" -eq 0 ] && {
        echo "[${stage_name}] 没有可执行任务"
        return 0
    }

    local GPU_PID=()
    local GPU_TASK=()

    local next=0
    local running=0

    local slot gpu task_index pid ret
    local completed_any

    local ok=0
    local fail=0

    local progress_drawn=0
    local FAILED_TASKS=()


    # ------------------------------------------------------------
    # TRAIN 专用：刷新 GPU0/GPU1/GPU2 的实时进度条
    # ------------------------------------------------------------

    draw_train_progress()
    {
        local draw_slot
        local draw_gpu
        local draw_pid
        local draw_idx
        local draw_name
        local draw_iter
        local draw_bar

        # 前一次已经画过，就把光标移动回这几行顶部，
        # 然后覆盖原来的三行，而不是不断刷屏。
        if [ "${progress_drawn}" -eq 1 ]; then
            printf "\033[%dA" "${gpu_count}"
        fi

        for ((draw_slot=0; draw_slot<gpu_count; draw_slot++))
        do
            draw_gpu="${GPU_LIST[$draw_slot]}"
            draw_pid="${GPU_PID[$draw_slot]:-}"

            printf "\033[2K\r"

            if [ -z "${draw_pid}" ]; then

                printf "[GPU%s] %-30s [%s] %3s\n" \
                    "${draw_gpu}" \
                    "IDLE" \
                    "------------------------------" \
                    "-"

                continue
            fi

            draw_idx="${GPU_TASK[$draw_slot]}"
            draw_name=$(basename "${SCENES[$draw_idx]}")
            draw_iter=$(get_train_iteration "${draw_idx}")
            draw_bar=$(make_progress_bar "${draw_iter}" "${ITERATIONS}")

            # 场景名过长时只显示前 30 个字符，避免进度条乱掉。
            printf "[GPU%s] %-30.30s %s\n" \
                "${draw_gpu}" \
                "${draw_name}" \
                "${draw_bar}"
        done

        progress_drawn=1
    }


    # ------------------------------------------------------------
    # 给某张 GPU 启动下一个未分配任务
    # ------------------------------------------------------------

    launch_next()
    {
        local launch_slot="$1"

        [ "${next}" -ge "${task_count}" ] && return 1

        local launch_gpu="${GPU_LIST[$launch_slot]}"
        local launch_task="${TASKS[$next]}"

        "${job_func}" "${launch_task}" "${launch_gpu}" &

        GPU_PID[$launch_slot]="$!"
        GPU_TASK[$launch_slot]="${launch_task}"

        next=$((next + 1))
        running=$((running + 1))

        return 0
    }


    # ------------------------------------------------------------
    # 初始：三张 GPU 各拿一个场景
    # ------------------------------------------------------------

    for ((slot=0; slot<gpu_count; slot++))
    do
        [ "${next}" -ge "${task_count}" ] && break
        launch_next "${slot}"
    done


    # ------------------------------------------------------------
    # 动态轮询
    # ------------------------------------------------------------

    while [ "${running}" -gt 0 ]
    do
        completed_any=0

        for ((slot=0; slot<gpu_count; slot++))
        do
            pid="${GPU_PID[$slot]:-}"

            [ -z "${pid}" ] && continue

            if process_finished "${pid}"; then

                task_index="${GPU_TASK[$slot]}"
                gpu="${GPU_LIST[$slot]}"

                wait "${pid}"
                ret=$?

                GPU_PID[$slot]=""
                GPU_TASK[$slot]=""

                running=$((running - 1))

                if [ "${ret}" -eq 0 ]; then
                    ok=$((ok + 1))
                else
                    fail=$((fail + 1))
                    FAILED_TASKS+=("GPU${gpu}:$(basename "${SCENES[$task_index]}")")
                fi

                # 当前 GPU 一空闲，立即接下一个场景。
                if [ "${next}" -lt "${task_count}" ]; then
                    launch_next "${slot}"
                fi

                completed_any=1
            fi
        done


        if [ "${stage_name}" = "TRAIN" ]; then
            draw_train_progress
        fi


        if [ "${running}" -gt 0 ]; then
            sleep "${GPU_POLL_INTERVAL}"
        fi
    done


    # TRAIN 最后刷新一次，让三张卡显示为 IDLE。
    if [ "${stage_name}" = "TRAIN" ]; then
        draw_train_progress
        echo ""
    fi


    echo "--------------------------------------------------"
    echo "[${stage_name} GPU SUMMARY]"
    echo "OK   : ${ok}"
    echo "FAIL : ${fail}"

    if [ "${#FAILED_TASKS[@]}" -gt 0 ]; then
        echo "Failed tasks:"
        printf "  %s\n" "${FAILED_TASKS[@]}"
    fi

    echo "--------------------------------------------------"

    return 0
}


# ================================================================
# TRAIN JOB
#
# train.py 的原始输出仍完整保存到：
#   logs/batch_pipeline/train_<scene>.log
#
# 终端的三条进度条由 gpu_schedule() 从日志实时解析。
# ================================================================

train_job()
{
    local idx="$1"
    local gpu="$2"

    local scene="${SCENES[$idx]}"
    local output="${OUTPUTS[$idx]}"

    local name
    local log_file

    name=$(basename "${scene}")
    log_file="${log_root}/train_${name}.log"

    mkdir -p "${output}"

    # 清空旧日志，避免读到上一次训练的迭代进度。
    : > "${log_file}"

    {
        echo "=================================================="
        echo "GPU    : ${gpu}"
        echo "Scene  : ${scene}"
        echo "Output : ${output}"
        echo "=================================================="

        CUDA_VISIBLE_DEVICES="${gpu}" \
        PYTHONUNBUFFERED=1 \
        python "${train_script}" \
            -s "${scene}" \
            -m "${output}" \
            --iterations "${ITERATIONS}" \
            --resolution "${TRAIN_RESOLUTION}" \
            --data_device cuda \
            --ncc_scale "${TRAIN_NCC_SCALE}" \
            --densification_interval "${TRAIN_DENSIFICATION_INTERVAL}" \
            --opacity_cull_threshold "${TRAIN_OPACITY_CULL_THRESHOLD}" \
            --exposure_compensation \
            --semantic_pipeline

        ret=$?

        if [ "${ret}" -ne 0 ]; then
            echo "[TRAIN ERROR] python exit code=${ret}"
            exit "${ret}"
        fi

        if ! has_train_result "${output}"; then
            echo "[TRAIN ERROR] 缺少："
            echo "${output}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
            exit 1
        fi

        echo "[TRAIN DONE]"
    } >> "${log_file}" 2>&1
}


stage_train()
{
    stage_header "[6/8] TRAIN - DYNAMIC 3-GPU"

    stage_activate "${train_env}"

    local tasks=()
    local i scene output

    local skip_label=0
    local skip_done=0

    for i in "${!SCENES[@]}"
    do
        scene="${SCENES[$i]}"
        output="${OUTPUTS[$i]}"

        if ! has_label_result "${scene}"; then
            skip_label=$((skip_label + 1))
            echo "[TRAIN SKIP] 缺少 Label：$(basename "${scene}")"
            continue
        fi

        if has_train_result "${output}"; then
            skip_done=$((skip_done + 1))
            echo "[TRAIN SKIP] 已训练：$(basename "${scene}")"
            continue
        fi

        tasks+=("${i}")
    done

    echo ""
    echo "Train tasks        : ${#tasks[@]}"
    echo "Skip missing label : ${skip_label}"
    echo "Skip already done  : ${skip_done}"
    echo "GPU list           : ${GPU_LIST[*]}"
    echo ""
    echo "下面三行分别显示 GPU0 / GPU1 / GPU2 当前训练进度："
    echo ""

    gpu_schedule \
        "TRAIN" \
        train_job \
        "${tasks[@]}"

    stage_deactivate
}


render_job()
{
    local idx="$1"
    local gpu="$2"

    local scene="${SCENES[$idx]}"
    local output="${OUTPUTS[$idx]}"
    local name log_file

    name=$(basename "${scene}")
    log_file="${log_root}/render_${name}.log"

    echo "[RENDER START] GPU${gpu} -> ${name}"

    CUDA_VISIBLE_DEVICES="${gpu}" \
    python "${render_script}" \
        -m "${output}" \
        --max_depth "${RENDER_MAX_DEPTH}" \
        --voxel_size "${RENDER_VOXEL_SIZE}" \
        > "${log_file}" 2>&1

    if [ $? -ne 0 ]; then
        echo "[RENDER ERROR] ${name}"
        echo "               ${log_file}"
        return 1
    fi

    if ! has_render_result "${output}"; then
        echo "[RENDER ERROR] 缺少 mesh/mesh_color.ply：${name}"
        return 1
    fi

    return 0
}

stage_render()
{
    stage_header "[7/8] RENDER - DYNAMIC 3-GPU"
    stage_activate "${train_env}"

    local tasks=()
    local i scene output
    local skip_train=0 skip_done=0

    for i in "${!SCENES[@]}"; do
        scene="${SCENES[$i]}"
        output="${OUTPUTS[$i]}"

        if ! has_train_result "${output}"; then
            skip_train=$((skip_train+1))
            echo "[RENDER SKIP] 缺少 Train：$(basename "${scene}")"
            continue
        fi

        if has_render_result "${output}"; then
            skip_done=$((skip_done+1))
            echo "[RENDER SKIP] 已完成：$(basename "${scene}")"
            continue
        fi

        tasks+=("${i}")
    done

    echo "Render tasks       : ${#tasks[@]}"
    echo "Skip missing train : ${skip_train}"
    echo "Skip already done  : ${skip_done}"
    echo "GPU list           : ${GPU_LIST[*]}"

    gpu_schedule "RENDER" render_job "${tasks[@]}"

    stage_deactivate
}

stage_export()
{
    stage_header "[8/8] EXPORT"
    stage_activate "${train_env}"

    local ok=0 fail=0 skip=0
    local i scene output iteration_dir export_dir

    for i in "${!SCENES[@]}"; do
        scene="${SCENES[$i]}"
        output="${OUTPUTS[$i]}"

        iteration_dir="${output}/point_cloud/iteration_${ITERATIONS}"
        export_dir="${output}/target_exports"

        echo "[EXPORT $((i+1))/${SCENE_COUNT}] $(basename "${scene}")"

        if ! has_train_result "${output}"; then
            skip=$((skip+1))
            echo "[SKIP] 缺少 Train 结果："
            echo "       ${output}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
            continue
        fi

        python "${export_script}" \
            --iteration_dir "${iteration_dir}" \
            --output_dir "${export_dir}" \
            --mode both \
            --unknown_target_threshold "${EXPORT_UNKNOWN_TARGET_THRESHOLD}" \
            --export_per_class \
            --export_debug_cloud \
            --overwrite

        if [ $? -eq 0 ]; then
            ok=$((ok+1))
        else
            fail=$((fail+1))
            echo "[FAIL] Export 失败：${scene}"
        fi
    done

    stage_deactivate
    stage_summary "EXPORT" "${ok}" "${fail}" "${skip}"
}

[ "${RUN_EXTRACT}" -eq 1 ] && stage_extract
[ "${RUN_MASK}" -eq 1 ] && stage_mask
[ "${RUN_COLMAP}" -eq 1 ] && stage_colmap
[ "${RUN_ARUCO}" -eq 1 ] && stage_aruco
[ "${RUN_LABEL}" -eq 1 ] && stage_label
[ "${RUN_TRAIN}" -eq 1 ] && stage_train
[ "${RUN_RENDER}" -eq 1 ] && stage_render
[ "${RUN_EXPORT}" -eq 1 ] && stage_export

echo ""
echo "=================================================="
echo "Fast-PGSR pipeline finished"
echo "=================================================="
