#!/bin/bash
# Historical filename retained; the default device is now GPU 0.
set -uo pipefail
project_root="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
video_dir="${VIDEO_DIR:-${project_root}/videos}"
data_root="${DATA_ROOT:-${project_root}/data}"
output_root="${OUTPUT_ROOT:-${project_root}/output}"
log_root="${LOG_ROOT:-${project_root}/logs}"
colmap_env="${COLMAP_ENV:-fast-pgsr-old}"
aruco_env="${ARUCO_ENV:-Fast-Sam2}"
mask_env="${MASK_ENV:-Fast-Sam2}"
train_env="${TRAIN_ENV:-fast-pgsr-old}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
RUBIK_COUNT="${RUBIK_COUNT:-1}"
PLANT_INDEX="${PLANT_INDEX:-0}"
EXPORT_MODE="${EXPORT_MODE:-strict}"
FORCE="${FORCE:-0}"
ITERATIONS="${ITERATIONS:-30000}"
FPS_TARGET="${FPS_TARGET:-2.5}"
COLMAP_USE_GPU="${COLMAP_USE_GPU:-1}"
EXTRACT_USE_GPU="${EXTRACT_USE_GPU:-1}"
TRAIN_DATA_DEVICE="${TRAIN_DATA_DEVICE:-cuda}"
RENDER_VOXEL_SIZE="${RENDER_VOXEL_SIZE:-0.01}"
TARGET_SPATIAL_RADIUS="${TARGET_SPATIAL_RADIUS:-0.8}"
SCENE="${SCENE:-}"
RUN_EXTRACT="${RUN_EXTRACT:-1}"
RUN_COLMAP="${RUN_COLMAP:-1}"
RUN_MASK="${RUN_MASK:-1}"
RUN_ALIGN="${RUN_ALIGN:-1}"
RUN_LABEL="${RUN_LABEL:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_RENDER="${RUN_RENDER:-1}"
RUN_EXPORT="${RUN_EXPORT:-1}"

[[ "$RUBIK_COUNT" == 1 || "$RUBIK_COUNT" == 2 ]] || { echo "RUBIK_COUNT 必须为 1/2" >&2; exit 1; }
[[ "$FORCE" == 0 || "$FORCE" == 1 ]] || { echo "FORCE 必须为 0/1" >&2; exit 1; }
[[ "$ITERATIONS" =~ ^[1-9][0-9]*$ && "$PLANT_INDEX" =~ ^[0-9]+$ ]] || exit 1
[[ "$EXPORT_MODE" == strict || "$EXPORT_MODE" == both || "$EXPORT_MODE" == conservative ]] || exit 1
[[ -z "$SCENE" || ( "$SCENE" != *'/'* && "$SCENE" != *'\\'* && "$SCENE" != . && "$SCENE" != .. ) ]] || exit 1
for enabled in "$COLMAP_USE_GPU" "$EXTRACT_USE_GPU" "$RUN_EXTRACT" "$RUN_COLMAP" "$RUN_MASK" "$RUN_ALIGN" "$RUN_LABEL" "$RUN_TRAIN" "$RUN_RENDER" "$RUN_EXPORT"; do
    [[ "$enabled" == 0 || "$enabled" == 1 ]] || exit 1
done
cd "$project_root" || exit 1
command -v conda >/dev/null || { echo "找不到 conda" >&2; exit 1; }
mkdir -p "$data_root" "$output_root" "$log_root" || exit 1
project_root=$(pwd -P)
data_root=$(cd "$data_root" && pwd -P)
output_root=$(cd "$output_root" && pwd -P)
log_root=$(cd "$log_root" && pwd -P)
# Roots must not alias, contain one another, or contain the source checkout.
for pair in "$data_root|$output_root" "$output_root|$data_root" "$data_root|$project_root" "$output_root|$project_root"; do
    parent=${pair%%|*}; child=${pair#*|}
    [[ "$child" != "$parent" && "$child" != "$parent/"* ]] || { echo "数据/输出目录重叠或包含源码" >&2; exit 1; }
done
batch_log_root=$(mktemp -d "${log_root}/single-rubik-$(date +%Y%m%d-%H%M%S)-XXXXXX") || exit 1
printf 'Project: %s\nVideos: %s\nData: %s\nOutput: %s\nGPU visible: %s\nRubik count: %s\nExport: %s\n' \
    "$project_root" "$video_dir" "$data_root" "$output_root" "$CUDA_VISIBLE_DEVICES" "$RUBIK_COUNT" "$EXPORT_MODE"
declare -A failed=()
failures=0
run_python() { local env_name="$1"; shift; conda run --no-capture-output -n "$env_name" python "$@"; }
validate() { local env_name="$1" stage="$2"; shift 2; run_python "$env_name" "$project_root/scripts/validate_semantic_inputs.py" "$stage" --scene "$scene_dir" --rubik-count "$RUBIK_COUNT" "$@"; }
model_ready() { local f; for f in cameras.bin images.bin points3D.bin; do [[ -s "$1/$f" ]] || return 1; done; }
# Only delete a verified scene descendant when FORCE was explicitly enabled.
remove_owned() {
    local target="$1" resolved
    [[ "$FORCE" == 1 ]] || { echo "已有产物；请另选目录或显式 FORCE=1: $target" >&2; return 1; }
    resolved=$(realpath -m -- "$target") || return 1
    case "$resolved" in
        "$data_root/$scene_name/"*|"$output_root/$scene_name"|"$output_root/$scene_name/"*) ;;
        *) echo "拒绝删除场景目录以外路径: $resolved" >&2; return 1 ;;
    esac
    [[ ! -L "$target" ]] || { echo "拒绝删除符号链接: $target" >&2; return 1; }
    rm -rf -- "$target"
}

extract_scene() {
    local video="$1"
    [[ ! -L "$scene_dir" ]] || { echo "拒绝符号链接场景" >&2; return 1; }
    if [[ -d "$scene_dir/input" ]]; then
        if [[ "$FORCE" == 0 ]] && find "$scene_dir/input" -maxdepth 1 -name '*.jpg' -print -quit | grep -q .; then
            echo "复用已有抽帧"; return 0
        fi
        remove_owned "$scene_dir/input" || return 1
    fi
    mkdir -p "$scene_dir/input" || return 1
    run_python "$colmap_env" "$project_root/scripts/preprocess/extract_frames.py" \
        --video "$video" --output_dir "$scene_dir/input" --fps "$FPS_TARGET" --gpu_index 0 \
        $( [[ "$EXTRACT_USE_GPU" == 0 ]] && printf '%s' '--no_gpu' ) || return 1
    find "$scene_dir/input" -maxdepth 1 -name '*.jpg' -print -quit | grep -q .
}

process_stage() {
    local stage="$1" output_path="$output_root/$scene_name" stamp f
    case "$stage" in
        Colmap)
            if [[ "$FORCE" == 0 ]] && model_ready "$scene_dir/sparse" && [[ -d "$scene_dir/images" ]]; then
                echo "复用已有 COLMAP 模型"; return 0
            fi
            run_python "$colmap_env" "$project_root/scripts/preprocess/convert.py" \
                --scene_path "$scene_dir" --camera OPENCV --gpu_index 0 \
                $( [[ "$COLMAP_USE_GPU" == 0 ]] && printf '%s' '--no_gpu' ) $( [[ "$FORCE" == 1 ]] && printf '%s' '--overwrite' ) || return 1
            model_ready "$scene_dir/sparse"
            ;;
        Mask)
            if [[ -e "$scene_dir/object_mask" && "$FORCE" == 0 ]]; then
                validate "$mask_env" masks --require-instance-guard; return $?
            fi
            run_python "$mask_env" "$project_root/scripts/mask_generation/run_rubik_scene_auto.py" \
                --scene "$scene_dir" --rubik-count "$RUBIK_COUNT" --plant-index "$PLANT_INDEX" \
                --scan-step 5 --refine-top-k 5 --box-threshold 0.30 --text-threshold 0.25 \
                --nms-threshold 0.72 --direction both --fill-hole-area 0 \
                $( [[ "$FORCE" == 1 ]] && printf '%s' '--overwrite' ) || return 1
            validate "$mask_env" masks --require-instance-guard
            ;;
        Align)
            model_ready "$scene_dir/sparse" || return 1
            stamp=$(sha256sum "$scene_dir/sparse/"{cameras,images,points3D}.bin) || return 1
            if [[ -d "$scene_dir/sparse/aligned" ]]; then
                if [[ "$FORCE" == 0 ]]; then
                    model_ready "$scene_dir/sparse/aligned" && [[ -f "$scene_dir/sparse/aligned/input.sha256" ]] && \
                        [[ "$(cat "$scene_dir/sparse/aligned/input.sha256")" == "$stamp" ]]
                    return $?
                fi
                remove_owned "$scene_dir/sparse/aligned" || return 1
            fi
            run_python "$aruco_env" "$project_root/align_aruco_refine.py" "$scene_dir" || return 1
            model_ready "$scene_dir/sparse/aligned" || return 1
            printf '%s\n' "$stamp" > "$scene_dir/sparse/aligned/input.sha256"
            ;;
        Label)
            validate "$mask_env" masks || return 1
            if [[ -f "$scene_dir/semantic_initialization/point_semantics.npz" && "$FORCE" == 0 ]]; then
                validate "$mask_env" init; return $?
            fi
            # If ALIGN is disabled, an explicitly prepared sparse/0 can be reused.
            if [[ "$RUN_ALIGN" == 1 ]]; then
                model_ready "$scene_dir/sparse/aligned" || return 1
                if [[ -d "$scene_dir/sparse/0" ]]; then
                    if [[ "$FORCE" == 1 ]]; then remove_owned "$scene_dir/sparse/0" || return 1
                    elif [[ -z "$(find "$scene_dir/sparse/0" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
                        : # COLMAP leaves an empty sparse/0 after moving its model.
                    elif model_ready "$scene_dir/sparse/0"; then
                        for f in cameras.bin images.bin points3D.bin; do
                            cmp -s "$scene_dir/sparse/0/$f" "$scene_dir/sparse/aligned/$f" || { echo "已有 sparse/0 与对齐模型不同；需要 FORCE=1"; return 1; }
                        done
                    else echo "已有 sparse/0 不完整；需要 FORCE=1"; return 1
                    fi
                fi
                mkdir -p "$scene_dir/sparse/0" || return 1
                cp "$scene_dir/sparse/aligned/"{cameras,images,points3D}.bin "$scene_dir/sparse/0/" || return 1
            fi
            model_ready "$scene_dir/sparse/0" || return 1
            run_python "$mask_env" "$project_root/scripts/mask_generation/label_colmap_points_viewer.py" \
                --scene "$scene_dir" --patch-size 5 --target-dilation 5 --min-views 3 \
                --target-confidence 0.70 --background-confidence 0.90 --no-visualize \
                $( [[ "$FORCE" == 1 ]] && printf '%s' '--overwrite' ) || return 1
            validate "$mask_env" bind
            ;;
        Train)
            validate "$train_env" init || return 1
            if [[ -d "$output_path" ]]; then
                if [[ "$FORCE" == 0 ]]; then
                    validate "$train_env" trained --model "$output_path" --iteration "$ITERATIONS"; return $?
                fi
                remove_owned "$output_path" || return 1
            fi
            run_python "$train_env" "$project_root/train.py" -s "$scene_dir" -m "$output_path" \
                --iterations "$ITERATIONS" --resolution 1 --data_device "$TRAIN_DATA_DEVICE" --ncc_scale 0.5 \
                --densification_interval 500 --opacity_cull_threshold 0.05 --exposure_compensation \
                --semantic_pipeline || return 1
            validate "$train_env" bind-trained --model "$output_path" --iteration "$ITERATIONS"
            ;;
        Render)
            validate "$train_env" trained --model "$output_path" --iteration "$ITERATIONS" || return 1
            # Render the same cleaned target as the final export, never the full scene.
            if [[ -d "$output_path/mesh" ]]; then remove_owned "$output_path/mesh" || return 1; fi
            run_python "$train_env" "$project_root/render.py" -m "$output_path" \
                --iteration "$ITERATIONS" --max_depth 15.0 --voxel_size "$RENDER_VOXEL_SIZE" \
                --target_ply "$output_path/target_exports/strict/point_cloud.ply"
            ;;
        Export)
            validate "$train_env" trained --model "$output_path" --iteration "$ITERATIONS" || return 1
            if [[ -d "$output_path/target_exports" ]]; then remove_owned "$output_path/target_exports" || return 1; fi
            run_python "$train_env" "$project_root/scripts/export_target_gaussians.py" \
                --iteration_dir "$output_path/point_cloud/iteration_$ITERATIONS" \
                --output_dir "$output_path/target_exports" --mode "$EXPORT_MODE" \
                --rubik-count "$RUBIK_COUNT" --unknown_target_threshold 0.90 --export_per_class --export_debug_cloud \
                --spatial-radius "$TARGET_SPATIAL_RADIUS"
            ;;
    esac
}

# Build the queue without extracting other plants ahead of the current one.
shopt -s nullglob nocaseglob
declare -A scene_video=() queued=()
scene_names=()
if [[ "$RUN_EXTRACT" == 1 ]]; then
    videos=("$video_dir"/*.mp4 "$video_dir"/*.mov)
    ((${#videos[@]})) || { echo "视频目录为空，请设置 VIDEO_DIR 或 RUN_EXTRACT=0" >&2; exit 1; }
    for video in "${videos[@]}"; do
        file_name=$(basename "$video"); scene_name="${file_name%.*}"
        [[ -z "$SCENE" || "$scene_name" == "$SCENE" ]] || continue
        [[ -z "${queued[$scene_name]:-}" ]] || { echo "视频场景名重复: $scene_name" >&2; exit 1; }
        scene_names+=("$scene_name"); queued["$scene_name"]=1; scene_video["$scene_name"]="$video"
    done
fi
# Retain existing scenes with no video, including RUN_EXTRACT=0 workflows.
for scene_dir in "$data_root"/*/; do
    scene_name=$(basename "${scene_dir%/}")
    [[ -z "$SCENE" || "$scene_name" == "$SCENE" ]] || continue
    [[ -z "${queued[$scene_name]:-}" ]] || continue
    scene_names+=("$scene_name"); queued["$scene_name"]=1
done
((${#scene_names[@]})) || { echo "没有匹配的视频/场景" >&2; exit 1; }
selected=0
for scene_name in "${scene_names[@]}"; do
    scene_dir="$data_root/$scene_name"
    selected=$((selected+1))
    [[ ! -L "$scene_dir" ]] || { failed["$scene_name"]=1; failures=$((failures+1)); continue; }
    # All stages of this plant finish before touching the next video's frames.
    for stage in Extract Colmap Mask Align Label Train Export Render; do
        case "$stage" in
            Extract) enabled=$RUN_EXTRACT; [[ -n "${scene_video[$scene_name]:-}" ]] || continue;;
            Colmap) enabled=$RUN_COLMAP;; Mask) enabled=$RUN_MASK;; Align) enabled=$RUN_ALIGN;;
            Label) enabled=$RUN_LABEL;; Train) enabled=$RUN_TRAIN;; Render) enabled=$RUN_RENDER;; Export) enabled=$RUN_EXPORT;;
        esac
        [[ "$enabled" == 1 ]] || continue
        mkdir -p "$batch_log_root/${stage}_Correct" "$batch_log_root/${stage}_Error"
        echo "[$stage] $scene_name"
        if [[ "$stage" == Extract ]]; then
            extract_scene "${scene_video[$scene_name]}" 2>&1 | tee "$batch_log_root/.$stage-$scene_name.log"
        else
            process_stage "$stage" 2>&1 | tee "$batch_log_root/.$stage-$scene_name.log"
        fi
        ret=${PIPESTATUS[0]}
        if [[ "$ret" == 0 ]]; then result=Correct; else result=Error; failed["$scene_name"]=1; failures=$((failures+1)); fi
        mv "$batch_log_root/.$stage-$scene_name.log" "$batch_log_root/${stage}_$result/$scene_name.log"
        [[ "$ret" == 0 ]] || { echo "跳过失败场景的后续阶段: $scene_name"; break; }
    done
done
[[ -z "$SCENE" || "$selected" != 0 ]] || { echo "指定场景没有执行: $SCENE" >&2; exit 1; }
echo "完成；失败阶段数=$failures；日志=$batch_log_root"
echo "目标高斯: $output_root/<scene>/target_exports/strict/point_cloud.ply"
[[ "$failures" == 0 ]]
