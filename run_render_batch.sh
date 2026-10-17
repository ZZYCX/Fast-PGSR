#!/bin/bash
set -uo pipefail

# ============================================================
# 1. 基本配置
# ============================================================

project_root="/media/ubuntu2/A/LWL/code/Train/Fast-PGSR"

# PLYOutput 大根目录（注意这里不是某一天，而是“大豆”总目录）
output_base_root="/media/ubuntu2/My Passport/LWL/PLYOutput/启林北盆栽2026.6.28/大豆"

render_script="${project_root}/render.py"

# Conda 环境
conda_env="fast-pgsr-old"

# 使用哪张 GPU
GPU_ID=1

# Render 参数（与你原脚本保持一致）
MAX_DEPTH=15.0
VOXEL_SIZE=0.01

# 日志总目录
log_root="${project_root}/logs/render_repair_multi_dates"

mkdir -p "${log_root}"

cd "${project_root}" || exit 1


# ============================================================
# 2. 工具函数
# ============================================================

stage_title()
{
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

# 判断是否存在“有效 mesh”
# 只要 mesh_color.ply 或 mesh_nocolor.ply 有一个存在且非空，就认为有效
has_valid_mesh()
{
    local mesh_dir="$1"

    if [ -s "${mesh_dir}/mesh_color.ply" ] || [ -s "${mesh_dir}/mesh_nocolor.ply" ]; then
        return 0
    else
        return 1
    fi
}

# 根据日期返回该日期需要补 Render 的株号列表
get_target_ids()
{
    local date="$1"

    case "${date}" in
        "2026.07.21")
            echo "26"
            ;;
        "2026.07.30")
            echo ""
            ;;
        "2026.08.04")
            echo "4 5 6 7 8 9 13 17 18 25 26 27 28 31 34 36 37 38 39 41 43 51 52 54 55 57 58 59 60"
            ;;
        "2026.08.15")
            echo "3 5 9 10 12 13 14 15 16 18 19 26 27 29 30 31 33 34 36 38 39 40 41 43 51 52 54 55 57 58 59 60"
            ;;
        *)
            echo ""
            ;;
    esac
}


# ============================================================
# 3. 需要处理的日期
# ============================================================

DATES=(
    "2026.07.21"
    "2026.07.30"
    "2026.08.04"
    "2026.08.15"
)


# ============================================================
# 4. 开始补 Render
# ============================================================

stage_title "Fast-PGSR 多日期补 Render"

echo "project_root     : ${project_root}"
echo "output_base_root : ${output_base_root}"
echo "render_script    : ${render_script}"
echo "conda_env        : ${conda_env}"
echo "GPU_ID           : ${GPU_ID}"
echo "MAX_DEPTH        : ${MAX_DEPTH}"
echo "VOXEL_SIZE       : ${VOXEL_SIZE}"
echo "log_root         : ${log_root}"

grand_total=0
grand_ok=0
grand_fail=0
grand_skip=0

for date in "${DATES[@]}"; do

    stage_title "处理日期：${date}"

    ids_str=$(get_target_ids "${date}")

    if [ -z "${ids_str}" ]; then
        echo "该日期没有需要补 Render 的植株，跳过。"
        continue
    fi

    # 从日期提取 MMDD，例如 2026.08.04 -> 0804
    month="${date:5:2}"
    day="${date:8:2}"
    suffix="${month}${day}"

    output_root="${output_base_root}/${date}"
    date_log_root="${log_root}/${date}"
    mkdir -p "${date_log_root}"

    total=0
    ok=0
    fail=0
    skip=0

    for id in ${ids_str}; do
        total=$((total+1))
    done

    echo "输出目录：${output_root}"
    echo "目标数量：${total}"

    idx=0

    for id in ${ids_str}; do
        idx=$((idx+1))

        scene_name="soybean${id}_${suffix}"
        output_path="${output_root}/${scene_name}"

        iter_dir="${output_path}/point_cloud/iteration_30000"
        final_ply="${iter_dir}/point_cloud.ply"

        mesh_dir="${output_path}/mesh"
        log_file="${date_log_root}/${scene_name}.log"

        echo
        echo "------------------------------------------------------------"
        echo "[RENDER ${idx}/${total}] ${scene_name}"
        echo "------------------------------------------------------------"

        # --------------------------------------------------------
        # 1) 检查场景输出目录是否存在
        # --------------------------------------------------------
        if [ ! -d "${output_path}" ]; then
            echo "❌ 场景目录不存在：${output_path}"

            {
                echo "RESULT: SKIP"
                echo "REASON: output directory does not exist"
                echo "OUTPUT: ${output_path}"
            } > "${log_file}"

            skip=$((skip+1))
            grand_skip=$((grand_skip+1))
            continue
        fi

        # --------------------------------------------------------
        # 2) 检查是否有训练结果
        # --------------------------------------------------------
        if [ ! -f "${final_ply}" ]; then
            echo "❌ 缺少训练结果：${final_ply}"

            {
                echo "RESULT: SKIP"
                echo "REASON: missing iteration_30000 point_cloud.ply"
                echo "OUTPUT: ${output_path}"
            } > "${log_file}"

            skip=$((skip+1))
            grand_skip=$((grand_skip+1))
            continue
        fi

        # --------------------------------------------------------
        # 3) 如果已经有有效 mesh，则跳过
        # --------------------------------------------------------
        if has_valid_mesh "${mesh_dir}"; then
            echo "✅ 已存在有效 mesh，跳过"

            {
                echo "RESULT: SKIP"
                echo "REASON: valid mesh already exists"
                echo "MESH_DIR: ${mesh_dir}"
            } > "${log_file}"

            skip=$((skip+1))
            grand_skip=$((grand_skip+1))
            continue
        fi

        # --------------------------------------------------------
        # 4) 删除旧的失败 mesh 残留
        # --------------------------------------------------------
        if [ -d "${mesh_dir}" ]; then
            echo "[CLEAN] 删除旧 mesh 目录：${mesh_dir}"
            rm -rf "${mesh_dir}"
        fi

        # --------------------------------------------------------
        # 5) 执行 Render
        # --------------------------------------------------------
        echo "[START] Render ${scene_name}"

        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        conda run \
            --no-capture-output \
            -n "${conda_env}" \
            python "${render_script}" \
                -m "${output_path}" \
                --max_depth "${MAX_DEPTH}" \
                --voxel_size "${VOXEL_SIZE}" \
            2>&1 | tee "${log_file}"

        ret=${PIPESTATUS[0]}

        # --------------------------------------------------------
        # 6) 检查 Render 结果
        # --------------------------------------------------------
        if [ "${ret}" -eq 0 ] && has_valid_mesh "${mesh_dir}"; then
            echo "✅ ${scene_name} Render 成功"

            {
                echo ""
                echo "RESULT: SUCCESS"
                echo "MESH_DIR: ${mesh_dir}"
            } >> "${log_file}"

            ok=$((ok+1))
            grand_ok=$((grand_ok+1))

        else
            echo "❌ ${scene_name} Render 失败"

            {
                echo ""
                echo "RESULT: FAILED"
                echo "EXIT_CODE: ${ret}"
                echo "EXPECTED_MESH_DIR: ${mesh_dir}"
            } >> "${log_file}"

            fail=$((fail+1))
            grand_fail=$((grand_fail+1))
        fi
    done

    grand_total=$((grand_total+total))

    echo
    echo "============================================================"
    echo "${date} 补 Render 完成"
    echo "总数   : ${total}"
    echo "成功   : ${ok}"
    echo "失败   : ${fail}"
    echo "跳过   : ${skip}"
    echo "日志   : ${date_log_root}"
    echo "============================================================"
done


# ============================================================
# 5. 总汇总
# ============================================================

echo
echo "============================================================"
echo "Fast-PGSR 多日期补 Render 结束"
echo "============================================================"
echo "目标总数 : ${grand_total}"
echo "成功     : ${grand_ok}"
echo "失败     : ${grand_fail}"
echo "跳过     : ${grand_skip}"
echo "日志根目录: ${log_root}"
echo "============================================================"