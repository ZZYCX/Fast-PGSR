#!/bin/bash
set -uo pipefail

SCRIPT_VERSION="2026-09-22-project-tmp-v5"

# ============================================================
# Leaf-only Gaussian -> PGSR TSDF Mesh
#
# ONLY:
#   soybean9_0815
#   soybean10_0815
#   soybean12_0815
#
# Input camera/training information:
#   /media/ubuntu2/A/LWL/code/Train/Fast-PGSR/data/data_test/<scene>
#
# leaf_only.ply:
#   .../2026.08.15/<scene>/point_cloud/iteration_30000/leaf_only.ply
#
# Output:
#   .../2026.08.15/<scene>/leaf_mesh/
#
# Temporary leaf-only loading directory:
#   /media/ubuntu2/A/LWL/code/Train/Fast-PGSR/tmp/
#
# Original mesh/ and original train/ render outputs are NOT touched.
# ============================================================

project_root="/media/ubuntu2/A/LWL/code/Train/Fast-PGSR"
data_root="${project_root}/data/data_test"
output_root="/media/ubuntu2/My Passport/LWL/PLYOutput/启林北盆栽2026.6.28/大豆/2026.08.15"

# MUST be the modified leaf-only renderer, not render.py.
render_script="${project_root}/render_leaf.py"

conda_env="fast-pgsr-old"
GPU_ID=1
ITERATION=30000

MAX_DEPTH=15.0
VOXEL_SIZE=0.01

# Keep every non-empty disconnected leaf component.
MIN_COMPONENT_TRIANGLES=1

SCENES=(
    "soybean9_0815"
    "soybean10_0815"
    "soybean12_0815"
)

log_root="${project_root}/logs/leaf_mesh_0815"
mkdir -p "${log_root}"

cd "${project_root}" || exit 1

# ============================================================
# Basic checks
# ============================================================

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda command not found"
    exit 1
fi

if [ ! -f "${render_script}" ]; then
    echo "ERROR: modified render_leaf.py not found:"
    echo "  ${render_script}"
    exit 1
fi

# Prevent accidentally running an older render_leaf.py.
if ! grep -q "2026-09-22-project-tmp-v5" "${render_script}"; then
    echo "ERROR: render_leaf.py is not the expected v5 version."
    echo "Expected marker: 2026-09-22-project-tmp-v5"
    echo "Please overwrite:"
    echo "  ${render_script}"
    echo "with the latest v5 render_leaf.py."
    exit 1
fi

if [ ! -d "${data_root}" ]; then
    echo "ERROR: data_root not found:"
    echo "  ${data_root}"
    exit 1
fi

if [ ! -d "${output_root}" ]; then
    echo "ERROR: output_root not found:"
    echo "  ${output_root}"
    exit 1
fi

total=${#SCENES[@]}
ok=0
fail=0

echo "============================================================"
echo "Leaf-only Gaussian -> Mesh"
echo "Script ver   : ${SCRIPT_VERSION}"
echo "Scenes       : ${SCENES[*]}"
echo "Data root    : ${data_root}"
echo "Output root  : ${output_root}"
echo "Renderer     : ${render_script}"
echo "Temp root    : ${project_root}/tmp"
echo "GPU          : ${GPU_ID}"
echo "Iteration    : ${ITERATION}"
echo "============================================================"

# ============================================================
# Serial processing: exactly 3 scenes
# ============================================================

for i in "${!SCENES[@]}"; do
    scene_name="${SCENES[$i]}"

    scene_dir="${data_root}/${scene_name}"
    output_path="${output_root}/${scene_name}"
    iter_dir="${output_path}/point_cloud/iteration_${ITERATION}"

    leaf_ply="${iter_dir}/leaf_only.ply"
    leaf_mesh_dir="${output_path}/leaf_mesh"

    log_file="${log_root}/${scene_name}.log"

    echo
    echo "======================================================================"
    echo "[$((i+1))/${total}] ${scene_name}"
    echo "======================================================================"
    echo "Scene : ${scene_dir}"
    echo "Leaf  : ${leaf_ply}"
    echo "Out   : ${leaf_mesh_dir}"

    if [ ! -d "${scene_dir}" ]; then
        echo "[FAIL] scene directory missing: ${scene_dir}" | tee "${log_file}"
        fail=$((fail+1))
        continue
    fi

    if [ ! -d "${scene_dir}/images" ]; then
        echo "[FAIL] images directory missing: ${scene_dir}/images" | tee "${log_file}"
        fail=$((fail+1))
        continue
    fi

    for f in cameras.bin images.bin points3D.bin; do
        if [ ! -f "${scene_dir}/sparse/0/${f}" ]; then
            echo "[FAIL] COLMAP file missing: ${scene_dir}/sparse/0/${f}" | tee "${log_file}"
            fail=$((fail+1))
            continue 2
        fi
    done

    semantic_init_file="${scene_dir}/semantic_initialization/point_semantics.npz"
    if [ ! -s "${semantic_init_file}" ]; then
        echo "[FAIL] semantic initialization missing or empty: ${semantic_init_file}" | tee "${log_file}"
        fail=$((fail+1))
        continue
    fi

    if [ ! -d "${output_path}" ]; then
        echo "[FAIL] PGSR output directory missing: ${output_path}" | tee "${log_file}"
        fail=$((fail+1))
        continue
    fi

    # Scene(...) still loads the original full Gaussian model first.
    if [ ! -f "${iter_dir}/point_cloud.ply" ]; then
        echo "[FAIL] original point_cloud.ply missing: ${iter_dir}/point_cloud.ply" | tee "${log_file}"
        fail=$((fail+1))
        continue
    fi

    # Full-model semantic sidecar used during Scene() initialization.
    if [ ! -f "${iter_dir}/semantic_state.npz" ]; then
        echo "[FAIL] semantic_state.npz missing: ${iter_dir}/semantic_state.npz" | tee "${log_file}"
        fail=$((fail+1))
        continue
    fi

    if [ ! -s "${leaf_ply}" ]; then
        echo "[FAIL] leaf_only.ply missing or empty: ${leaf_ply}" | tee "${log_file}"
        fail=$((fail+1))
        continue
    fi

    # Recreate only leaf_mesh/. Original mesh/ is untouched.
    rm -rf "${leaf_mesh_dir}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    conda run \
        --no-capture-output \
        -n "${conda_env}" \
        python "${render_script}" \
            -s "${scene_dir}" \
            --scene_override "${scene_dir}" \
            -m "${output_path}" \
            --iteration "${ITERATION}" \
            --leaf_ply "${leaf_ply}" \
            --skip_test \
            --max_depth "${MAX_DEPTH}" \
            --voxel_size "${VOXEL_SIZE}" \
            --min_component_triangles "${MIN_COMPONENT_TRIANGLES}" \
        2>&1 | tee "${log_file}"

    ret=${PIPESTATUS[0]}

    if [ "${ret}" -eq 0 ]; then
        if [ -s "${leaf_mesh_dir}/tsdf_fusion.ply" ]; then
            echo "[OK] ${scene_name}"
            echo "Mesh: ${leaf_mesh_dir}"
            ok=$((ok+1))
        else
            echo "[FAIL] command exited normally but tsdf_fusion.ply was not created"
            fail=$((fail+1))
        fi
    else
        echo "[FAIL] ${scene_name}, exit code=${ret}"
        fail=$((fail+1))
    fi
done

echo
echo "============================================================"
echo "DONE"
echo "OK   : ${ok}"
echo "FAIL : ${fail}"
echo "Logs : ${log_root}"
echo "============================================================"

if [ "${fail}" -ne 0 ]; then
    exit 1
fi
