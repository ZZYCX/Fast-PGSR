date="2026.07.30"

mkdir -p "target_output/$date"

for scene in "$date"/*; do

    [ -d "$scene" ] || continue

    scene_name=$(basename "$scene")
    dst_dir="target_output/$date/$scene_name"

    plant_src="$scene/target_exports/plant_only/point_cloud.ply"
    rubik1_src="$scene/target_exports/rubik_1_only/point_cloud.ply"
    rubik2_src="$scene/target_exports/rubik_2_only/point_cloud.ply"
    mesh_src="$scene/mesh"

    echo "========================================"
    echo "Checking: $date/$scene_name"

    # plant_only 必须存在
    if [ ! -f "$plant_src" ]; then
        echo "[SKIP] Missing plant_only: $scene_name"
        continue
    fi

    # rubik_1 和 rubik_2 至少存在一个
    if [ ! -f "$rubik1_src" ] && [ ! -f "$rubik2_src" ]; then
        echo "[SKIP] Both rubik files missing: $scene_name"
        continue
    fi

    # 满足基本条件后创建目标目录
    mkdir -p "$dst_dir"

    # plant_only 必复制
    cp -a "$plant_src" "$dst_dir/plant_only.ply"
    echo "[COPY] plant_only.ply"

    # rubik_1 有就复制
    if [ -f "$rubik1_src" ]; then
        cp -a "$rubik1_src" "$dst_dir/rubik_1_only.ply"
        echo "[COPY] rubik_1_only.ply"
    fi

    # rubik_2 有就复制
    if [ -f "$rubik2_src" ]; then
        cp -a "$rubik2_src" "$dst_dir/rubik_2_only.ply"
        echo "[COPY] rubik_2_only.ply"
    fi

    # mesh 可选，有就复制
    if [ -d "$mesh_src" ]; then
        cp -a "$mesh_src" "$dst_dir/"
        echo "[COPY] mesh/"
    else
        echo "[INFO] No mesh: $scene_name"
    fi

    echo "[OK] $date/$scene_name"

done