python scripts/leaf_fit/leafFit_segment.py     --input output/mungbean13_semantic_v4/target_exports/plant_only/point_cloud.ply     --leaf-fit-root third_party/leaf_fit     --output output/mungbean13_semantic_v4/target_exports/plant_only/leaf_fit_segments     --pick-root --tip-k 32
--path-k 256     --disable-apex-cluster   



python scripts/leaf_fit/leafFit_segment_root_path.py --input output/mungbean1_0804/target_exports/plant_only/point_cloud.ply --leaf-fit-root third_party/leaf_fit --disable-apex-cluster  --tip-k 32 --path-k 64 --disable-apex-cluster
