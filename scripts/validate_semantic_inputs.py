"""CPU checks and provenance for the single/two-cube batch pipeline (Python 3.7+)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


# The deployed strict guard remains trusted for existing completed scenes.
# Other old/unknown implementations must still regenerate their masks.
LEGACY_STRICT_GUARD_SHA256 = "1492fa2e4fb0a6bc10711e6ce51a7acceeafb18d16f7a9d8d50bd454865f043b"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def model_files(directory):
    result = {}
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        path = Path(directory) / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("缺少 COLMAP 模型文件: {}".format(path))
        result[name] = digest(path)
    return result


def images(scene):
    paths = sorted(p for p in (scene / "images").iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError("训练图像为空或 stem 重复")
    return paths


def validate_masks(scene, rubik_count, require_instance_guard=False):
    report = json.loads((scene / "grounded_sam2_scene/summary.json").read_text(encoding="utf-8"))
    active = set(range(1, rubik_count + 2))
    if report.get("rubik_count") != rubik_count or set(map(int, report["object_ids"])) != active:
        raise ValueError("旧 Mask 魔方配置不匹配；请重做 MASK")
    guarded = (report.get("instance_guard") or {}).get("version") == "sfm_sam_instance_v1"
    if require_instance_guard and rubik_count == 1 and not guarded:
        raise ValueError("旧 Mask 未执行源头单株实例约束；请备份后重做 MASK/LABEL")
    accepted_guard_hashes = {digest(Path(__file__).parent/'mask_generation/instance_mask_guard.py'),
                             LEGACY_STRICT_GUARD_SHA256}
    if guarded and report['instance_guard'].get('implementation_sha256') not in accepted_guard_hashes:
        raise ValueError('实例约束实现已更新，请重新生成 MASK/LABEL')
    paths = images(scene)
    masks = list((scene / "object_mask").glob("*.png"))
    if {p.stem for p in masks} != {p.stem for p in paths}:
        raise ValueError("Mask 必须与训练图像 stem 一一对应")
    counts = {i: 0 for i in active}
    for path in paths:
        with Image.open(path) as image, Image.open(scene / "object_mask" / (path.stem + ".png")) as mask:
            if report.get("image_sha256", {}).get(path.name) != digest(path):
                raise ValueError("Mask 对应图像已改变，请重做 MASK: {}".format(path.name))
            values = np.asarray(mask)
            if guarded and report.get("mask_sha256", {}).get(path.stem) != digest(scene / "object_mask" / (path.stem + ".png")):
                raise ValueError("实例约束 Mask 已改变，请重新验证: {}".format(path.name))
            if values.ndim != 2 or mask.size != image.size:
                raise ValueError("Mask 通道或尺寸错误: {}".format(path.name))
            if not set(np.unique(values).tolist()).issubset({0} | active):
                raise ValueError("Mask 含未启用类别: {}".format(path.name))
            for i in active:
                counts[i] += int(np.count_nonzero(values == i))
    # An occluded object may be absent in individual views, but not the entire scene.
    if not all(counts.values()):
        raise ValueError("整个场景中存在空目标类别: {}".format(counts))
    return paths


def initialization_signature(scene, rubik_count):
    paths = validate_masks(scene, rubik_count)
    npz = scene / "semantic_initialization/point_semantics.npz"
    with np.load(str(npz), allow_pickle=False) as data:
        ids = data["object_id"]
        probs = data["object_probs"]
        if (ids.size == 0 or probs.shape != (len(ids), 4)
                or not np.isfinite(probs).all() or (probs < 0).any()
                or not np.allclose(probs.sum(axis=1), 1, atol=1e-4)
                or not np.isin(ids, [-1] + list(range(rubik_count + 2))).all()):
            raise ValueError("初始化标签/四类概率不合法")
        point_ids = data["point3D_ids"]
        if len(point_ids) != len(ids) or len(np.unique(point_ids)) != len(ids):
            raise ValueError("初始化 POINT3D_ID 不匹配或重复")
    return {
        "rubik_count": rubik_count,
        "sparse": model_files(scene / "sparse/0"),
        "initialization": digest(npz),
        "images": {p.name: digest(p) for p in paths},
        "masks": {p.stem: digest(scene / "object_mask" / (p.stem + ".png")) for p in paths},
    }


def binding_path(scene):
    return scene / "semantic_initialization/input_binding.json"


def check_binding(scene, rubik_count):
    current = initialization_signature(scene, rubik_count)
    saved = json.loads(binding_path(scene).read_text(encoding="utf-8"))
    if saved != current:
        raise ValueError("初始化与当前相机/点云/图像/Mask 不一致；请重做 LABEL")
    return current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("masks", "bind", "init", "trained", "bind-trained"))
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--rubik-count", type=int, choices=(1, 2), default=1)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--require-instance-guard", action="store_true")
    args = parser.parse_args()
    if args.stage == "masks":
        validate_masks(args.scene, args.rubik_count, args.require_instance_guard)
    elif args.stage == "bind":
        signature = initialization_signature(args.scene, args.rubik_count)
        binding_path(args.scene).write_text(json.dumps(signature, indent=2), encoding="utf-8")
    else:
        signature = check_binding(args.scene, args.rubik_count)
        if args.stage in ("trained", "bind-trained"):
            if args.model is None:
                parser.error("此阶段需要 --model")
            # Offline exporter imports CPU-only checksums, not CUDA training extensions.
            from export_target_gaussians import validate_input
            iteration = args.model / "point_cloud" / ("iteration_{}".format(args.iteration))
            _, _, sem = validate_input(str(iteration / "point_cloud.ply"), str(iteration / "semantic_state.npz"))
            if args.rubik_count == 1 and np.any(sem["object_id"] == 3):
                raise ValueError("单魔方训练结果出现类别3")
            target = args.model / "semantic_input_binding.json"
            if args.stage == "bind-trained":
                target.write_text(json.dumps(signature, indent=2), encoding="utf-8")
            elif json.loads(target.read_text(encoding="utf-8")) != signature:
                raise ValueError("训练结果与当前语义输入不一致")
    print("[Semantic validation] {} OK".format(args.stage))


if __name__ == "__main__":
    main()
