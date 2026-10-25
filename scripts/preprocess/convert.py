"""COLMAP preparation with explicit scene selection and safe argv paths."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sqlite3


def prepare_scene(scene, args):
    if scene.is_symlink():
        raise ValueError("Refusing symlink scene: {}".format(scene))
    scene = scene.resolve()
    frames = scene / "input"
    if not frames.is_dir() or not any(frames.iterdir()):
        raise ValueError("Empty input frames: {}".format(frames))
    generated = [scene / name for name in (
        "images", "sparse", "stereo", "distorted", "images_2", "images_4", "images_8",
        "database.db", "database.db-journal", "database.db-wal", "database.db-shm")]
    resume_mapper = getattr(args, 'resume_mapper', False)
    protected = [p for p in generated if not (resume_mapper and p.name == 'database.db')]
    if not args.overwrite and any(p.exists() or p.is_symlink() for p in protected):
        raise FileExistsError("COLMAP outputs already exist; use --overwrite or a new scene")
    # All deletion targets are literal children of this scene. Never follow aliases.
    for path in protected:
        if path.is_symlink():
            raise ValueError("Refusing symlink output: {}".format(path))
    for path in protected:
        if path.is_dir():
            shutil.rmtree(str(path))
        elif path.exists():
            path.unlink()
    (scene / "sparse").mkdir()
    executable = args.colmap_executable or "colmap"
    database = str(scene / "database.db")
    use_gpu = "0" if args.no_gpu else "1"
    def run(command, *options):
        argv = [args.mapper_executable] if command == 'mapper' and getattr(args, 'mapper_executable', '') else [executable, command]
        subprocess.run(argv + list(options), check=True)
    if not resume_mapper:
        run("feature_extractor", "--database_path", database, "--image_path", str(frames),
            "--ImageReader.single_camera", "1", "--ImageReader.camera_model", args.camera,
            "--SiftExtraction.use_gpu", use_gpu, "--SiftExtraction.gpu_index", str(args.gpu_index))
    with sqlite3.connect(database) as connection:
        feature_images, features = connection.execute(
            'SELECT COUNT(*), COALESCE(SUM(rows), 0) FROM descriptors').fetchone()
    if not feature_images or not features:
        raise RuntimeError('COLMAP extracted no image descriptors. Check image decoding/bit depth before matching; the database is empty.')
    if not resume_mapper:
        run("exhaustive_matcher", "--database_path", database,
            "--SiftMatching.use_gpu", use_gpu, "--SiftMatching.gpu_index", str(args.gpu_index))
    else:
        with sqlite3.connect(database) as connection:
            count = connection.execute('SELECT COUNT(*) FROM two_view_geometries WHERE rows>=15').fetchone()[0]
        if not count:
            raise RuntimeError('Mapper resume requires existing geometrically verified matches')
    run("mapper", "--database_path", database, "--image_path", str(frames),
        "--output_path", str(scene / "sparse"), "--Mapper.ba_global_function_tolerance", "0.000001")
    for path in (scene / "sparse/0").iterdir():
        shutil.move(str(path), str(scene / "sparse" / path.name))
    run("image_undistorter", "--image_path", str(frames), "--input_path", str(scene / "sparse"),
        "--output_path", str(scene), "--output_type", "COLMAP")
    if args.resize:
        for factor, percent in ((2, "50%"), (4, "25%"), (8, "12.5%")):
            output = scene / ("images_{}".format(factor))
            output.mkdir()
            for path in (scene / "images").iterdir():
                if not path.is_file():
                    continue
                dest = output / path.name
                shutil.copy2(str(path), str(dest))
                subprocess.run([args.magick_executable or "magick", "mogrify", "-resize", percent, str(dest)], check=True)


def require_cuda(executable):
    result = subprocess.run([executable, "-h"], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    if result.returncode != 0 or "with cuda" not in result.stdout.lower():
        raise RuntimeError("COLMAP GPU mode requires a CUDA-enabled build (help must report 'with CUDA'). "
                           "Install that build, or explicitly choose --no_gpu / COLMAP_USE_GPU=0.")


def init_colmap(args):
    if not args.no_gpu:
        require_cuda(args.colmap_executable or "colmap")
    paths = [Path(args.scene_path)] if args.scene_path else sorted(p for p in Path(args.data_path).iterdir() if p.is_dir())
    if not paths:
        raise ValueError("No scenes found")
    for scene in paths:
        prepare_scene(scene, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Colmap converter")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data_path")
    source.add_argument("--scene_path")
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--camera", default="OPENCV")
    parser.add_argument("--colmap_executable", default="")
    parser.add_argument("--mapper_executable", default="")
    parser.add_argument("--resume_mapper", action="store_true")
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--magick_executable", default="")
    parser.add_argument("--overwrite", action="store_true")
    init_colmap(parser.parse_args())
    print("Done.")
