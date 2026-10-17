"""Prefer CUDA video decoding; keep fps selection and JPEG encoding unchanged."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def extract(video, output_dir, fps=2.5, use_gpu=True, gpu_index=0, executable="ffmpeg"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("Extraction output must be empty: {}".format(output_dir))
    for gpu in ([True, False] if use_gpu else [False]):
        # A failed CUDA attempt must not leave extra frames in the CPU result.
        with tempfile.TemporaryDirectory(prefix=".extract-", dir=str(output_dir.parent)) as temp:
            command = [executable, "-y"]
            if gpu:
                command += ["-hwaccel", "cuda", "-hwaccel_device", str(gpu_index)]
            command += ["-i", str(video), "-vf", "fps={}".format(fps), "-q:v", "1",
                        "-start_number", "1", str(Path(temp) / "frame_%06d.jpg"),
                        "-hide_banner", "-loglevel", "error"]
            print("[Extract] {} decoding".format("CUDA" if gpu else "CPU"), flush=True)
            result = subprocess.run(command)
            frames = sorted(Path(temp).glob("frame_*.jpg"))
            if result.returncode == 0 and frames:
                for frame in frames:
                    shutil.move(str(frame), str(output_dir / frame.name))
                return len(frames)
            if gpu:
                print("[Extract] CUDA decode failed or produced no frames; retry CPU", flush=True)
    raise RuntimeError("Video extraction failed: {}".format(video))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fps", type=float, default=2.5)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--no_gpu", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("fps must be positive")
    extract(args.video, args.output_dir, args.fps, not args.no_gpu, args.gpu_index)
