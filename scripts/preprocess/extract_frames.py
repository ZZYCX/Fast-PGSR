"""Prefer CUDA video decoding; keep fps selection and JPEG encoding unchanged."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
import hashlib
import json
import os


def extract_retained(video, output_dir, fps, manifest, executable, native=False):
    """Lossless, chronological sampling plus exact, previously decoded source frames.

    Opt-in recovery only. All frames still require new SfM registration. There is
    no camera-pose transfer from the old frame sequence.
    """
    from PIL import Image
    manifest = Path(manifest) if manifest else None
    records = [r for r in json.loads(manifest.read_text(encoding='utf-8'))
               if r['scene'] == Path(video).stem] if manifest else []
    if not records and not native:
        raise ValueError('No retained frames for ' + str(video))
    def sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(4*1024*1024), b''):
                h.update(block)
        return h.hexdigest()
    video_sha = sha(video)
    retained = {}
    for row in records:
        image = manifest.parent / row['image']
        if row['video_sha256'] != video_sha or sha(image) != row['image_sha256']:
            raise ValueError('Retained frame provenance mismatch')
        index = int(row['source_frame_zero_based'])
        if index in retained:
            raise ValueError('Duplicate retained source frame')
        retained[index] = (row, image)
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time',
                            '-of', 'json', str(video)], stdout=subprocess.PIPE, check=True)
    times = [float(f['best_effort_timestamp_time']) for f in json.loads(probe.stdout)['frames']]
    if not times or any(b < a for a, b in zip(times, times[1:])):
        raise ValueError('Invalid source frame timestamps')
    for index, (row, _) in retained.items():
        if index < 0 or index >= len(times) or abs(times[index] - row['pts_s']) > .002:
            raise ValueError('Retained source frame PTS mismatch')
    indices, next_time = [], times[0]
    for index, timestamp in enumerate(times):
        regular = timestamp + 1e-7 >= next_time
        if native or regular or index in retained:
            indices.append(index)
        if regular:
            next_time += 1.0 / fps
    expression = '+'.join('eq(n\\,%s)' % index for index in indices) if not native else None
    with tempfile.TemporaryDirectory(prefix='.retained-', dir=str(output_dir.parent)) as temp:
        # Software decode matches the source-frame audit; ffmpeg honors display rotation.
        subprocess.run([executable, '-v', 'error', '-threads', '1', '-i', str(video),
                        '-filter_threads', '1', '-vf', 'select='+expression if expression else 'null',
                        '-vsync', '0', '-pix_fmt', 'rgb24', '-threads', '1', '-start_number', '1',
                        str(Path(temp)/'frame_%06d.png')], check=True)
        frames = sorted(Path(temp).glob('frame_*.png'))
        if len(frames) != len(indices):
            raise RuntimeError('Lossless source frame count mismatch')
        mapping = []
        for frame, index in zip(frames, indices):
            row = {'image': frame.name, 'source_frame_zero_based': index, 'pts_s': times[index]}
            if index in retained:
                proof, original = retained[index]
                with Image.open(frame) as decoded, Image.open(original) as confirmed:
                    if decoded.size != confirmed.size:
                        raise ValueError('Retained frame rotation/dimensions mismatch')
                shutil.copyfile(str(original), str(frame))
                row['confirmed_image_sha256'] = proof['image_sha256']
            row['image_sha256'] = sha(frame)
            mapping.append(row)
        for frame in frames:
            shutil.move(str(frame), str(output_dir/frame.name))
    (output_dir.parent/'retained-extraction.json').write_text(json.dumps(
        {'video_sha256': video_sha, 'fps': fps, 'native_all_frames': native, 'pixel_format': 'rgb24',
         'frames': mapping}, indent=2), encoding='utf-8')
    return len(mapping)


def extract(video, output_dir, fps=2.5, use_gpu=True, gpu_index=0, executable="ffmpeg"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("Extraction output must be empty: {}".format(output_dir))
    manifest = os.environ.get('ARUCO_RETAINED_MANIFEST')
    native = os.environ.get('ARUCO_NATIVE_FRAMES', '0') == '1'
    if manifest or native:
        return extract_retained(video, output_dir, fps, manifest, executable, native)
    for gpu in ([True, False] if use_gpu else [False]):
        # A failed CUDA attempt must not leave extra frames in the CPU result.
        with tempfile.TemporaryDirectory(prefix=".extract-", dir=str(output_dir.parent)) as temp:
            command = [executable, "-nostdin", "-y"]
            if gpu:
                command += ["-hwaccel", "cuda", "-hwaccel_device", str(gpu_index)]
            command += ["-i", str(video), "-vf", "fps={}".format(fps), "-q:v", "1",
                        "-start_number", "1", str(Path(temp) / "frame_%06d.jpg"),
                        "-hide_banner", "-loglevel", "error"]
            print("[Extract] {} decoding".format("CUDA" if gpu else "CPU"), flush=True)
            result = subprocess.run(command, stdin=subprocess.DEVNULL)
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
