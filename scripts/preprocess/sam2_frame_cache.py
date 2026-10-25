"""Bound the CPU video cache while reproducing SAM2's synchronous input tensors."""
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path


@contextmanager
def bounded_video_frames():
    import torch
    from sam2.utils import misc
    original = misc.load_video_frames_from_jpg_images

    def load(video_path, image_size, offload_video_to_cpu, img_mean=(.485, .456, .406),
             img_std=(.229, .224, .225), async_loading_frames=False, compute_device=None):
        if not offload_video_to_cpu or async_loading_frames:
            raise ValueError('Bounded frames require the existing synchronous CPU-offload mode')
        paths = sorted((p for p in Path(video_path).iterdir()
                        if p.suffix.lower() in ('.jpg', '.jpeg')), key=lambda p: int(p.stem))
        if not paths:
            raise ValueError('Empty SAM2 frame directory')
        mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        _, height, width = misc._load_img_as_tensor(str(paths[0]), image_size)

        class Frames:
            def __len__(self):
                return len(paths)

            @lru_cache(maxsize=2)
            def __getitem__(self, index):
                image, _, _ = misc._load_img_as_tensor(str(paths[index]), image_size)
                # The original synchronous loader assigns into float32 zeros
                # BEFORE normalizing. Preserve that conversion and operation order.
                image = image.float().contiguous()
                image -= mean
                image /= std
                return image

        return Frames(), height, width

    misc.load_video_frames_from_jpg_images = load
    try:
        yield
    finally:
        misc.load_video_frames_from_jpg_images = original
