"""Reusable tensor grids and device-side training operations (PyTorch 1.12+)."""
from functools import lru_cache
import torch


@lru_cache(maxsize=4)
def pixel_grid(height, width, device):
    # Shared across cameras with the same resolution; never mutate this tensor.
    x, y = torch.meshgrid(torch.arange(width, device=device),
                          torch.arange(height, device=device), indexing="xy")
    return torch.stack((x, y), dim=-1).float()


def sample_valid_indices(mask, count):
    valid = torch.nonzero(mask.reshape(-1), as_tuple=True)[0]
    if valid.numel() > count:
        valid = valid[torch.randperm(valid.numel(), device=valid.device)[:count]]
    return valid


def nearest_view_indices(centers, rays, max_angle, min_distance, max_distance, count):
    distance = torch.norm(centers[:, None] - centers[None], dim=-1)
    angle = torch.arccos(torch.sum(rays[:, None] * rays[None], dim=-1)) * 180 / 3.14159
    # np.lexsort((angle, distance)): distance primary, angle secondary, stable ties.
    # PyTorch 1.12 supports stable sort, but not argsort's stable keyword.
    order = torch.sort(angle, dim=-1, stable=True).indices
    distance_order = torch.sort(distance.gather(1, order), dim=-1, stable=True).indices
    order = order.gather(1, distance_order)
    valid = (angle < max_angle) & (distance > min_distance) & (distance < max_distance)
    result = []
    for row in range(order.shape[0]):
        selected = order[row][valid[row, order[row]]][:count]
        result.append(selected.tolist())  # Python camera objects / JSON require host IDs.
    return result
