from __future__ import annotations

import cv2
import numpy as np
import torch
from comfy_api.latest import io

from .mask_bbox import nonzero_mask_bounds

MODES = ["convex_hull", "gap"]


def _mask_batch(mask: torch.Tensor, input_name: str) -> tuple[torch.Tensor, bool]:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"Bridge Masks {input_name} must be a torch.Tensor MASK, got {type(mask).__name__}.")
    if mask.ndim == 2:
        batch = mask.unsqueeze(0)
        rank_two = True
    elif mask.ndim == 3:
        batch = mask
        rank_two = False
    else:
        raise ValueError(f"Bridge Masks {input_name} must be [H,W] or [B,H,W], got {tuple(mask.shape)}.")
    if any(dimension == 0 for dimension in batch.shape):
        raise ValueError(f"Bridge Masks {input_name} requires nonempty dimensions, got {tuple(batch.shape)}.")
    return batch, rank_two


def _bbox_points(bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array(
        ((x1, y1), (x2 - 1, y1), (x2 - 1, y2 - 1), (x1, y2 - 1)),
        dtype=np.int32,
    )


def _hull_vertices(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    points = np.concatenate((_bbox_points(bbox_a), _bbox_points(bbox_b)))
    hull = cv2.convexHull(points.reshape((-1, 1, 2)))
    return [tuple(int(value) for value in point) for point in hull[:, 0, :]]


def _filled_hull(
    height: int,
    width: int,
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> np.ndarray:
    result = np.zeros((height, width), dtype=np.float32)
    vertices = np.asarray(_hull_vertices(bbox_a, bbox_b), dtype=np.int32)
    cv2.fillConvexPoly(result, vertices, 1.0)
    return result


def _has_bbox_gap(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> bool:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    return ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1


def bridge_masks(mask_a: torch.Tensor, mask_b: torch.Tensor, mode: str) -> torch.Tensor:
    if mode not in MODES:
        raise ValueError(f"Unsupported Bridge Masks mode {mode!r}; expected one of {MODES}.")
    batch_a, rank_two_a = _mask_batch(mask_a, "mask_a")
    batch_b, rank_two_b = _mask_batch(mask_b, "mask_b")
    if tuple(batch_a.shape[1:]) != tuple(batch_b.shape[1:]):
        raise ValueError(
            "Bridge Masks requires matching canvas dimensions, "
            f"got {tuple(batch_a.shape[1:])} and {tuple(batch_b.shape[1:])}."
        )
    count_a, count_b = batch_a.shape[0], batch_b.shape[0]
    if count_a != count_b and count_a != 1 and count_b != 1:
        raise ValueError(
            "Bridge Masks requires equal batch sizes or a singleton side, "
            f"got {count_a} and {count_b}."
        )

    count = max(count_a, count_b)
    height, width = batch_a.shape[1:]
    output_dtype = torch.promote_types(batch_a.dtype, batch_b.dtype)
    results: list[torch.Tensor] = []
    for index in range(count):
        item_a = batch_a[0 if count_a == 1 else index]
        item_b = batch_b[0 if count_b == 1 else index]
        bbox_a = nonzero_mask_bounds(item_a)
        bbox_b = nonzero_mask_bounds(item_b)
        if bbox_a is None:
            raise ValueError(f"Bridge Masks mask_a batch item {index} has no nonzero bounding box.")
        if bbox_b is None:
            raise ValueError(f"Bridge Masks mask_b batch item {index} has no nonzero bounding box.")

        hull = torch.from_numpy(_filled_hull(height, width, bbox_a, bbox_b)).to(
            device=batch_a.device,
            dtype=output_dtype,
        )
        if mode == "convex_hull":
            result = hull
        else:
            source_a = item_a.to(device=batch_a.device, dtype=output_dtype)
            source_b = item_b.to(device=batch_a.device, dtype=output_dtype)
            result = torch.maximum(source_a, source_b)
            if _has_bbox_gap(bbox_a, bbox_b):
                bridge = hull.clone()
                ax1, ay1, ax2, ay2 = bbox_a
                bx1, by1, bx2, by2 = bbox_b
                bridge[ay1:ay2, ax1:ax2] = 0
                bridge[by1:by2, bx1:bx2] = 0
                result = torch.maximum(result, bridge)
        results.append(result)

    output = torch.stack(results)
    return output[0] if rank_two_a and rank_two_b and count == 1 else output


class BridgeMasks(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteBridgeMasks",
            display_name="Bridge Masks",
            category="Utility Suite/Mask",
            description="Connects exact-nonzero mask bounds with straight-sided convex geometry.",
            inputs=[
                io.Mask.Input("mask_a"),
                io.Mask.Input("mask_b"),
                io.Combo.Input("mode", options=MODES),
            ],
            outputs=[io.Mask.Output("mask", display_name="mask")],
        )

    @classmethod
    def execute(cls, mask_a: torch.Tensor, mask_b: torch.Tensor, mode: str) -> io.NodeOutput:
        return io.NodeOutput(bridge_masks(mask_a, mask_b, mode))


__all__ = ["BridgeMasks", "bridge_masks"]
