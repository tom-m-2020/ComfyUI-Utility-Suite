from __future__ import annotations

import torch
from comfy_api.latest import io

from .mask_bbox import nonzero_mask_bounds


def _feather_crop(
    crop: torch.Tensor,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> torch.Tensor:
    output = crop.clone()
    height, width = output.shape
    left = min(left, width)
    right = min(right, width)
    top = min(top, height)
    bottom = min(bottom, height)

    for x in range(left):
        output[:, x] *= (x + 1.0) / left
    for x in range(right):
        output[:, -(x + 1)] *= (x + 1) / right
    for y in range(top):
        output[y, :] *= (y + 1) / top
    for y in range(bottom):
        output[-(y + 1), :] *= (y + 1) / bottom
    return output


def feather_mask_from_boundary(
    mask: torch.Tensor,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"Feather Mask from Boundary expects a torch.Tensor MASK, got {type(mask).__name__}.")
    if mask.ndim == 2:
        batch = mask.unsqueeze(0)
        restore_rank_two = True
    elif mask.ndim == 3:
        batch = mask
        restore_rank_two = False
    else:
        raise ValueError(
            f"Feather Mask from Boundary expects MASK [H,W] or [B,H,W], got shape {tuple(mask.shape)}."
        )
    if batch.shape[0] == 0 or batch.shape[1] == 0 or batch.shape[2] == 0:
        raise ValueError(
            f"Feather Mask from Boundary requires nonempty dimensions, got shape {tuple(batch.shape)}."
        )
    widths = (left, top, right, bottom)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in widths):
        raise TypeError("Feather Mask from Boundary widths must be integers.")
    if any(value < 0 for value in widths):
        raise ValueError(f"Feather Mask from Boundary widths must be nonnegative, got {widths}.")

    output = torch.zeros_like(batch)
    for index, item in enumerate(batch):
        bounds = nonzero_mask_bounds(item)
        if bounds is None:
            continue
        x0, y0, x1, y1 = bounds
        output[index, y0:y1, x0:x1] = _feather_crop(
            item[y0:y1, x0:x1], left, top, right, bottom
        )
    return output[0] if restore_rank_two else output


class FeatherMaskFromBoundary(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteFeatherMaskFromBoundary",
            display_name="Feather Mask from Boundary",
            category="Utility Suite/Mask",
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input("left", default=0, min=0, max=16384, step=1),
                io.Int.Input("top", default=0, min=0, max=16384, step=1),
                io.Int.Input("right", default=0, min=0, max=16384, step=1),
                io.Int.Input("bottom", default=0, min=0, max=16384, step=1),
            ],
            outputs=[io.Mask.Output("mask", display_name="mask")],
        )

    @classmethod
    def execute(
        cls,
        mask: torch.Tensor,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> io.NodeOutput:
        return io.NodeOutput(feather_mask_from_boundary(mask, left, top, right, bottom))
