from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io


def _normalize_region(limit: int, start: float, size: float) -> tuple[int, int]:
    if start < 0:
        end = min(limit, size)
        start = 0
    elif start + size > limit:
        start = max(0, limit - size)
        end = limit
    else:
        end = min(limit, start + size)
    return int(start), int(end)


def _combined_crop(width: int, height: int, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    bbox_width = x1 - x0
    bbox_height = y1 - y0
    crop_width = bbox_width * 3.0
    crop_height = bbox_height * 3.0
    start_x = x0 + bbox_width / 2 - crop_width / 2
    start_y = y0 + bbox_height / 2 - crop_height / 2
    start_x, end_x = _normalize_region(width, int(start_x), crop_width)
    start_y, end_y = _normalize_region(height, int(start_y), crop_height)
    return start_x, start_y, end_x, end_y


def fill_combined_mask(
    mask: torch.Tensor,
    bbox_fill: bool,
    drop_size: int,
    contour_fill: bool,
) -> torch.Tensor:
    del drop_size, contour_fill

    if mask.ndim == 4:
        mask = mask.squeeze(0).squeeze(0)
    elif mask.ndim == 3:
        mask = mask.squeeze(0)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"Mask Fill Combined expects a rank-2 or rank-3 MASK, got shape {tuple(mask.shape)}.")
    if mask.shape[1] == 0 or mask.shape[2] == 0:
        raise ValueError(f"Mask Fill Combined requires nonempty spatial dimensions, got shape {tuple(mask.shape)}.")

    mask_array = mask.numpy()
    source_height, source_width = mask_array.shape[1:]
    results: list[torch.Tensor] = []

    for mask_plane in mask_array:
        nonzero_y, nonzero_x = np.nonzero(mask_plane)
        if len(nonzero_y) == 0 or len(nonzero_x) == 0:
            continue

        bbox = (int(nonzero_x.min()), int(nonzero_y.min()), int(nonzero_x.max()), int(nonzero_y.max()))
        crop_x0, crop_y0, crop_x1, crop_y1 = _combined_crop(source_width, source_height, bbox)
        if crop_x1 - crop_x0 <= 0 or crop_y1 - crop_y0 <= 0:
            continue

        cropped = mask_plane[crop_y0:crop_y1, crop_x0:crop_x1]
        if bbox_fill:
            bbox_x0, bbox_y0, bbox_x1, bbox_y1 = bbox
            cropped = cropped.copy()
            cropped[bbox_y0:bbox_y1, bbox_x0:bbox_x1] = 1.0

        canvas = torch.zeros((source_height, source_width), dtype=torch.uint8, device="cpu")
        canvas[crop_y0:crop_y1, crop_x0:crop_x1] |= torch.from_numpy((cropped * 255).astype(np.uint8))
        results.append((canvas / 255.0).to(torch.float32))

    if not results:
        return torch.zeros((1, source_height, source_width), dtype=torch.float32, device="cpu")
    return torch.stack(results)


class MaskFillCombined(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteMaskFillCombined",
            display_name="Mask Fill Combined",
            category="Utility Suite/Mask",
            description="Keeps each source mask's disconnected nonzero regions combined in one full-canvas mask.",
            inputs=[
                io.Mask.Input("mask"),
                io.Boolean.Input("bbox_fill", default=False, label_on="enabled", label_off="disabled"),
                io.Int.Input("drop_size", default=10, min=1, max=16384, step=1),
                io.Boolean.Input("contour_fill", default=False, label_on="enabled", label_off="disabled"),
            ],
            outputs=[io.Mask.Output("mask", display_name="mask")],
        )

    @classmethod
    def execute(
        cls,
        mask: torch.Tensor,
        bbox_fill: bool,
        drop_size: int,
        contour_fill: bool,
    ) -> io.NodeOutput:
        return io.NodeOutput(fill_combined_mask(mask, bbox_fill, drop_size, contour_fill))
