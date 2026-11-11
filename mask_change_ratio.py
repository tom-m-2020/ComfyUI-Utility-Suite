from __future__ import annotations

import math

import numpy as np
import torch
from comfy_api.latest import io

from .mask_bbox import quantized_mask_bounds
from .segs_change_ratio import _expanded_crop, _feather_mask, _tile_dimensions
from .tiling import SpatialTileRecord, build_tile_layout


def _mask_batch(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(
            f"MASK Change Ratio expects a torch.Tensor MASK, got {type(mask).__name__}."
        )
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(
            f"MASK Change Ratio expects MASK [B,H,W], got shape {tuple(mask.shape)}."
        )
    if any(dimension == 0 for dimension in mask.shape):
        raise ValueError(
            f"MASK Change Ratio requires nonempty dimensions, got shape {tuple(mask.shape)}."
        )
    return mask


def _filled_shape(height: int, width: int, shape: str) -> np.ndarray:
    if shape == "square":
        return np.ones((height, width), dtype=np.float32)

    y, x = np.ogrid[:height, :width]
    normalized_x = (x + 0.5 - width / 2) / (width / 2)
    normalized_y = (y + 0.5 - height / 2) / (height / 2)
    return np.asarray(
        normalized_x * normalized_x + normalized_y * normalized_y <= 1.0,
        dtype=np.float32,
    )


def _full_canvas_mask(
    height: int,
    width: int,
    crop: tuple[int, int, int, int],
    shape: str,
    record: SpatialTileRecord | None = None,
) -> np.ndarray:
    x1, y1, x2, y2 = crop
    local = _filled_shape(y2 - y1, x2 - x1, shape)
    if record is not None:
        local *= _feather_mask(record)
    output = np.zeros((height, width), dtype=np.float32)
    output[y1:y2, x1:x2] = local
    return output


def mask_change_ratio(
    mask: torch.Tensor,
    ratio: float,
    tiling: bool,
    overlap_mode: str,
    min_overlap: int,
    feather_overlap: bool,
    max_tile_size: int,
    shape: str,
) -> torch.Tensor:
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(
            f"MASK Change Ratio ratio must be finite and positive, got {ratio}."
        )
    if min_overlap <= 0:
        raise ValueError(
            f"MASK Change Ratio min_overlap must be positive, got {min_overlap}."
        )
    if overlap_mode not in ("pixel", "percent"):
        raise ValueError(
            "MASK Change Ratio overlap_mode must be pixel or percent, "
            f"got {overlap_mode!r}."
        )
    if overlap_mode == "percent" and min_overlap > 90:
        raise ValueError(
            "MASK Change Ratio min_overlap cannot exceed 90 when "
            f"overlap_mode is percent, got {min_overlap}."
        )
    if not 512 <= max_tile_size <= 2048:
        raise ValueError(
            "MASK Change Ratio max_tile_size must be between 512 and 2048, "
            f"got {max_tile_size}."
        )
    if shape not in ("square", "circle"):
        raise ValueError(
            f"MASK Change Ratio shape must be square or circle, got {shape!r}."
        )

    source = _mask_batch(mask)
    height, width = source.shape[1:]
    output = []
    for item in source.detach().cpu():
        bbox = quantized_mask_bounds(item)
        if bbox is None:
            output.append(np.zeros((height, width), dtype=np.float32))
            continue

        expanded, constrained = _expanded_crop(
            bbox,
            width,
            height,
            ratio,
            max_tile_size if tiling else None,
        )
        region_width = expanded[2] - expanded[0]
        region_height = expanded[3] - expanded[1]
        exceeds_maximum = region_width * region_height > max_tile_size * max_tile_size
        if not tiling or (not constrained and not exceeds_maximum):
            output.append(
                _full_canvas_mask(height, width, tuple(expanded), shape)
            )
            continue

        tile_width, tile_height = _tile_dimensions(
            region_width,
            region_height,
            ratio,
            max_tile_size,
        )
        if overlap_mode == "percent":
            overlap_x = math.ceil(tile_width * min_overlap / 100)
            overlap_y = math.ceil(tile_height * min_overlap / 100)
        else:
            overlap_x = min_overlap
            overlap_y = min_overlap
        layout = build_tile_layout(
            (1, region_height, region_width, 1),
            "uniform_grid",
            tile_width,
            tile_height,
            0,
            0,
            1,
            1,
            overlap_x,
            overlap_y,
            1,
            1,
        )
        for record in layout.spatial_records:
            local = record.source_rect
            tile_crop = (
                expanded[0] + local.x0,
                expanded[1] + local.y0,
                expanded[0] + local.x1,
                expanded[1] + local.y1,
            )
            output.append(
                _full_canvas_mask(
                    height,
                    width,
                    tile_crop,
                    shape,
                    record if feather_overlap else None,
                )
            )

    dtype = source.dtype if torch.is_floating_point(source) else torch.float32
    return torch.from_numpy(np.stack(output)).to(device=source.device, dtype=dtype)


class MASKChangeRatio(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteMASKChangeRatio",
            display_name="MASK Change Ratio",
            category="Utility Suite/Mask",
            description="Fills each MASK union bounding box, changes its ratio, and optionally emits overlapping tiles.",
            inputs=[
                io.Mask.Input("mask"),
                io.Float.Input(
                    "ratio", default=1.0, min=0.01, max=100.0, step=0.01
                ),
                io.Boolean.Input("tiling", default=False),
                io.Combo.Input(
                    "overlap_mode", options=["pixel", "percent"], default="pixel"
                ),
                io.Int.Input("min_overlap", default=64, min=1, max=2047, step=1),
                io.Boolean.Input("feather_overlap", default=False),
                io.Int.Input(
                    "max_tile_size", default=2048, min=512, max=2048, step=1
                ),
                io.Combo.Input(
                    "shape", options=["square", "circle"], default="square"
                ),
            ],
            outputs=[io.Mask.Output("mask", display_name="mask")],
        )

    @classmethod
    def execute(
        cls,
        mask: torch.Tensor,
        ratio: float,
        tiling: bool,
        overlap_mode: str,
        min_overlap: int,
        feather_overlap: bool,
        max_tile_size: int,
        shape: str,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            mask_change_ratio(
                mask,
                ratio,
                tiling,
                overlap_mode,
                min_overlap,
                feather_overlap,
                max_tile_size,
                shape,
            )
        )


__all__ = ["MASKChangeRatio", "mask_change_ratio"]
