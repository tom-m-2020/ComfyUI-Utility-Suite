from __future__ import annotations

import math
from collections import namedtuple

import numpy as np
import torch
from comfy_api.latest import io

from .mask_bbox import quantized_mask_bounds
from .segs_change_ratio import _feather_mask, _tile_dimensions
from .tiling import build_tile_layout

SEGS = io.Custom("SEGS")
SEG = namedtuple(
    "SEG",
    [
        "cropped_image",
        "cropped_mask",
        "confidence",
        "crop_region",
        "bbox",
        "label",
        "control_net_wrapper",
    ],
    defaults=[None],
)


def _mask_batch(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(
            "Mask Boundary to Tiles (SEGS) expects a torch.Tensor MASK, "
            f"got {type(mask).__name__}."
        )
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(
            "Mask Boundary to Tiles (SEGS) expects MASK [B,H,W], "
            f"got shape {tuple(mask.shape)}."
        )
    if any(dimension == 0 for dimension in mask.shape):
        raise ValueError(
            "Mask Boundary to Tiles (SEGS) requires nonempty dimensions, "
            f"got shape {tuple(mask.shape)}."
        )
    return mask.detach().cpu()


def mask_boundary_to_tiles_segs(
    mask: torch.Tensor,
    ratio: float,
    overlap_mode: str,
    min_overlap: int,
    feather_overlap: bool,
    max_tile_size: int,
) -> tuple[tuple[int, int], list[SEG]]:
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(
            "Mask Boundary to Tiles (SEGS) ratio must be finite and positive, "
            f"got {ratio}."
        )
    if min_overlap <= 0:
        raise ValueError(
            "Mask Boundary to Tiles (SEGS) min_overlap must be positive, "
            f"got {min_overlap}."
        )
    if overlap_mode not in ("pixel", "percent"):
        raise ValueError(
            "Mask Boundary to Tiles (SEGS) overlap_mode must be pixel or "
            f"percent, got {overlap_mode!r}."
        )
    if overlap_mode == "percent" and min_overlap > 90:
        raise ValueError(
            "Mask Boundary to Tiles (SEGS) min_overlap cannot exceed 90 "
            f"when overlap_mode is percent, got {min_overlap}."
        )
    if not 128 <= max_tile_size <= 2048:
        raise ValueError(
            "Mask Boundary to Tiles (SEGS) max_tile_size must be between 128 "
            f"and 2048, got {max_tile_size}."
        )

    source = _mask_batch(mask)
    height, width = source.shape[1:]
    segments = []
    for item in source:
        bbox = quantized_mask_bounds(item)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        region_width = x2 - x1
        region_height = y2 - y1
        tile_width, tile_height = _tile_dimensions(
            region_width,
            region_height,
            ratio,
            max_tile_size,
        )
        if overlap_mode == "percent":
            requested_overlap_x = math.ceil(tile_width * min_overlap / 100)
            requested_overlap_y = math.ceil(tile_height * min_overlap / 100)
        else:
            requested_overlap_x = min_overlap
            requested_overlap_y = min_overlap
        overlap_x = min(requested_overlap_x, max(0, tile_width - 1))
        overlap_y = min(requested_overlap_y, max(0, tile_height - 1))
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
            crop_region = [
                x1 + local.x0,
                y1 + local.y0,
                x1 + local.x1,
                y1 + local.y1,
            ]
            if feather_overlap:
                cropped_mask = _feather_mask(record)
            else:
                cropped_mask = np.ones(
                    (local.height, local.width), dtype=np.float32
                )
            segments.append(
                SEG(
                    None,
                    cropped_mask,
                    1.0,
                    crop_region,
                    bbox,
                    "",
                    None,
                )
            )

    return (height, width), segments


class MaskBoundaryToTilesSEGS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteMaskBoundaryToTilesSEGS",
            display_name="Mask Boundary to Tiles (SEGS)",
            category="Utility Suite/SEGS",
            description="Fills each MASK union bounding box with ratio-preserving tiles without expanding or shrinking its boundary.",
            inputs=[
                io.Mask.Input("mask"),
                io.Float.Input(
                    "ratio", default=1.0, min=0.01, max=100.0, step=0.01
                ),
                io.Combo.Input(
                    "overlap_mode", options=["pixel", "percent"], default="pixel"
                ),
                io.Int.Input("min_overlap", default=64, min=1, max=2047, step=1),
                io.Boolean.Input("feather_overlap", default=False),
                io.Int.Input(
                    "max_tile_size", default=2048, min=128, max=2048, step=1
                ),
            ],
            outputs=[SEGS.Output("segs", display_name="segs")],
        )

    @classmethod
    def execute(
        cls,
        mask: torch.Tensor,
        ratio: float,
        overlap_mode: str,
        min_overlap: int,
        feather_overlap: bool,
        max_tile_size: int,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            mask_boundary_to_tiles_segs(
                mask,
                ratio,
                overlap_mode,
                min_overlap,
                feather_overlap,
                max_tile_size,
            )
        )


__all__ = ["MaskBoundaryToTilesSEGS", "mask_boundary_to_tiles_segs"]
