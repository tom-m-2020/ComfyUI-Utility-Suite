from __future__ import annotations

from collections import namedtuple
from typing import Any

import numpy as np
import torch
from comfy_api.latest import io

from .tiling import TileLayout, build_tile_layout

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


def _single_mask(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"MASK to Tile SEGS expects a torch.Tensor MASK, got {type(mask).__name__}.")
    if mask.ndim == 2:
        source = mask
    elif mask.ndim == 3:
        if mask.shape[0] != 1:
            raise ValueError(
                "MASK to Tile SEGS represents one source canvas and requires MASK batch size 1, "
                f"got {mask.shape[0]}."
            )
        source = mask[0]
    else:
        raise ValueError(f"MASK to Tile SEGS expects MASK [H,W] or [1,H,W], got shape {tuple(mask.shape)}.")
    if source.shape[0] <= 0 or source.shape[1] <= 0:
        raise ValueError(f"MASK to Tile SEGS requires nonempty spatial dimensions, got {tuple(source.shape)}.")
    return source


def _validate_image(image: torch.Tensor | None, height: int, width: int) -> None:
    if image is None:
        return
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"MASK to Tile SEGS expects a torch.Tensor IMAGE, got {type(image).__name__}.")
    if image.ndim != 4:
        raise ValueError(f"MASK to Tile SEGS expects IMAGE [1,H,W,C], got shape {tuple(image.shape)}.")
    if image.shape[0] != 1:
        raise ValueError(
            "MASK to Tile SEGS represents one source canvas and requires IMAGE batch size 1, "
            f"got {image.shape[0]}."
        )
    if tuple(image.shape[1:3]) != (height, width):
        raise ValueError(
            "MASK to Tile SEGS requires IMAGE and MASK spatial dimensions to match, "
            f"got IMAGE {tuple(image.shape[1:3])} and MASK {(height, width)}."
        )
    if image.shape[3] <= 0:
        raise ValueError("MASK to Tile SEGS requires IMAGE to have at least one channel.")


def _numpy_float(tensor: torch.Tensor) -> np.ndarray:
    source = tensor.detach().cpu()
    if source.dtype == torch.bfloat16:
        source = source.to(torch.float32)
    return source.numpy().copy()


def _content_bbox(mask: np.ndarray, crop_region: list[int]) -> tuple[int, int, int, int]:
    quantized = (mask * 255.0).astype(np.uint8)
    ys, xs = np.nonzero(quantized)
    if len(xs) == 0:
        return tuple(crop_region)
    x0, y0 = crop_region[0], crop_region[1]
    return (
        x0 + int(xs.min()),
        y0 + int(ys.min()),
        x0 + int(xs.max()) + 1,
        y0 + int(ys.max()) + 1,
    )


def tile_mask_batch(mask: torch.Tensor, layout: TileLayout) -> torch.Tensor:
    tiles = torch.zeros(
        (len(layout.spatial_records), layout.tile_tensor_height, layout.tile_tensor_width),
        dtype=mask.dtype,
        device=mask.device,
    )
    for index, record in enumerate(layout.spatial_records):
        source = record.source_rect
        valid = record.valid_rect
        tiles[index, valid.y0:valid.y1, valid.x0:valid.x1].copy_(
            mask[source.y0:source.y1, source.x0:source.x1]
        )
    return tiles


def mask_to_tile_segs(
    mask: torch.Tensor,
    mode: str,
    tile_width: int,
    tile_height: int,
    overlap_x: int,
    overlap_y: int,
    max_columns: int,
    max_rows: int,
    min_overlap_x: int,
    min_overlap_y: int,
    min_columns: int,
    min_rows: int,
    image: torch.Tensor | None = None,
) -> tuple[tuple[tuple[int, int], list[Any]], torch.Tensor]:
    source_mask = _single_mask(mask)
    height, width = source_mask.shape
    _validate_image(image, height, width)
    layout = build_tile_layout(
        (1, height, width, 1),
        mode,
        tile_width,
        tile_height,
        overlap_x,
        overlap_y,
        max_columns,
        max_rows,
        min_overlap_x,
        min_overlap_y,
        min_columns,
        min_rows,
    )

    segments: list[Any] = []
    for record in layout.spatial_records:
        source = record.source_rect
        crop_region = [source.x0, source.y0, source.x1, source.y1]
        cropped_mask = _numpy_float(source_mask[source.y0:source.y1, source.x0:source.x1])
        cropped_image = None
        if image is not None:
            cropped_image = _numpy_float(image[:, source.y0:source.y1, source.x0:source.x1, :])
        segments.append(
            SEG(
                cropped_image,
                cropped_mask,
                1.0,
                crop_region,
                _content_bbox(cropped_mask, crop_region),
                "",
                None,
            )
        )

    return ((height, width), segments), tile_mask_batch(source_mask, layout)


class MaskToTileSEGS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteMaskToTileSEGS",
            display_name="MASK to Tile SEGS",
            category="Utility Suite/Mask/Tiling",
            description="Tiles one MASK canvas into geometry-defined Impact-compatible SEGS and a MASK batch.",
            inputs=[
                io.Mask.Input("mask"),
                io.Image.Input("image", optional=True),
                io.Combo.Input("mode", options=["fixed_tile", "bounded_grid", "uniform_grid"]),
                io.Int.Input("tile_width", default=1024, min=1, max=32768),
                io.Int.Input("tile_height", default=1024, min=1, max=32768),
                io.Int.Input("overlap_x", default=128, min=0, max=16384),
                io.Int.Input("overlap_y", default=128, min=0, max=16384),
                io.Int.Input("max_columns", default=3, min=1, max=256),
                io.Int.Input("max_rows", default=3, min=1, max=256),
                io.Int.Input("min_overlap_x", default=128, min=0, max=32768),
                io.Int.Input("min_overlap_y", default=128, min=0, max=32768),
                io.Int.Input("min_columns", default=1, min=1, max=32768),
                io.Int.Input("min_rows", default=1, min=1, max=32768),
            ],
            outputs=[SEGS.Output(display_name="segs"), io.Mask.Output(display_name="masks")],
        )

    @classmethod
    def execute(
        cls,
        mask: torch.Tensor,
        mode: str,
        tile_width: int,
        tile_height: int,
        overlap_x: int,
        overlap_y: int,
        max_columns: int,
        max_rows: int,
        min_overlap_x: int = 128,
        min_overlap_y: int = 128,
        min_columns: int = 1,
        min_rows: int = 1,
        image: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            *mask_to_tile_segs(
                mask,
                mode,
                tile_width,
                tile_height,
                overlap_x,
                overlap_y,
                max_columns,
                max_rows,
                min_overlap_x,
                min_overlap_y,
                min_columns,
                min_rows,
                image,
            )
        )
