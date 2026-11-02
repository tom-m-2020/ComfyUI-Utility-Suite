from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from comfy_api.latest import io

from .segs_geometry import validated_segs
from .tiling import SpatialTileRecord, build_tile_layout

SEGS = io.Custom("SEGS")


def _expanded_axis(limit: int, start: int, end: int, desired: int) -> tuple[int, int]:
    center = start + (end - start) / 2
    new_start = int(center - desired / 2)
    if desired >= limit:
        return 0, limit
    if new_start < 0:
        return 0, desired
    if new_start + desired > limit:
        return limit - desired, limit
    return new_start, new_start + desired


def _expanded_crop(
    crop: tuple[int, int, int, int],
    canvas_width: int,
    canvas_height: int,
    ratio: float,
) -> tuple[list[int], bool]:
    x1, y1, x2, y2 = crop
    width = x2 - x1
    height = y2 - y1
    desired_width = width
    desired_height = height
    if width / height < ratio:
        desired_width = max(width, math.ceil(height * ratio))
    elif width / height > ratio:
        desired_height = max(height, math.ceil(width / ratio))

    clipped = desired_width > canvas_width or desired_height > canvas_height
    new_x1, new_x2 = _expanded_axis(canvas_width, x1, x2, desired_width)
    new_y1, new_y2 = _expanded_axis(canvas_height, y1, y2, desired_height)
    return [new_x1, new_y1, new_x2, new_y2], clipped


def _cropped_mask(segment: Any, crop: tuple[int, int, int, int]) -> np.ndarray:
    mask = segment.cropped_mask
    if mask is None:
        return np.zeros((crop[3] - crop[1], crop[2] - crop[0]), dtype=np.float32)
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    expected = (crop[3] - crop[1], crop[2] - crop[0])
    if mask.ndim != 2 or mask.shape != expected:
        raise ValueError(
            f"SEGS Change Ratio cropped_mask shape {mask.shape} does not match crop_region extent {expected}."
        )
    return np.asarray(mask, dtype=np.float32)


def _project_mask(
    mask: np.ndarray,
    source_crop: tuple[int, int, int, int],
    target_crop: tuple[int, int, int, int],
) -> np.ndarray:
    output = np.zeros(
        (target_crop[3] - target_crop[1], target_crop[2] - target_crop[0]),
        dtype=np.float32,
    )
    x1 = max(source_crop[0], target_crop[0])
    y1 = max(source_crop[1], target_crop[1])
    x2 = min(source_crop[2], target_crop[2])
    y2 = min(source_crop[3], target_crop[3])
    if x2 <= x1 or y2 <= y1:
        return output
    output[
        y1 - target_crop[1] : y2 - target_crop[1],
        x1 - target_crop[0] : x2 - target_crop[0],
    ] = mask[
        y1 - source_crop[1] : y2 - source_crop[1],
        x1 - source_crop[0] : x2 - source_crop[0],
    ]
    return output


def _tile_dimensions(
    region_width: int,
    region_height: int,
    ratio: float,
    max_tile_size: int,
) -> tuple[int, int]:
    maximum_area = max_tile_size * max_tile_size
    height_limit = min(
        region_height,
        region_width / ratio,
        math.sqrt(maximum_area / ratio),
    )
    tile_height = max(1, math.floor(height_limit))
    tile_width = max(1, min(region_width, math.floor(tile_height * ratio)))
    while tile_width * tile_height > maximum_area:
        tile_height -= 1
        tile_width = max(1, min(region_width, math.floor(tile_height * ratio)))
    return tile_width, tile_height


def _ramp(length: int, ascending: bool) -> np.ndarray:
    if length == 1:
        values = np.array([0.5], dtype=np.float32)
    else:
        values = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return values if ascending else values[::-1]


def _feather_mask(record: SpatialTileRecord) -> np.ndarray:
    height = record.source_rect.height
    width = record.source_rect.width
    y_weight = np.ones(height, dtype=np.float32)
    x_weight = np.ones(width, dtype=np.float32)
    overlaps = record.neighbor_overlaps
    if overlaps.top:
        y_weight[: overlaps.top] *= _ramp(overlaps.top, True)
    if overlaps.bottom:
        y_weight[-overlaps.bottom :] *= _ramp(overlaps.bottom, False)
    if overlaps.left:
        x_weight[: overlaps.left] *= _ramp(overlaps.left, True)
    if overlaps.right:
        x_weight[-overlaps.right :] *= _ramp(overlaps.right, False)
    return y_weight[:, None] * x_weight[None, :]


def _replace_segment(
    segment: Any,
    source_mask: np.ndarray,
    source_crop: tuple[int, int, int, int],
    target_crop: tuple[int, int, int, int],
    feather: np.ndarray | None = None,
) -> Any:
    mask = _project_mask(source_mask, source_crop, target_crop)
    if feather is not None:
        mask = mask * feather
    replacements = {
        "cropped_mask": mask,
        "crop_region": list(target_crop),
    }
    if tuple(target_crop) != source_crop:
        replacements["cropped_image"] = None
    try:
        return segment._replace(**replacements)
    except (AttributeError, TypeError) as error:
        raise TypeError(
            "SEGS Change Ratio requires namedtuple-compatible SEG entries."
        ) from error


def segs_change_ratio(
    segs: Any,
    ratio: float,
    tiling: bool,
    min_overlap: int,
    feather_overlap: bool,
    max_tile_size: int,
) -> tuple[Any, list[Any]]:
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(
            f"SEGS Change Ratio ratio must be finite and positive, got {ratio}."
        )
    if min_overlap <= 0:
        raise ValueError(
            f"SEGS Change Ratio min_overlap must be positive, got {min_overlap}."
        )
    if not 512 <= max_tile_size <= 2048:
        raise ValueError(
            f"SEGS Change Ratio max_tile_size must be between 512 and 2048, got {max_tile_size}."
        )

    source_shape, entries, crops = validated_segs(segs, "SEGS Change Ratio")
    canvas_height, canvas_width = source_shape
    output = []
    for segment, crop in zip(entries, crops):
        source_mask = _cropped_mask(segment, crop)
        expanded, clipped = _expanded_crop(
            crop,
            canvas_width,
            canvas_height,
            ratio,
        )
        region_width = expanded[2] - expanded[0]
        region_height = expanded[3] - expanded[1]
        exceeds_maximum = region_width * region_height > max_tile_size * max_tile_size
        if not tiling or (not clipped and not exceeds_maximum):
            output.append(_replace_segment(segment, source_mask, crop, tuple(expanded)))
            continue

        tile_width, tile_height = _tile_dimensions(
            region_width,
            region_height,
            ratio,
            max_tile_size,
        )
        layout = build_tile_layout(
            (1, region_height, region_width, 1),
            "uniform_grid",
            tile_width,
            tile_height,
            0,
            0,
            1,
            1,
            min_overlap,
            min_overlap,
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
            feather = _feather_mask(record) if feather_overlap else None
            output.append(
                _replace_segment(
                    segment,
                    source_mask,
                    crop,
                    tile_crop,
                    feather,
                )
            )
    return source_shape, output


class SEGSChangeRatio(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSEGSChangeRatio",
            display_name="SEGS Change Ratio",
            category="Utility Suite/SEGS",
            description="Expands SEG crop regions to a target width-to-height ratio and optionally tiles constrained regions.",
            inputs=[
                SEGS.Input("segs"),
                io.Float.Input("ratio", default=1.0, min=0.01, max=100.0, step=0.01),
                io.Boolean.Input("tiling", default=False),
                io.Int.Input("min_overlap", default=64, min=1, max=2047, step=1),
                io.Boolean.Input("feather_overlap", default=False),
                io.Int.Input("max_tile_size", default=2048, min=512, max=2048, step=1),
            ],
            outputs=[SEGS.Output("segs", display_name="segs")],
        )

    @classmethod
    def execute(
        cls,
        segs: Any,
        ratio: float,
        tiling: bool,
        min_overlap: int,
        feather_overlap: bool,
        max_tile_size: int,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            segs_change_ratio(
                segs,
                ratio,
                tiling,
                min_overlap,
                feather_overlap,
                max_tile_size,
            )
        )


__all__ = ["SEGSChangeRatio", "segs_change_ratio"]
