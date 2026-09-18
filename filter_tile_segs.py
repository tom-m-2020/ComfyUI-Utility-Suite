from __future__ import annotations

import operator
from typing import Any

import numpy as np
import torch
from comfy_api.latest import io

SEGS = io.Custom("SEGS")


def _integer(value: Any, description: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{description} must be an integer, got bool.")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{description} must be an integer, got {type(value).__name__}.") from error


def _rectangle(value: Any, description: str) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise TypeError(f"{description} must contain four XYXY coordinates.")
    return tuple(_integer(coordinate, f"{description} coordinate {index}") for index, coordinate in enumerate(value))


def _validated_segs(segs: Any) -> tuple[Any, list[Any] | tuple[Any, ...], list[tuple[int, int, int, int]]]:
    if not isinstance(segs, (list, tuple)) or len(segs) != 2:
        raise TypeError("Filter Tile SEGS expects SEGS shaped as ((height, width), segment_entries).")
    source_shape, entries = segs
    if not isinstance(source_shape, (list, tuple)) or len(source_shape) != 2:
        raise TypeError("Filter Tile SEGS source shape must contain height and width.")
    height = _integer(source_shape[0], "SEGS source height")
    width = _integer(source_shape[1], "SEGS source width")
    if height <= 0 or width <= 0:
        raise ValueError(f"Filter Tile SEGS requires positive source dimensions, got {(height, width)}.")
    if not isinstance(entries, (list, tuple)):
        raise TypeError("Filter Tile SEGS segment entries must be a list or tuple.")

    crop_regions: list[tuple[int, int, int, int]] = []
    for index, segment in enumerate(entries):
        if not hasattr(segment, "crop_region"):
            raise TypeError(f"SEGS entry {index} has no crop_region field.")
        crop = _rectangle(segment.crop_region, f"SEGS entry {index} crop_region")
        x1, y1, x2, y2 = crop
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"SEGS entry {index} crop_region must be nonempty, got {crop}.")
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"SEGS entry {index} crop_region is outside source canvas {(height, width)}: {crop}.")
        crop_regions.append(crop)
    return source_shape, entries, crop_regions


def _coverage_map(height: int, width: int, crop_regions: list[tuple[int, int, int, int]]) -> np.ndarray:
    difference = np.zeros((height + 1, width + 1), dtype=np.int64)
    for x1, y1, x2, y2 in crop_regions:
        difference[y1, x1] += 1
        difference[y1, x2] -= 1
        difference[y2, x1] -= 1
        difference[y2, x2] += 1
    np.cumsum(difference, axis=0, out=difference)
    np.cumsum(difference, axis=1, out=difference)
    return difference[:height, :width]


def _mask_array(segment: Any, index: int, height: int, width: int) -> np.ndarray:
    if not hasattr(segment, "cropped_mask") or segment.cropped_mask is None:
        raise TypeError(f"SEGS entry {index} has no cropped_mask for mask mode.")
    mask = segment.cropped_mask
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu()
        if mask.dtype == torch.bfloat16:
            mask = mask.to(torch.float32)
        mask = mask.numpy()
    elif not isinstance(mask, np.ndarray):
        raise TypeError(
            f"SEGS entry {index} cropped_mask must be a NumPy array or Torch tensor, got {type(mask).__name__}."
        )
    if mask.ndim == 3:
        if mask.shape[0] != 1:
            raise ValueError(
                f"SEGS entry {index} cropped_mask rank-3 form must have one leading item, got {mask.shape}."
            )
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"SEGS entry {index} cropped_mask must be [H,W] or [1,H,W], got {mask.shape}.")
    if mask.shape != (height, width):
        raise ValueError(
            f"SEGS entry {index} cropped_mask shape {mask.shape} does not match crop_region extent {(height, width)}."
        )
    return mask


def partition_tile_segs(segs: Any, mode: str) -> tuple[tuple[Any, list[Any]], tuple[Any, list[Any]]]:
    if mode not in ("mask", "bbox"):
        raise ValueError(f"Unsupported Filter Tile SEGS mode {mode!r}; expected 'mask' or 'bbox'.")
    source_shape, entries, crop_regions = _validated_segs(segs)
    height = _integer(source_shape[0], "SEGS source height")
    width = _integer(source_shape[1], "SEGS source width")
    coverage = _coverage_map(height, width, crop_regions)

    kept: list[Any] = []
    excluded: list[Any] = []
    for index, (segment, crop) in enumerate(zip(entries, crop_regions)):
        x1, y1, x2, y2 = crop
        local_coverage = coverage[y1:y2, x1:x2]
        local_stride = local_coverage == 1
        keep = False
        if mode == "mask":
            mask = _mask_array(segment, index, y2 - y1, x2 - x1)
            if local_stride.any():
                keep = bool(np.any((mask > 0) & local_stride))
        else:
            if not hasattr(segment, "bbox") or segment.bbox is None:
                raise TypeError(f"SEGS entry {index} has no bbox for bbox mode.")
            bx1, by1, bx2, by2 = _rectangle(segment.bbox, f"SEGS entry {index} bbox")
            if bx2 < bx1 or by2 < by1:
                raise ValueError(f"SEGS entry {index} bbox has reversed coordinates: {(bx1, by1, bx2, by2)}.")
            if local_stride.any():
                ix1 = max(0, x1, bx1)
                iy1 = max(0, y1, by1)
                ix2 = min(width, x2, bx2)
                iy2 = min(height, y2, by2)
                if ix1 < ix2 and iy1 < iy2:
                    keep = bool(np.any(coverage[iy1:iy2, ix1:ix2] == 1))
        (kept if keep else excluded).append(segment)

    return (source_shape, kept), (source_shape, excluded)


class FilterTileSEGS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteFilterTileSEGS",
            display_name="Filter Tile SEGS",
            category="Utility Suite/SEGS",
            description="Partitions SEGS by content in pixels uniquely covered by one crop region.",
            inputs=[SEGS.Input("segs"), io.Combo.Input("mode", options=["mask", "bbox"])],
            outputs=[
                SEGS.Output("segs", display_name="segs"),
                SEGS.Output("excluded_segs", display_name="excluded_segs"),
            ],
        )

    @classmethod
    def execute(cls, segs: Any, mode: str) -> io.NodeOutput:
        return io.NodeOutput(*partition_tile_segs(segs, mode))
