from __future__ import annotations

import operator
from typing import Any

import numpy as np


def integer(value: Any, description: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{description} must be an integer, got bool.")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{description} must be an integer, got {type(value).__name__}.") from error


def rectangle(value: Any, description: str) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise TypeError(f"{description} must contain four XYXY coordinates.")
    return tuple(integer(coordinate, f"{description} coordinate {index}") for index, coordinate in enumerate(value))


def validated_segs(
    segs: Any, node_name: str
) -> tuple[Any, list[Any] | tuple[Any, ...], list[tuple[int, int, int, int]]]:
    if not isinstance(segs, (list, tuple)) or len(segs) != 2:
        raise TypeError(f"{node_name} expects SEGS shaped as ((height, width), segment_entries).")
    source_shape, entries = segs
    if not isinstance(source_shape, (list, tuple)) or len(source_shape) != 2:
        raise TypeError(f"{node_name} source shape must contain height and width.")
    height = integer(source_shape[0], "SEGS source height")
    width = integer(source_shape[1], "SEGS source width")
    if height <= 0 or width <= 0:
        raise ValueError(f"{node_name} requires positive source dimensions, got {(height, width)}.")
    if not isinstance(entries, (list, tuple)):
        raise TypeError(f"{node_name} segment entries must be a list or tuple.")

    crop_regions: list[tuple[int, int, int, int]] = []
    for index, segment in enumerate(entries):
        if not hasattr(segment, "crop_region"):
            raise TypeError(f"SEGS entry {index} has no crop_region field.")
        crop = rectangle(segment.crop_region, f"SEGS entry {index} crop_region")
        x1, y1, x2, y2 = crop
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"SEGS entry {index} crop_region must be nonempty, got {crop}.")
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"SEGS entry {index} crop_region is outside source canvas {(height, width)}: {crop}.")
        crop_regions.append(crop)
    return source_shape, entries, crop_regions


def coverage_map(height: int, width: int, crop_regions: list[tuple[int, int, int, int]]) -> np.ndarray:
    difference = np.zeros((height + 1, width + 1), dtype=np.int64)
    for x1, y1, x2, y2 in crop_regions:
        difference[y1, x1] += 1
        difference[y1, x2] -= 1
        difference[y2, x1] -= 1
        difference[y2, x2] += 1
    np.cumsum(difference, axis=0, out=difference)
    np.cumsum(difference, axis=1, out=difference)
    return difference[:height, :width]
