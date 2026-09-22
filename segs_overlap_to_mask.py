from __future__ import annotations

from typing import Any

import numpy as np
from comfy_api.latest import io

from .filter_tile_segs import _mask_array
from .segs_geometry import integer, validated_segs

SEGS = io.Custom("SEGS")
COMPARE_MODES = ("all_segs", "previous_segs")


def _intersection(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    return (x1, y1, x2, y2) if x1 < x2 and y1 < y2 else None


def _cap_interval(start: int, end: int, current_start: int, current_end: int, cap: int) -> tuple[int, int]:
    if cap == 0 or end - start <= cap:
        return start, end
    touches_start = start == current_start
    touches_end = end == current_end
    if touches_start and not touches_end:
        return start, start + cap
    if touches_end and not touches_start:
        return end - cap, end
    return start, end


def _capped_overlap(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int], cap: int
) -> tuple[int, int, int, int] | None:
    overlap = _intersection(first, second)
    if overlap is None:
        return None
    x1, x2 = _cap_interval(overlap[0], overlap[2], first[0], first[2], cap)
    y1, y2 = _cap_interval(overlap[1], overlap[3], first[1], first[3], cap)
    return x1, y1, x2, y2


def _keep_original(generated: np.ndarray, original: np.ndarray) -> np.ndarray:
    return np.maximum(generated - (1.0 - original), 0.0).astype(np.float32)


def segs_overlap_to_mask(
    segs: Any, keep_original_mask: bool, compare_mode: str, cap_overlap_to: int, invert: bool
) -> tuple[Any, list[Any]]:
    if compare_mode not in COMPARE_MODES:
        raise ValueError(f"Unsupported SEGS Overlap to MASK compare_mode {compare_mode!r}.")
    cap_overlap_to = integer(cap_overlap_to, "cap_overlap_to")
    if cap_overlap_to < 0:
        raise ValueError(f"cap_overlap_to must be nonnegative, got {cap_overlap_to}.")
    header, entries, crops = validated_segs(segs, "SEGS Overlap to MASK")
    original_masks = [entry.cropped_mask if hasattr(entry, "cropped_mask") else None for entry in entries]
    if keep_original_mask:
        masks = []
        for index, (entry, crop) in enumerate(zip(entries, crops)):
            if original_masks[index] is None:
                raise TypeError(f"SEGS entry {index} has no cropped_mask for keep_original_mask.")
            masks.append(_mask_array(entry, index, crop[3] - crop[1], crop[2] - crop[0]))
    else:
        masks = None

    result: list[Any] = []
    for index, (entry, crop) in enumerate(zip(entries, crops)):
        if not hasattr(entry, "_replace") or not hasattr(entry, "cropped_mask"):
            raise TypeError(f"SEGS entry {index} must support structural cropped_mask replacement.")
        x1, y1, x2, y2 = crop
        generated = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
        comparisons = range(len(crops)) if compare_mode == "all_segs" else range(index)
        for other_index in comparisons:
            if other_index == index:
                continue
            overlap = _capped_overlap(crop, crops[other_index], cap_overlap_to)
            if overlap is None:
                continue
            ox1, oy1, ox2, oy2 = overlap
            generated[oy1 - y1 : oy2 - y1, ox1 - x1 : ox2 - x1] = 1.0
        if invert:
            generated = 1.0 - generated
        if masks is not None:
            generated = _keep_original(generated, masks[index])
        result.append(entry._replace(cropped_mask=generated))
    return header, result


class SEGSOverlapToMASK(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSEGSOverlapToMASK",
            display_name="SEGS Overlap to MASK",
            category="Utility Suite/SEGS",
            description="Builds independent hard local SEG masks from pairwise crop-region overlaps.",
            inputs=[
                SEGS.Input("segs"),
                io.Boolean.Input("keep_original_mask", default=False),
                io.Combo.Input("compare_mode", options=list(COMPARE_MODES)),
                io.Int.Input("cap_overlap_to", default=0, min=0, max=32768),
                io.Boolean.Input("invert", default=False),
            ],
            outputs=[SEGS.Output("segs", display_name="segs")],
        )

    @classmethod
    def execute(
        cls, segs: Any, keep_original_mask: bool, compare_mode: str, cap_overlap_to: int, invert: bool
    ) -> io.NodeOutput:
        return io.NodeOutput(segs_overlap_to_mask(segs, keep_original_mask, compare_mode, cap_overlap_to, invert))
