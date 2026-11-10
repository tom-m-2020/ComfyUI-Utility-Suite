from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from comfy_api.latest import io

from .segs_change_ratio import _feather_mask
from .segs_geometry import validated_segs
from .tiling import NeighborOverlaps, Rect, SpatialTileRecord

SEGS = io.Custom("SEGS")


def _cropped_mask(
    segment: Any,
    crop: tuple[int, int, int, int],
    index: int,
) -> np.ndarray:
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
            f"Crossfade Mask Overlap (SEGS) entry {index} cropped_mask shape "
            f"{mask.shape} does not match crop_region extent {expected}."
        )
    if not np.isfinite(mask).all():
        raise ValueError(
            f"Crossfade Mask Overlap (SEGS) entry {index} cropped_mask must be finite."
        )
    return np.clip(mask, 0.0, 1.0).astype(np.float32, copy=False)


def _neighbor_overlaps(
    crops: list[tuple[int, int, int, int]],
    max_feather: int = 0,
    feather_mode: str = "pixel",
) -> list[NeighborOverlaps]:
    def capped_amount(
        amount: int,
        first_length: int,
        second_length: int,
    ) -> int:
        if max_feather == 0:
            return amount
        if feather_mode == "percent":
            first_cap = math.ceil(first_length * max_feather / 100)
            second_cap = math.ceil(second_length * max_feather / 100)
            return min(amount, first_cap, second_cap)
        return min(amount, max_feather)

    overlaps = [[0, 0, 0, 0] for _ in crops]
    for first_index, first in enumerate(crops):
        first_width = first[2] - first[0]
        first_height = first[3] - first[1]
        first_center_x = (first[0] + first[2]) / 2
        first_center_y = (first[1] + first[3]) / 2
        for second_index in range(first_index + 1, len(crops)):
            second = crops[second_index]
            overlap_width = min(first[2], second[2]) - max(first[0], second[0])
            overlap_height = min(first[3], second[3]) - max(first[1], second[1])
            if overlap_width <= 0 or overlap_height <= 0:
                continue

            second_width = second[2] - second[0]
            second_height = second[3] - second[1]
            second_center_x = (second[0] + second[2]) / 2
            second_center_y = (second[1] + second[3]) / 2
            delta_x = second_center_x - first_center_x
            delta_y = second_center_y - first_center_y
            if delta_x == 0 and delta_y == 0:
                continue

            overlap_x_ratio = overlap_width / min(first_width, second_width)
            overlap_y_ratio = overlap_height / min(first_height, second_height)
            if overlap_x_ratio == overlap_y_ratio and delta_x != 0 and delta_y != 0:
                continue
            if overlap_x_ratio < overlap_y_ratio or delta_y == 0:
                amount = capped_amount(overlap_width, first_width, second_width)
                if delta_x > 0:
                    overlaps[first_index][2] = max(
                        overlaps[first_index][2], amount
                    )
                    overlaps[second_index][0] = max(
                        overlaps[second_index][0], amount
                    )
                else:
                    overlaps[first_index][0] = max(
                        overlaps[first_index][0], amount
                    )
                    overlaps[second_index][2] = max(
                        overlaps[second_index][2], amount
                    )
            elif delta_y > 0:
                amount = capped_amount(overlap_height, first_height, second_height)
                overlaps[first_index][3] = max(
                    overlaps[first_index][3], amount
                )
                overlaps[second_index][1] = max(
                    overlaps[second_index][1], amount
                )
            else:
                amount = capped_amount(overlap_height, first_height, second_height)
                overlaps[first_index][1] = max(
                    overlaps[first_index][1], amount
                )
                overlaps[second_index][3] = max(
                    overlaps[second_index][3], amount
                )
    return [NeighborOverlaps(*values) for values in overlaps]


def crossfade_mask_overlap_segs(
    segs: Any,
    feather_mode: str = "pixel",
    max_feather: int = 0,
) -> tuple[Any, list[Any]]:
    if feather_mode not in ("pixel", "percent"):
        raise ValueError(
            "Crossfade Mask Overlap (SEGS) feather_mode must be pixel or "
            f"percent, got {feather_mode!r}."
        )
    if isinstance(max_feather, bool) or not isinstance(max_feather, int):
        raise TypeError("Crossfade Mask Overlap (SEGS) max_feather must be an integer.")
    if max_feather < 0:
        raise ValueError(
            "Crossfade Mask Overlap (SEGS) max_feather must be nonnegative, "
            f"got {max_feather}."
        )
    if feather_mode == "percent" and max_feather > 100:
        raise ValueError(
            "Crossfade Mask Overlap (SEGS) max_feather cannot exceed 100 "
            f"when feather_mode is percent, got {max_feather}."
        )
    source_shape, entries, crops = validated_segs(
        segs, "Crossfade Mask Overlap (SEGS)"
    )
    overlaps = _neighbor_overlaps(crops, max_feather, feather_mode)
    output = []
    for index, (segment, crop, neighbor_overlap) in enumerate(
        zip(entries, crops, overlaps)
    ):
        mask = _cropped_mask(segment, crop, index)
        height, width = mask.shape
        record = SpatialTileRecord(
            row=0,
            column=0,
            source_rect=Rect(0, 0, width, height),
            valid_rect=Rect(0, 0, width, height),
            neighbor_overlaps=neighbor_overlap,
        )
        cropped_mask = mask * _feather_mask(record)
        try:
            output.append(segment._replace(cropped_mask=cropped_mask))
        except (AttributeError, TypeError) as error:
            raise TypeError(
                "Crossfade Mask Overlap (SEGS) requires namedtuple-compatible SEG entries."
            ) from error
    return source_shape, output


class CrossfadeMaskOverlapSEGS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteCrossfadeMaskOverlapSEGS",
            display_name="Crossfade Mask Overlap (SEGS)",
            category="Utility Suite/SEGS",
            description="Applies complementary directional feathers wherever SEG crop regions overlap.",
            inputs=[
                SEGS.Input("segs"),
                io.Combo.Input(
                    "feather_mode", options=["pixel", "percent"], default="pixel"
                ),
                io.Int.Input(
                    "max_feather", default=0, min=0, max=16384, step=1
                ),
            ],
            outputs=[SEGS.Output("segs", display_name="segs")],
        )

    @classmethod
    def execute(
        cls,
        segs: Any,
        feather_mode: str,
        max_feather: int,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            crossfade_mask_overlap_segs(segs, feather_mode, max_feather)
        )


__all__ = ["CrossfadeMaskOverlapSEGS", "crossfade_mask_overlap_segs"]
