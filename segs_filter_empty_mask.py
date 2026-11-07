from __future__ import annotations

from typing import Any

import torch
from comfy_api.latest import io

from .mask_bbox import quantized_mask_union_bounds

SEGS = io.Custom("SEGS")
SEGS_INDICES = io.Custom("SEGS_INDICES")


def mask_is_empty(mask: Any, strict: bool) -> bool:
    if mask is None:
        return True
    try:
        tensor = torch.as_tensor(mask)
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError(
            f"SEGS Filter (empty mask) requires tensor-compatible cropped_mask values, got {type(mask).__name__}."
        ) from error
    if strict:
        return bool(torch.all(tensor == 0).item())
    return quantized_mask_union_bounds(tensor) is None


def filter_segs_empty_mask(
    segs: Any,
    mode: bool,
    strict: bool,
) -> tuple[tuple[Any, list[Any]], tuple[Any, list[Any]], list[int]]:
    filtered_segs = []
    remained_segs = []
    indices = []

    for index, seg in enumerate(segs[1]):
        if not hasattr(seg, "cropped_mask"):
            raise TypeError(f"SEGS entry {index} has no cropped_mask.")
        empty = mask_is_empty(seg.cropped_mask, strict)
        if empty == mode:
            filtered_segs.append(seg)
            indices.append(index)
        else:
            remained_segs.append(seg)

    return (segs[0], filtered_segs), (segs[0], remained_segs), indices


class SEGSFilterEmptyMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSEGSFilterEmptyMask",
            display_name="SEGS Filter (empty mask)",
            category="Utility Suite/SEGS",
            description="Partitions SEGS by whether cropped masks are empty and returns original selected indices.",
            inputs=[
                SEGS.Input("segs"),
                io.Boolean.Input("mode", default=True, label_on="empty", label_off="nonempty"),
                io.Boolean.Input("strict", default=True),
            ],
            outputs=[
                SEGS.Output("filtered_SEGS"),
                SEGS.Output("remained_SEGS"),
                SEGS_INDICES.Output("index"),
            ],
        )

    @classmethod
    def execute(cls, segs: Any, mode: bool, strict: bool) -> io.NodeOutput:
        return io.NodeOutput(*filter_segs_empty_mask(segs, mode, strict))
