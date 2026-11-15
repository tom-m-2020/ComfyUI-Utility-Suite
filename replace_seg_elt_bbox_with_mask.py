from __future__ import annotations

from typing import Any

import torch
from comfy_api.latest import io

from .mask_bbox import quantized_mask_bounds
from .segs_geometry import integer, validated_segs

SEGS = io.Custom("SEGS")


def _single_mask(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"Replace SEG_ELT bbox with MASK expects a torch.Tensor MASK, got {type(mask).__name__}.")
    if mask.ndim == 2:
        source = mask
    elif mask.ndim == 3 and mask.shape[0] == 1:
        source = mask[0]
    else:
        raise ValueError(
            "Replace SEG_ELT bbox with MASK expects MASK [H,W] or [1,H,W], "
            f"got shape {tuple(mask.shape)}."
        )
    if source.shape[0] <= 0 or source.shape[1] <= 0:
        raise ValueError(
            "Replace SEG_ELT bbox with MASK requires nonempty spatial dimensions, "
            f"got {tuple(source.shape)}."
        )
    return source.detach().cpu().to(torch.float32).clamp(0.0, 1.0)


def replace_seg_elt_bbox_with_mask(
    segs: Any,
    mask: torch.Tensor,
    index: int,
) -> tuple[Any, list[Any]]:
    header, entries, crops = validated_segs(segs, "Replace SEG_ELT bbox with MASK")
    selected = integer(index, "Replace SEG_ELT bbox with MASK index")
    if not 0 <= selected < len(entries):
        raise IndexError(
            f"Replace SEG_ELT bbox with MASK index {selected} is outside SEGS entry range 0..{len(entries) - 1}."
        )
    segment = entries[selected]
    if not hasattr(segment, "_replace"):
        raise TypeError(f"SEGS entry {selected} must support structural bbox replacement.")

    height = integer(header[0], "SEGS source height")
    width = integer(header[1], "SEGS source width")
    bounds = quantized_mask_bounds(_single_mask(mask))
    if bounds is None:
        bbox = crops[selected]
    else:
        bbox = (
            min(width, max(0, bounds[0])),
            min(height, max(0, bounds[1])),
            min(width, max(0, bounds[2])),
            min(height, max(0, bounds[3])),
        )
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            bbox = crops[selected]

    output = list(entries)
    output[selected] = segment._replace(bbox=bbox)
    return header, output


class ReplaceSEGELTBBoxWithMASK(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteReplaceSEGELTBBoxWithMASK",
            display_name="Replace SEG_ELT bbox with MASK",
            category="Utility Suite/SEGS",
            description="Replaces one SEG bbox with a source-coordinate MASK boundary without applying a crop offset.",
            inputs=[
                SEGS.Input("segs"),
                io.Mask.Input("mask"),
                io.Int.Input("index", default=0, min=0, step=1),
            ],
            outputs=[SEGS.Output("segs")],
        )

    @classmethod
    def execute(cls, segs: Any, mask: torch.Tensor, index: int) -> io.NodeOutput:
        return io.NodeOutput(replace_seg_elt_bbox_with_mask(segs, mask, index))


__all__ = ["ReplaceSEGELTBBoxWithMASK", "replace_seg_elt_bbox_with_mask"]
