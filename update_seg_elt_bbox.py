from __future__ import annotations

from typing import Any

import torch
from comfy_api.latest import io

from .mask_bbox import quantized_mask_bounds
from .segs_geometry import validated_segs

SEGS = io.Custom("SEGS")


def _cropped_mask(segment: Any, index: int, height: int, width: int) -> torch.Tensor:
    if not hasattr(segment, "cropped_mask") or segment.cropped_mask is None:
        raise TypeError(f"Update SEG_ELT bbox entry {index} requires a cropped_mask.")
    try:
        mask = torch.as_tensor(segment.cropped_mask)
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError(f"Update SEG_ELT bbox entry {index} cropped_mask must be tensor-compatible.") from error
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2 or tuple(mask.shape) != (height, width):
        raise ValueError(
            f"Update SEG_ELT bbox entry {index} cropped_mask shape {tuple(mask.shape)} "
            f"does not match crop_region extent {(height, width)}."
        )
    return mask.detach().cpu()


def update_seg_elt_bbox(segs: Any) -> tuple[Any, list[Any]]:
    header, entries, crops = validated_segs(segs, "Update SEG_ELT bbox")
    updated = []
    for index, (segment, crop) in enumerate(zip(entries, crops)):
        if not hasattr(segment, "_replace"):
            raise TypeError(f"Update SEG_ELT bbox entry {index} must support structural bbox replacement.")
        x1, y1, x2, y2 = crop
        bounds = quantized_mask_bounds(_cropped_mask(segment, index, y2 - y1, x2 - x1))
        if bounds is None:
            bbox = crop
        else:
            bbox = (
                x1 + bounds[0],
                y1 + bounds[1],
                x1 + bounds[2],
                y1 + bounds[3],
            )
        updated.append(segment._replace(bbox=bbox))
    return header, updated


class UpdateSEGELTBBox(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteUpdateSEGELTBBox",
            display_name="Update SEG_ELT bbox",
            category="Utility Suite/SEGS",
            description="Replaces each SEG bbox with its cropped-mask boundary in source-canvas coordinates.",
            inputs=[SEGS.Input("segs")],
            outputs=[SEGS.Output("segs")],
        )

    @classmethod
    def execute(cls, segs: Any) -> io.NodeOutput:
        return io.NodeOutput(update_seg_elt_bbox(segs))


__all__ = ["UpdateSEGELTBBox", "update_seg_elt_bbox"]
