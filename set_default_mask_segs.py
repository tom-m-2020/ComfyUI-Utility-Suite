from __future__ import annotations

from typing import Any

import numpy as np
import torch
from comfy_api.latest import io

from .combine_tiles_segs import SEGS
from .segs_geometry import validated_segs


def set_default_mask_for_segs(segs: Any, mask: torch.Tensor) -> tuple[Any, list[Any]]:
    header, entries, crops = validated_segs(segs, "Set Default Mask for SEGS")
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"mask must be a Torch MASK tensor, got {type(mask).__name__}.")
    if mask.ndim == 2:
        full_mask = mask
    elif mask.ndim == 3 and mask.shape[0] == 1:
        full_mask = mask[0]
    else:
        raise ValueError(f"mask must be [H,W] or [1,H,W], got {tuple(mask.shape)}.")
    if tuple(full_mask.shape) != tuple(header):
        raise ValueError(
            "Set Default Mask for SEGS requires the MASK canvas to match the "
            f"SEGS source shape {tuple(header)}, got {tuple(full_mask.shape)}."
        )
    mask_array = full_mask.detach().cpu().numpy().astype(np.float32, copy=True)
    if not np.isfinite(mask_array).all():
        raise ValueError("Set Default Mask for SEGS requires finite mask values.")

    output = []
    for index, (entry, crop) in enumerate(zip(entries, crops)):
        if not hasattr(entry, "_replace"):
            raise TypeError(f"SEGS entry {index} must support structural cropped_mask replacement.")
        x1, y1, x2, y2 = crop
        output.append(entry._replace(cropped_mask=mask_array[y1:y2, x1:x2].copy()))
    return header, output


class SetDefaultMaskForSEGS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSetDefaultMaskForSEGS",
            display_name="Set Default Mask for SEGS",
            category="Utility Suite/SEGS",
            description="Crops one full-canvas MASK into every existing SEG crop without changing SEGS geometry.",
            inputs=[SEGS.Input("segs"), io.Mask.Input("mask")],
            outputs=[SEGS.Output("segs", display_name="segs")],
        )

    @classmethod
    def execute(cls, segs: Any, mask: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput(set_default_mask_for_segs(segs, mask))


__all__ = ["SetDefaultMaskForSEGS", "set_default_mask_for_segs"]
