from __future__ import annotations

import cv2
import numpy as np
import torch
from comfy_api.latest import io


def fill_mask_holes(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"Mask Fill Holes expects a torch.Tensor MASK, got {type(mask).__name__}.")
    if mask.ndim == 2:
        batch = mask.unsqueeze(0)
    elif mask.ndim == 3:
        batch = mask
    else:
        raise ValueError(f"Mask Fill Holes expects MASK [H,W] or [B,H,W], got shape {tuple(mask.shape)}.")
    if any(dimension == 0 for dimension in batch.shape):
        raise ValueError(f"Mask Fill Holes requires nonempty dimensions, got shape {tuple(batch.shape)}.")

    source = batch.detach().to(device="cpu").numpy()
    results = []
    for item in source:
        binary = np.asarray(item > 0, dtype=np.uint8)
        padded = np.pad(binary, 1, mode="constant")
        exterior = padded.copy()
        cv2.floodFill(exterior, None, (0, 0), 1, flags=4)
        holes = exterior[1:-1, 1:-1] == 0
        results.append(torch.from_numpy(np.asarray(binary | holes, dtype=np.float32)))
    return torch.stack(results)


class MaskFillHoles(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteMaskFillHoles",
            display_name="Mask Fill Holes",
            category="Utility Suite/Mask",
            description="Fills enclosed holes independently in each item of a standard MASK batch.",
            inputs=[io.Mask.Input("mask")],
            outputs=[io.Mask.Output("mask", display_name="mask")],
        )

    @classmethod
    def execute(cls, mask: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput(fill_mask_holes(mask))


__all__ = ["MaskFillHoles", "fill_mask_holes"]
