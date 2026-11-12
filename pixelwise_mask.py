from __future__ import annotations

from typing import Literal

import torch
from comfy_api.latest import io

Operation = Literal["min", "max"]


def _single_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a Torch MASK tensor, got {type(mask).__name__}.")
    if mask.ndim == 2:
        return mask.unsqueeze(0)
    if mask.ndim == 3 and mask.shape[0] == 1:
        return mask
    raise ValueError(f"{name} must be [H,W] or [1,H,W], got {tuple(mask.shape)}.")


def pixelwise_masks(
    mask1: torch.Tensor, mask2: torch.Tensor, operation: Operation
) -> torch.Tensor:
    destination = _single_mask(mask1, "mask1").clone()
    source = _single_mask(mask2, "mask2").to(
        device=destination.device, dtype=destination.dtype
    )
    height = min(destination.shape[-2], source.shape[-2])
    width = min(destination.shape[-1], source.shape[-1])
    destination_portion = destination[:, :height, :width]
    source_portion = source[:, :height, :width]
    if operation == "min":
        destination[:, :height, :width] = torch.minimum(destination_portion, source_portion)
    else:
        destination[:, :height, :width] = torch.maximum(destination_portion, source_portion)
    return destination


def _schema(node_id: str, display_name: str) -> io.Schema:
    return io.Schema(
        node_id=node_id,
        display_name=display_name,
        category="Utility Suite/Mask",
        description="Compares two single masks at x=0, y=0 using mask1 as the output canvas.",
        inputs=[io.Mask.Input("mask1"), io.Mask.Input("mask2")],
        outputs=[io.Mask.Output("mask", display_name="mask")],
    )


class PixelwiseMinMaskAndMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _schema("UtilitySuitePixelwiseMinMaskAndMask", "Pixelwise Min (MASK & MASK)")

    @classmethod
    def execute(cls, mask1: torch.Tensor, mask2: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput(pixelwise_masks(mask1, mask2, "min"))


class PixelwiseMaxMaskAndMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _schema("UtilitySuitePixelwiseMaxMaskAndMask", "Pixelwise Max (MASK & MASK)")

    @classmethod
    def execute(cls, mask1: torch.Tensor, mask2: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput(pixelwise_masks(mask1, mask2, "max"))


__all__ = ["PixelwiseMaxMaskAndMask", "PixelwiseMinMaskAndMask", "pixelwise_masks"]
