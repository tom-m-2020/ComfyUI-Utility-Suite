from __future__ import annotations

import torch
from comfy_api.latest import io


def combine_mask_batch(masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim == 2:
        return masks.unsqueeze(0)
    if masks.ndim != 3:
        raise ValueError(f"Mask Batch to Mask expects a rank-2 or rank-3 MASK, got shape {tuple(masks.shape)}.")
    if masks.shape[0] == 0:
        raise ValueError("Mask Batch to Mask requires at least one mask in the batch.")
    if masks.shape[1] == 0 or masks.shape[2] == 0:
        raise ValueError(f"Mask Batch to Mask requires nonempty spatial dimensions, got shape {tuple(masks.shape)}.")
    if masks.shape[0] == 1:
        return masks

    result = masks[0:1]
    for index in range(1, masks.shape[0]):
        result = torch.clamp(result + masks[index : index + 1], 0.0, 1.0)
    return result


class MaskBatchToMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteMaskBatchToMask",
            display_name="Mask Batch to Mask",
            category="Utility Suite/Mask",
            description="Combines every item in an ordinary MASK batch using core mask-add semantics.",
            inputs=[io.Mask.Input("masks")],
            outputs=[io.Mask.Output("mask", display_name="mask")],
        )

    @classmethod
    def execute(cls, masks: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput(combine_mask_batch(masks))
