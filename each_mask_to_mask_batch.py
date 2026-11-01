from __future__ import annotations

import comfy.utils
import torch
from comfy_api.latest import io

from .mask_batch import mask_to_mask_batch


def _normalize_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(
            f"Each Mask to Mask Batch {name} must be a torch.Tensor MASK, got {type(mask).__name__}."
        )
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(
            f"Each Mask to Mask Batch {name} expects MASK [B,H,W], got shape {tuple(mask.shape)}."
        )
    if any(dimension == 0 for dimension in mask.shape):
        raise ValueError(
            f"Each Mask to Mask Batch {name} requires nonempty dimensions, got shape {tuple(mask.shape)}."
        )
    return mask


def _mask_index(name: str) -> int:
    return int(name.removeprefix("mask"))


def each_mask_to_mask_batch(
    masks: dict[str, torch.Tensor],
    bbox_fill: bool,
    drop_size: int,
    contour_fill: bool,
) -> torch.Tensor:
    if not masks:
        raise ValueError("Each Mask to Mask Batch requires mask0.")

    ordered = [
        _normalize_mask(masks[name], name) for name in sorted(masks, key=_mask_index)
    ]
    height, width = ordered[0].shape[1:]
    resized = []
    for mask in ordered:
        if mask.shape[1:] != (height, width):
            mask = comfy.utils.common_upscale(
                mask.unsqueeze(1), width, height, "bilinear", "center"
            ).squeeze(1)
        resized.append(mask)

    results = [
        mask_to_mask_batch(item.unsqueeze(0), bbox_fill, drop_size, contour_fill)
        for batch in resized
        for item in batch
    ]
    return torch.cat(results, dim=0)


class EachMaskToMaskBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        masks = io.Autogrow.TemplatePrefix(
            io.Mask.Input("mask"),
            prefix="mask",
            min=1,
            max=100,
        )
        return io.Schema(
            node_id="UtilitySuiteEachMaskToMaskBatch",
            display_name="Each Mask to Mask Batch",
            category="Utility Suite/Mask",
            description="Separates every connected mask into contours and combines the results into one MASK batch.",
            inputs=[
                io.Autogrow.Input("masks", template=masks),
                io.Boolean.Input(
                    "bbox_fill",
                    default=False,
                    label_on="enabled",
                    label_off="disabled",
                ),
                io.Int.Input("drop_size", default=10, min=1, max=16384, step=1),
                io.Boolean.Input(
                    "contour_fill",
                    default=False,
                    label_on="enabled",
                    label_off="disabled",
                ),
            ],
            outputs=[io.Mask.Output("masks", display_name="masks")],
        )

    @classmethod
    def execute(
        cls,
        masks: io.Autogrow.Type,
        bbox_fill: bool,
        drop_size: int,
        contour_fill: bool,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            each_mask_to_mask_batch(masks, bbox_fill, drop_size, contour_fill)
        )


__all__ = ["EachMaskToMaskBatch", "each_mask_to_mask_batch"]
