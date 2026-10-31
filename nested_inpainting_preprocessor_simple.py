from __future__ import annotations

import torch
from comfy_api.latest import io

from .blur_mask_outward import blur_mask_outward
from .mask_batch_combine import combine_mask_batch
from .reorder_batch_mask import reorder_batch_mask

COVERAGE_TOLERANCE = 1e-6


def _validate_mask_batch(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(
            f"Nested Inpainting Preprocessor Simple expects a torch.Tensor MASK, got {type(mask).__name__}."
        )
    if mask.ndim != 3:
        raise ValueError(
            f"Nested Inpainting Preprocessor Simple expects MASK [B,H,W], got shape {tuple(mask.shape)}."
        )
    if any(dimension == 0 for dimension in mask.shape):
        raise ValueError(
            f"Nested Inpainting Preprocessor Simple requires nonempty dimensions, got shape {tuple(mask.shape)}."
        )
    return mask.detach().to(device="cpu", dtype=torch.float32)


def _remove_later_overlaps(masks: torch.Tensor) -> torch.Tensor:
    if masks.shape[0] == 1:
        return masks

    suffix_unions = torch.empty_like(masks)
    suffix_unions[-1:].copy_(masks[-1:])
    for index in range(masks.shape[0] - 2, -1, -1):
        torch.maximum(
            masks[index : index + 1],
            suffix_unions[index + 1 : index + 2],
            out=suffix_unions[index : index + 1],
        )

    result = torch.empty_like(masks)
    result[:-1].copy_(torch.clamp(masks[:-1] - suffix_unions[1:], 0.0, 1.0))
    result[-1:].copy_(masks[-1:])
    return result


def _coverage_is_complete(bg_init: torch.Tensor, masks: torch.Tensor) -> bool:
    covered = torch.clamp(bg_init + combine_mask_batch(masks), 0.0, 1.0)
    return bool(torch.all((1.0 - covered) <= COVERAGE_TOLERANCE).item())


def nested_inpainting_preprocess_simple(
    mask: torch.Tensor,
    reorder_first: bool,
    tapered_corners: bool,
    blur: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    source = _validate_mask_batch(mask)
    if reorder_first:
        source = reorder_batch_mask(source, "biggest")
    source = (source > 0.0).to(torch.float32)
    source = reorder_batch_mask(_remove_later_overlaps(source), "biggest")

    union_init = combine_mask_batch(source)
    bg_init = 1.0 - union_init
    blurred = blur_mask_outward(source, blur, tapered_corners, True)

    values = torch.cat((bg_init, blurred), dim=0)
    supports = values > 0.0
    result = torch.empty_like(values)
    for index in range(values.shape[0] - 1):
        matches = torch.any(
            supports[index + 1 :] & supports[index : index + 1], dim=(1, 2)
        )
        if bool(torch.any(matches).item()):
            later_union = torch.amax(values[index + 1 :][matches], dim=0, keepdim=True)
            result[index : index + 1].copy_(
                torch.clamp(values[index : index + 1] - later_union, 0.0, 1.0)
            )
        else:
            result[index : index + 1].copy_(values[index : index + 1])
    result[-1:].copy_(values[-1:])

    mask_no_bg = result[1:]
    bg_blurred = result[0:1]
    all_covered = _coverage_is_complete(bg_init, result)
    return result, mask_no_bg, union_init, bg_init, bg_blurred, all_covered


class NestedInpaintingPreprocessorSimple(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteNestedInpaintingPreprocessorSimple",
            display_name="Nested Inpainting Preprocessor Simple",
            category="Utility Suite/Mask",
            description="Builds a compact fixed-blur mask partition for sequential nested inpainting.",
            inputs=[
                io.Mask.Input("mask"),
                io.Boolean.Input("reorder_first", default=False),
                io.Boolean.Input("tapered_corners", default=True),
                io.Float.Input("blur", default=6.0, min=0.0, max=100.0, step=0.1),
            ],
            outputs=[
                io.Mask.Output("mask", display_name="MASK"),
                io.Mask.Output("mask_no_bg", display_name="mask_no_bg"),
                io.Mask.Output("union_init", display_name="union_init"),
                io.Mask.Output("bg_init", display_name="bg_init"),
                io.Mask.Output("bg_blurred", display_name="bg_blurred"),
                io.Boolean.Output("all_covered", display_name="all_covered"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mask: torch.Tensor,
        reorder_first: bool,
        tapered_corners: bool,
        blur: float,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            *nested_inpainting_preprocess_simple(
                mask, reorder_first, tapered_corners, blur
            )
        )


__all__ = [
    "COVERAGE_TOLERANCE",
    "NestedInpaintingPreprocessorSimple",
    "nested_inpainting_preprocess_simple",
]
