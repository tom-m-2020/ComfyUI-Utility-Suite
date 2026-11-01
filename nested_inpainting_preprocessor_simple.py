from __future__ import annotations

import numpy as np
import scipy.ndimage
import torch
from comfy_api.latest import io

from .blur_mask_outward import blur_mask_outward
from .mask_batch_combine import combine_mask_batch
from .mask_bbox import quantized_mask_bounds
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


def _blur_for_area(
    mask: torch.Tensor,
    max_mask_size_a: int,
    max_mask_size_b: int,
    blur_a: float,
    blur_b: float,
    blur_c: float,
) -> float:
    bounds = quantized_mask_bounds(mask)
    if bounds is None:
        return blur_a
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    area = width * height
    if area > max_mask_size_b * max_mask_size_b:
        return blur_c
    if area > max_mask_size_a * max_mask_size_a:
        return blur_b
    return blur_a


def _blur_by_area(
    masks: torch.Tensor,
    max_mask_size_a: int,
    max_mask_size_b: int,
    blur_a: float,
    blur_b: float,
    blur_c: float,
    tapered_corners: bool,
) -> torch.Tensor:
    radii = [
        _blur_for_area(
            item,
            max_mask_size_a,
            max_mask_size_b,
            blur_a,
            blur_b,
            blur_c,
        )
        for item in masks
    ]
    result = torch.empty_like(masks)
    for radius in dict.fromkeys(radii):
        indices = [index for index, selected in enumerate(radii) if selected == radius]
        selection = torch.tensor(indices, dtype=torch.long)
        blurred = blur_mask_outward(
            torch.index_select(masks, 0, selection), radius, tapered_corners, True
        )
        result.index_copy_(0, selection, blurred)
    return result


def _shrink_masks(
    masks: torch.Tensor,
    amount: int,
    tapered_corners: bool,
) -> torch.Tensor:
    if amount < 0:
        raise ValueError(
            f"Nested Inpainting Preprocessor Simple shrink must be nonnegative, got {amount}."
        )
    if amount == 0:
        return masks
    corner = 0 if tapered_corners else 1
    footprint = np.array([[corner, 1, corner], [1, 1, 1], [corner, 1, corner]])
    results = []
    for item in masks:
        output = item.numpy()
        for _ in range(amount):
            output = scipy.ndimage.grey_erosion(output, footprint=footprint)
        results.append(torch.from_numpy(output))
    return torch.stack(results)


def nested_inpainting_preprocess_simple(
    mask: torch.Tensor,
    reorder_first: bool,
    tapered_corners: bool,
    max_mask_size_a: int,
    max_mask_size_b: int,
    blur_a: float,
    blur_b: float,
    blur_c: float,
    shrink: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    bool,
    torch.Tensor,
]:
    source = _validate_mask_batch(mask)
    if reorder_first:
        source = reorder_batch_mask(source, "biggest")
    source = (source > 0.0).to(torch.float32)
    source = reorder_batch_mask(_remove_later_overlaps(source), "biggest")
    reordered_init = source

    union_init = combine_mask_batch(source)
    bg_init = 1.0 - union_init
    blurred = _blur_by_area(
        source,
        max_mask_size_a,
        max_mask_size_b,
        blur_a,
        blur_b,
        blur_c,
        tapered_corners,
    )
    blurred = _shrink_masks(blurred, shrink, tapered_corners)

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
    return (
        result,
        mask_no_bg,
        union_init,
        bg_init,
        bg_blurred,
        all_covered,
        reordered_init,
    )


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
                io.Int.Input("max_mask_size_a", default=256, min=1, max=16384, step=1),
                io.Int.Input("max_mask_size_b", default=512, min=1, max=16384, step=1),
                io.Float.Input("blur_a", default=4.0, min=0.0, max=100.0, step=0.1),
                io.Float.Input("blur_b", default=6.0, min=0.0, max=100.0, step=0.1),
                io.Float.Input("blur_c", default=10.0, min=0.0, max=100.0, step=0.1),
                io.Int.Input("shrink", default=0, min=0, max=16384, step=1),
            ],
            outputs=[
                io.Mask.Output("mask", display_name="MASK"),
                io.Mask.Output("mask_no_bg", display_name="mask_no_bg"),
                io.Mask.Output("union_init", display_name="union_init"),
                io.Mask.Output("bg_init", display_name="bg_init"),
                io.Mask.Output("bg_blurred", display_name="bg_blurred"),
                io.Boolean.Output("all_covered", display_name="all_covered"),
                io.Mask.Output("reordered_init", display_name="reordered_init"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mask: torch.Tensor,
        reorder_first: bool,
        tapered_corners: bool,
        max_mask_size_a: int,
        max_mask_size_b: int,
        blur_a: float,
        blur_b: float,
        blur_c: float,
        shrink: int,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            *nested_inpainting_preprocess_simple(
                mask,
                reorder_first,
                tapered_corners,
                max_mask_size_a,
                max_mask_size_b,
                blur_a,
                blur_b,
                blur_c,
                shrink,
            )
        )


__all__ = [
    "COVERAGE_TOLERANCE",
    "NestedInpaintingPreprocessorSimple",
    "nested_inpainting_preprocess_simple",
]
