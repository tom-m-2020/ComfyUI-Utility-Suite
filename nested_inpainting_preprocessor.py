from __future__ import annotations

import numpy as np
import scipy.ndimage
import torch
from comfy_api.latest import io

from .blur_mask_outward import blur_mask_outward
from .mask_batch_combine import combine_mask_batch
from .mask_bbox import quantized_mask_bounds
from .mask_fill_holes import fill_mask_holes
from .reorder_batch_mask import reorder_batch_mask


def _validate_mask_batch(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"Nested Inpainting Preprocessor expects a torch.Tensor MASK, got {type(mask).__name__}.")
    if mask.ndim != 3:
        raise ValueError(f"Nested Inpainting Preprocessor expects MASK [B,H,W], got shape {tuple(mask.shape)}.")
    if any(dimension == 0 for dimension in mask.shape):
        raise ValueError(f"Nested Inpainting Preprocessor requires nonempty dimensions, got shape {tuple(mask.shape)}.")
    return mask.detach().to(device="cpu", dtype=torch.float32)


def _threshold(mask: torch.Tensor, value: float) -> torch.Tensor:
    return (mask > value).to(torch.float32)


def _add(destination: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    return torch.clamp(destination + source, 0.0, 1.0)


def _subtract(destination: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    return torch.clamp(destination - source, 0.0, 1.0)


def _pixelwise_and(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    first = np.asarray(mask1.detach().cpu(), dtype=np.float32)
    second = np.asarray(mask2.detach().cpu(), dtype=np.float32)
    if first.shape != second.shape:
        raise ValueError(f"Pixelwise masks must have equal shapes, got {first.shape} and {second.shape}.")
    return torch.from_numpy(np.array(np.bitwise_and(first.view(np.int32), second.view(np.int32)).view(np.float32), copy=True))


def _grow_one_tapered(mask: torch.Tensor) -> torch.Tensor:
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    results = []
    for item in mask:
        grown = scipy.ndimage.grey_dilation(item.numpy(), footprint=kernel)
        results.append(torch.from_numpy(np.array(grown, dtype=np.float32, copy=True)))
    return torch.stack(results)


def _is_empty(mask: torch.Tensor) -> bool:
    return not bool(torch.any(mask != 0).item())


def _blur_for_area(
    mask: torch.Tensor,
    blur_a: float,
    blur_b: float,
    blur_c: float,
    blur_a_max_res: int,
    blur_b_max_res: int,
) -> float:
    bounds = quantized_mask_bounds(mask[0])
    if bounds is None:
        raise ValueError("Nested Inpainting Preprocessor cannot select a blur radius for an empty mask.")
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    area = width * height
    if area > blur_b_max_res * blur_b_max_res:
        return blur_c
    if area > blur_a_max_res * blur_a_max_res:
        return blur_b
    return blur_a


def _suffix_unions(masks: torch.Tensor) -> list[torch.Tensor]:
    suffix = [torch.empty(0)] * masks.shape[0]
    running = masks[-1:]
    suffix[-1] = running
    for index in range(masks.shape[0] - 2, -1, -1):
        running = _add(masks[index : index + 1], running)
        suffix[index] = running
    return suffix


def _remove_later_overlaps(masks: torch.Tensor) -> torch.Tensor:
    if masks.shape[0] == 1:
        return masks
    suffix = _suffix_unions(masks)
    results = [_subtract(masks[index : index + 1], suffix[index + 1]) for index in range(masks.shape[0] - 1)]
    results.append(masks[-1:])
    return torch.cat(results, dim=0)


def _inward_blurred_prefix(
    prefix_union: torch.Tensor,
    blur_radius: float,
    tapered_corners: bool,
) -> torch.Tensor:
    inverted = 1.0 - prefix_union
    blurred = blur_mask_outward(inverted, blur_radius, tapered_corners, True)
    return 1.0 - blurred


def _initial_counter_masks(
    masks: torch.Tensor,
    bg_init: torch.Tensor,
    tapered_corners: bool,
    counter_blurs: tuple[float, float, float],
    blur_a_max_res: int,
    blur_b_max_res: int,
    bg_blur_overlap_resistance: float,
) -> torch.Tensor:
    grown_background = _grow_one_tapered(bg_init)
    selected = [
        index
        for index in range(masks.shape[0])
        if not _is_empty(_pixelwise_and(masks[index : index + 1], grown_background))
    ]
    if not selected:
        return masks

    counter_radius = _blur_for_area(
        bg_init,
        *counter_blurs,
        blur_a_max_res,
        blur_b_max_res,
    )
    suffix = _suffix_unions(masks)
    filled = fill_mask_holes(masks)
    filled_prefix = []
    running = filled[0:1]
    filled_prefix.append(running)
    for index in range(1, filled.shape[0]):
        running = _add(running, filled[index : index + 1])
        filled_prefix.append(running)

    counters = [item.unsqueeze(0) for item in masks]
    for index in selected:
        counter = blur_mask_outward(masks[index : index + 1], counter_radius, tapered_corners, True)
        if index > 0:
            resistance = _inward_blurred_prefix(
                filled_prefix[index - 1],
                bg_blur_overlap_resistance,
                tapered_corners,
            )
            counter = _subtract(counter, resistance)
        if index + 1 < masks.shape[0]:
            counter = _subtract(counter, suffix[index + 1])
        counters[index] = counter
    return torch.cat(counters, dim=0)


def _cached_dummy_blur(
    masks: torch.Tensor,
    index: int,
    radius: float,
    tapered_corners: bool,
    cache: dict[tuple[int, float, bool], torch.Tensor],
) -> torch.Tensor:
    key = (index, float(radius), tapered_corners)
    encoded = cache.get(key)
    if encoded is None:
        blurred = blur_mask_outward(masks[index : index + 1], radius, tapered_corners, True)
        encoded = torch.clamp(torch.round(blurred * 255.0), 0, 255).to(torch.uint8)
        cache[key] = encoded
    return encoded.to(torch.float32) / 255.0


def _mold_and_bridge(
    counter_init: torch.Tensor,
    tapered_corners: bool,
    inner_blurred_mask: torch.Tensor,
    pixelwise: torch.Tensor,
    thresholded_dummy: torch.Tensor,
    dummy_blur_radius: float,
    blurred_dummy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded_match = blur_mask_outward(pixelwise, dummy_blur_radius, tapered_corners, True)
    grown_bridge = _threshold(expanded_match, 0.0)
    pixelwised_bridge = _pixelwise_and(thresholded_dummy, grown_bridge)

    pre_mold = _add(_threshold(inner_blurred_mask, 0.99), counter_init)
    inverse_mold = _add(pre_mold, pixelwised_bridge)
    mold = _subtract(torch.ones_like(inverse_mold), inverse_mold)

    cut_by_dummy = _subtract(pixelwised_bridge, _threshold(blurred_dummy, 0.99))
    grown_cut = _grow_one_tapered(cut_by_dummy)
    cut_by_counter = _subtract(pixelwised_bridge, counter_init)
    post_bridge = _add(grown_cut, cut_by_counter)
    return mold, post_bridge


def _update_future_counter(
    existing_counter: torch.Tensor,
    counter_init: torch.Tensor,
    pixelwise: torch.Tensor,
    thresholded_dummy: torch.Tensor,
    blurred_dummy: torch.Tensor,
    inner_blurred_mask: torch.Tensor,
    dummy_blur_radius: float,
    counter_blur_radius: float,
    counter_threshold: float,
    tapered_corners: bool,
) -> torch.Tensor:
    mold, bridge = _mold_and_bridge(
        counter_init,
        tapered_corners,
        inner_blurred_mask,
        pixelwise,
        thresholded_dummy,
        dummy_blur_radius,
        blurred_dummy,
    )
    counter = _subtract(_threshold(blurred_dummy, counter_threshold), mold)
    counter = _pixelwise_and(counter, bridge)
    counter = blur_mask_outward(counter, counter_blur_radius, tapered_corners, True)
    return _add(_subtract(existing_counter, counter), counter)


def nested_inpainting_preprocess(
    mask: torch.Tensor,
    reorder_first: bool,
    tapered_corners: bool,
    counter_threshold: float,
    blur_a_max_res: int,
    blur_b_max_res: int,
    dummy_blur_a: float,
    dummy_blur_b: float,
    dummy_blur_c: float,
    counter_blur_a: float,
    counter_blur_b: float,
    counter_blur_c: float,
    bg_blur_overlap_resistance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, torch.Tensor, torch.Tensor, torch.Tensor, list[int], int, torch.Tensor]:
    source = _validate_mask_batch(mask)
    if reorder_first:
        source = reorder_batch_mask(source, "biggest")
    source = _threshold(source, 0.0)
    source = reorder_batch_mask(_remove_later_overlaps(source), "biggest")

    union_init = combine_mask_batch(source)
    bg_init = 1.0 - union_init
    counters = _initial_counter_masks(
        source,
        bg_init,
        tapered_corners,
        (counter_blur_a, counter_blur_b, counter_blur_c),
        blur_a_max_res,
        blur_b_max_res,
        bg_blur_overlap_resistance,
    )
    bg_blurred = 1.0 - combine_mask_batch(counters)

    height, width = source.shape[1:]
    mask_and_bg = torch.empty((source.shape[0] + 1, height, width), dtype=torch.float32)
    mask_and_bg[0:1].copy_(bg_blurred)
    bg_blurred = mask_and_bg[0:1]
    empty_batch = torch.empty((0, height, width), dtype=torch.float32)
    zeroth_dummy = empty_batch
    zeroth_minus_init = counters[0:1]
    zeroth_pixelwise = empty_batch
    zeroth_match_index: list[int] = []
    zeroth_match_count = 0
    zeroth_counter_mask = empty_batch

    carried = [item.unsqueeze(0) for item in counters]
    dummy_cache: dict[tuple[int, float, bool], torch.Tensor] = {}
    source_suffix = _suffix_unions(source)

    for outer_index in range(source.shape[0] - 1):
        current = carried[0]
        future = carried[1:]
        dummy_radius = _blur_for_area(
            current,
            dummy_blur_a,
            dummy_blur_b,
            dummy_blur_c,
            blur_a_max_res,
            blur_b_max_res,
        )
        counter_radius = _blur_for_area(
            current,
            counter_blur_a,
            counter_blur_b,
            counter_blur_c,
            blur_a_max_res,
            blur_b_max_res,
        )
        dummies = [
            _cached_dummy_blur(source, index, dummy_radius, tapered_corners, dummy_cache)
            for index in range(outer_index + 1, source.shape[0])
        ]
        dummy_batch = torch.cat(dummies, dim=0)
        thresholded_dummies = _threshold(dummy_batch, 0.0)
        inner_blurred = _subtract(current, combine_mask_batch(dummy_batch))
        minus_init = _subtract(current, source_suffix[outer_index + 1])
        probe = _grow_one_tapered(_threshold(minus_init, 0.0))
        pixelwise = torch.cat(
            [_pixelwise_and(thresholded_dummies[index : index + 1], probe) for index in range(len(future))],
            dim=0,
        )
        matches = [index for index in range(len(future)) if not _is_empty(pixelwise[index : index + 1])]

        for relative_index in matches:
            source_index = outer_index + 1 + relative_index
            future[relative_index] = _update_future_counter(
                future[relative_index],
                source[source_index : source_index + 1],
                pixelwise[relative_index : relative_index + 1],
                thresholded_dummies[relative_index : relative_index + 1],
                dummy_batch[relative_index : relative_index + 1],
                inner_blurred,
                dummy_radius,
                counter_radius,
                counter_threshold,
                tapered_corners,
            )

        if outer_index == 0:
            zeroth_dummy = dummy_batch
            zeroth_minus_init = minus_init
            zeroth_pixelwise = pixelwise
            zeroth_match_index = matches
            zeroth_match_count = len(matches)
            zeroth_counter_mask = torch.cat(future, dim=0)

        mask_and_bg[outer_index + 1 : outer_index + 2].copy_(inner_blurred)
        carried = future
        for key in [key for key in dummy_cache if key[0] <= outer_index + 1]:
            del dummy_cache[key]

    mask_and_bg[-1:].copy_(carried[0])
    result = mask_and_bg[1:]
    covered = _add(bg_init, combine_mask_batch(result))
    all_covered = _is_empty(1.0 - covered)
    return (
        result,
        mask_and_bg,
        union_init,
        bg_init,
        bg_blurred,
        all_covered,
        zeroth_dummy,
        zeroth_minus_init,
        zeroth_pixelwise,
        zeroth_match_index,
        zeroth_match_count,
        zeroth_counter_mask,
    )


class NestedInpaintingPreprocessor(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteNestedInpaintingPreprocessor",
            display_name="Nested Inpainting Preprocessor",
            category="Utility Suite/Mask",
            description="Builds ordered target and counter masks for sequential nested inpainting.",
            inputs=[
                io.Mask.Input("mask"),
                io.Boolean.Input("reorder_first", default=False),
                io.Boolean.Input("tapered_corners", default=True),
                io.Float.Input("counter_threshold", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("blur_a_max_res", default=256, min=1, max=16384, step=1),
                io.Int.Input("blur_b_max_res", default=512, min=1, max=16384, step=1),
                io.Float.Input("dummy_blur_a", default=4.0, min=0.0, max=100.0, step=0.1),
                io.Float.Input("dummy_blur_b", default=6.0, min=0.0, max=100.0, step=0.1),
                io.Float.Input("dummy_blur_c", default=10.0, min=0.0, max=100.0, step=0.1),
                io.Float.Input("counter_blur_a", default=4.0, min=0.0, max=100.0, step=0.1),
                io.Float.Input("counter_blur_b", default=6.0, min=0.0, max=100.0, step=0.1),
                io.Float.Input("counter_blur_c", default=10.0, min=0.0, max=100.0, step=0.1),
                io.Float.Input(
                    "bg_blur_overlap_resistance",
                    default=4.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                ),
            ],
            outputs=[
                io.Mask.Output("mask", display_name="MASK"),
                io.Mask.Output("mask_and_bg", display_name="mask_and_bg"),
                io.Mask.Output("union_init", display_name="union_init"),
                io.Mask.Output("bg_init", display_name="bg_init"),
                io.Mask.Output("bg_blurred", display_name="bg_blurred"),
                io.Boolean.Output("all_covered", display_name="all_covered"),
                io.Mask.Output("zeroth_dummy", display_name="zeroth_dummy"),
                io.Mask.Output("zeroth_minus_init", display_name="zeroth_minus_init"),
                io.Mask.Output("zeroth_pixelwise", display_name="zeroth_pixelwise"),
                io.Int.Output("zeroth_match_index", display_name="zeroth_match_index", is_output_list=True),
                io.Int.Output("zeroth_match_count", display_name="zeroth_match_count"),
                io.Mask.Output("zeroth_counter_mask", display_name="zeroth_counter_mask"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mask: torch.Tensor,
        reorder_first: bool,
        tapered_corners: bool,
        counter_threshold: float,
        blur_a_max_res: int,
        blur_b_max_res: int,
        dummy_blur_a: float,
        dummy_blur_b: float,
        dummy_blur_c: float,
        counter_blur_a: float,
        counter_blur_b: float,
        counter_blur_c: float,
        bg_blur_overlap_resistance: float,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            *nested_inpainting_preprocess(
                mask,
                reorder_first,
                tapered_corners,
                counter_threshold,
                blur_a_max_res,
                blur_b_max_res,
                dummy_blur_a,
                dummy_blur_b,
                dummy_blur_c,
                counter_blur_a,
                counter_blur_b,
                counter_blur_c,
                bg_blur_overlap_resistance,
            )
        )


__all__ = ["NestedInpaintingPreprocessor", "nested_inpainting_preprocess"]
