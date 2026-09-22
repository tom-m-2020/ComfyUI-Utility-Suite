from __future__ import annotations

from typing import Any

import torch
from comfy_api.latest import io

from .filter_tile_segs import _mask_array
from .segs_geometry import integer, validated_segs

SEGS = io.Custom("SEGS")
OPERATIONS = ("multiply", "add", "subtract", "and", "or", "xor")


def _single_mask(mask: Any, description: str) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{description} must be a Torch MASK tensor, got {type(mask).__name__}.")
    if mask.ndim == 2:
        result = mask
    elif mask.ndim == 3 and mask.shape[0] == 1:
        result = mask[0]
    else:
        raise ValueError(f"{description} must be [H,W] or [1,H,W], got {tuple(mask.shape)}.")
    if result.shape[0] <= 0 or result.shape[1] <= 0:
        raise ValueError(f"{description} must have nonempty spatial dimensions, got {tuple(result.shape)}.")
    return result


def normalize_mask_sources(masks: Any) -> list[torch.Tensor]:
    if isinstance(masks, torch.Tensor):
        if masks.ndim == 2:
            return [_single_mask(masks, "source MASK")]
        if masks.ndim == 3:
            return [_single_mask(mask, f"source MASK batch item {index}") for index, mask in enumerate(masks)]
        raise ValueError(f"source MASK must be [H,W] or [B,H,W], got {tuple(masks.shape)}.")
    if not isinstance(masks, (list, tuple)):
        raise TypeError(f"source MASK must be a tensor or Comfy MASK list, got {type(masks).__name__}.")
    if len(masks) == 1 and isinstance(masks[0], torch.Tensor) and masks[0].ndim == 3:
        return normalize_mask_sources(masks[0])
    return [_single_mask(mask, f"source MASK list item {index}") for index, mask in enumerate(masks)]


def mask_composite(
    destination: torch.Tensor, source: torch.Tensor, x: int, y: int, operation: str
) -> torch.Tensor:
    if operation not in OPERATIONS:
        raise ValueError(f"Unsupported mask operation {operation!r}.")
    output = destination.reshape((-1, destination.shape[-2], destination.shape[-1])).clone()
    source = source.reshape((-1, source.shape[-2], source.shape[-1])).to(output.device)
    left, top = x, y
    right = min(left + source.shape[-1], destination.shape[-1])
    bottom = min(top + source.shape[-2], destination.shape[-2])
    visible_width, visible_height = right - left, bottom - top
    source_portion = source[:, :visible_height, :visible_width]
    destination_portion = output[:, top:bottom, left:right]
    if operation == "multiply":
        combined = destination_portion * source_portion
    elif operation == "add":
        combined = destination_portion + source_portion
    elif operation == "subtract":
        combined = destination_portion - source_portion
    elif operation == "and":
        combined = torch.bitwise_and(destination_portion.round().bool(), source_portion.round().bool()).float()
    elif operation == "or":
        combined = torch.bitwise_or(destination_portion.round().bool(), source_portion.round().bool()).float()
    else:
        combined = torch.bitwise_xor(destination_portion.round().bool(), source_portion.round().bool()).float()
    output[:, top:bottom, left:right] = combined
    return torch.clamp(output, 0.0, 1.0)


def _combine_destination(
    segs: Any, sources: list[torch.Tensor], x: int, y: int, operation: str, node_name: str
) -> tuple[Any, list[Any]]:
    header, entries, crops = validated_segs(segs, node_name)
    x = integer(x, "x")
    y = integer(y, "y")
    if x < 0 or y < 0:
        raise ValueError(f"{node_name} x and y must be nonnegative, got {(x, y)}.")
    if operation not in OPERATIONS:
        raise ValueError(f"Unsupported {node_name} operation {operation!r}.")
    if not entries:
        return header, []
    if not sources:
        raise ValueError(f"{node_name} requires at least one source mask for nonempty destination SEGS.")
    result: list[Any] = []
    for index, (entry, crop) in enumerate(zip(entries, crops)):
        if not hasattr(entry, "_replace") or not hasattr(entry, "cropped_mask"):
            raise TypeError(f"SEGS entry {index} must support structural cropped_mask replacement.")
        x1, y1, x2, y2 = crop
        destination = torch.from_numpy(_mask_array(entry, index, y2 - y1, x2 - x1))
        source = sources[min(index, len(sources) - 1)]
        combined = mask_composite(destination, source, x, y, operation)[0].detach().cpu().numpy().copy()
        result.append(entry._replace(cropped_mask=combined))
    return header, result


def combine_segs_and_mask(
    segs: Any, masks: Any, x: int, y: int, operation: str
) -> tuple[Any, list[Any]]:
    # Normalize lazily so an empty destination remains valid with an empty source.
    header, entries, _ = validated_segs(segs, "Combine Masks (SEGS & MASK)")
    if not entries:
        return header, []
    return _combine_destination(
        segs, normalize_mask_sources(masks), x, y, operation, "Combine Masks (SEGS & MASK)"
    )


def combine_segs_and_segs(
    segs_destination: Any, segs_source: Any, x: int, y: int, operation: str
) -> tuple[Any, list[Any]]:
    header, destination_entries, _ = validated_segs(segs_destination, "Combine Masks (SEGS & SEGS) destination")
    if not destination_entries:
        return header, []
    _, source_entries, source_crops = validated_segs(segs_source, "Combine Masks (SEGS & SEGS) source")
    sources = []
    for index, (entry, crop) in enumerate(zip(source_entries, source_crops)):
        x1, y1, x2, y2 = crop
        sources.append(torch.from_numpy(_mask_array(entry, index, y2 - y1, x2 - x1)))
    return _combine_destination(
        segs_destination, sources, x, y, operation, "Combine Masks (SEGS & SEGS)"
    )


def _schema(node_id: str, display_name: str, inputs: list[Any], *, is_input_list: bool = False) -> io.Schema:
    return io.Schema(
        node_id=node_id,
        display_name=display_name,
        category="Utility Suite/SEGS",
        is_input_list=is_input_list,
        inputs=inputs
        + [
            io.Int.Input("x", default=0, min=0, max=16384),
            io.Int.Input("y", default=0, min=0, max=16384),
            io.Combo.Input("operation", options=list(OPERATIONS)),
        ],
        outputs=[SEGS.Output("segs", display_name="segs")],
    )


class CombineMasksSEGSAndMASK(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _schema(
            "UtilitySuiteCombineMasksSEGSAndMASK",
            "Combine Masks (SEGS & MASK)",
            [SEGS.Input("segs"), io.Mask.Input("masks")],
            is_input_list=True,
        )

    @classmethod
    def execute(cls, segs: list[Any], masks: list[Any], x: list[int], y: list[int], operation: list[str]) -> io.NodeOutput:
        return io.NodeOutput(combine_segs_and_mask(segs[0], masks, x[0], y[0], operation[0]))


class CombineMasksSEGSAndSEGS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _schema(
            "UtilitySuiteCombineMasksSEGSAndSEGS",
            "Combine Masks (SEGS & SEGS)",
            [SEGS.Input("segs_destination"), SEGS.Input("segs_source")],
        )

    @classmethod
    def execute(
        cls, segs_destination: Any, segs_source: Any, x: int, y: int, operation: str
    ) -> io.NodeOutput:
        return io.NodeOutput(combine_segs_and_segs(segs_destination, segs_source, x, y, operation))
