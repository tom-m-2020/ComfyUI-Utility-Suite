from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from comfy_api.latest import io

from .combine_tiles_segs import SEGS, _cropped_mask
from .segs_geometry import validated_segs

Operation = Literal["min", "max"]


def _combine(destination: np.ndarray, source: np.ndarray, operation: Operation) -> None:
    if operation == "min":
        np.minimum(destination, source, out=destination)
    else:
        np.maximum(destination, source, out=destination)


def _full_canvas_masks(
    segs: Any, node_name: str, operation: Operation
) -> tuple[Any, list[Any] | tuple[Any, ...], list[tuple[int, int, int, int]], list[np.ndarray]]:
    header, entries, crops = validated_segs(segs, node_name)
    height, width = header
    masks = []
    for index, (entry, crop) in enumerate(zip(entries, crops)):
        x1, y1, x2, y2 = crop
        fill_value = 1.0 if operation == "min" else 0.0
        canvas = np.full((height, width), fill_value, dtype=np.float32)
        canvas[y1:y2, x1:x2] = _cropped_mask(entry, crop, index)
        masks.append(canvas)
    return header, entries, crops, masks


def _replace_masks(
    header: Any,
    entries: list[Any] | tuple[Any, ...],
    crops: list[tuple[int, int, int, int]],
    masks: list[np.ndarray],
) -> tuple[Any, list[Any]]:
    output = []
    for index, (entry, crop, mask) in enumerate(zip(entries, crops, masks)):
        if not hasattr(entry, "_replace"):
            raise TypeError(f"SEGS entry {index} must support structural cropped_mask replacement.")
        x1, y1, x2, y2 = crop
        output.append(entry._replace(cropped_mask=mask[y1:y2, x1:x2].copy()))
    return header, output


def pixelwise_in_segs(segs: Any, operation: Operation) -> tuple[Any, list[Any]]:
    node_name = f"Pixelwise {operation.title()} (in SEGS)"
    header, entries, crops, full_masks = _full_canvas_masks(segs, node_name, operation)
    if not full_masks:
        return header, []
    combined = full_masks[0].copy()
    for source in full_masks[1:]:
        _combine(combined, source, operation)
    return _replace_masks(header, entries, crops, [combined] * len(entries))


def pixelwise_segs_and_segs(segs1: Any, segs2: Any, operation: Operation) -> tuple[Any, list[Any]]:
    node_name = f"Pixelwise {operation.title()} (SEGS & SEGS)"
    header, entries, crops, destination_masks = _full_canvas_masks(
        segs1, f"{node_name} segs1", operation
    )
    source_header, _, _, source_masks = _full_canvas_masks(
        segs2, f"{node_name} segs2", operation
    )
    if tuple(header) != tuple(source_header):
        raise ValueError(
            f"{node_name} requires matching SEGS source shapes, got {tuple(header)} and {tuple(source_header)}."
        )
    output_masks = []
    for destination in destination_masks:
        combined = destination.copy()
        for source in source_masks:
            _combine(combined, source, operation)
        output_masks.append(combined)
    return _replace_masks(header, entries, crops, output_masks)


def _single_full_canvas_mask(mask: Any, header: Any, node_name: str) -> np.ndarray:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{node_name} mask must be a Torch MASK tensor, got {type(mask).__name__}.")
    if mask.ndim == 2:
        single = mask
    elif mask.ndim == 3 and mask.shape[0] == 1:
        single = mask[0]
    else:
        raise ValueError(f"{node_name} mask must be [H,W] or [1,H,W], got {tuple(mask.shape)}.")
    if tuple(single.shape) != tuple(header):
        raise ValueError(
            f"{node_name} mask canvas must match the SEGS source shape {tuple(header)}, got {tuple(single.shape)}."
        )
    result = single.detach().cpu().numpy().astype(np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{node_name} mask values must be finite.")
    return np.clip(result, 0.0, 1.0)


def pixelwise_segs_and_mask(segs: Any, mask: Any, operation: Operation) -> tuple[Any, list[Any]]:
    node_name = f"Pixelwise {operation.title()} (SEGS & MASK)"
    header, entries, crops, destination_masks = _full_canvas_masks(segs, node_name, operation)
    source = _single_full_canvas_mask(mask, header, node_name)
    for destination in destination_masks:
        _combine(destination, source, operation)
    return _replace_masks(header, entries, crops, destination_masks)


def _schema(node_id: str, display_name: str, inputs: list[Any]) -> io.Schema:
    return io.Schema(
        node_id=node_id,
        display_name=display_name,
        category="Utility Suite/SEGS",
        description="Applies a coordinate-correct full-canvas pixelwise mask operation and preserves SEGS structure.",
        inputs=inputs,
        outputs=[SEGS.Output("segs", display_name="segs")],
    )


class PixelwiseMinInSEGS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _schema("UtilitySuitePixelwiseMinInSEGS", "Pixelwise Min (in SEGS)", [SEGS.Input("segs")])

    @classmethod
    def execute(cls, segs: Any) -> io.NodeOutput:
        return io.NodeOutput(pixelwise_in_segs(segs, "min"))


class PixelwiseMaxInSEGS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _schema("UtilitySuitePixelwiseMaxInSEGS", "Pixelwise Max (in SEGS)", [SEGS.Input("segs")])

    @classmethod
    def execute(cls, segs: Any) -> io.NodeOutput:
        return io.NodeOutput(pixelwise_in_segs(segs, "max"))


class PixelwiseMinSEGSAndSEGS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _schema(
            "UtilitySuitePixelwiseMinSEGSAndSEGS",
            "Pixelwise Min (SEGS & SEGS)",
            [SEGS.Input("segs1"), SEGS.Input("segs2")],
        )

    @classmethod
    def execute(cls, segs1: Any, segs2: Any) -> io.NodeOutput:
        return io.NodeOutput(pixelwise_segs_and_segs(segs1, segs2, "min"))


class PixelwiseMaxSEGSAndSEGS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _schema(
            "UtilitySuitePixelwiseMaxSEGSAndSEGS",
            "Pixelwise Max (SEGS & SEGS)",
            [SEGS.Input("segs1"), SEGS.Input("segs2")],
        )

    @classmethod
    def execute(cls, segs1: Any, segs2: Any) -> io.NodeOutput:
        return io.NodeOutput(pixelwise_segs_and_segs(segs1, segs2, "max"))


class PixelwiseMinSEGSAndMASK(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _schema(
            "UtilitySuitePixelwiseMinSEGSAndMASK",
            "Pixelwise Min (SEGS & MASK)",
            [SEGS.Input("segs"), io.Mask.Input("mask")],
        )

    @classmethod
    def execute(cls, segs: Any, mask: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput(pixelwise_segs_and_mask(segs, mask, "min"))


class PixelwiseMaxSEGSAndMASK(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return _schema(
            "UtilitySuitePixelwiseMaxSEGSAndMASK",
            "Pixelwise Max (SEGS & MASK)",
            [SEGS.Input("segs"), io.Mask.Input("mask")],
        )

    @classmethod
    def execute(cls, segs: Any, mask: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput(pixelwise_segs_and_mask(segs, mask, "max"))


__all__ = [
    "PixelwiseMaxInSEGS",
    "PixelwiseMaxSEGSAndMASK",
    "PixelwiseMaxSEGSAndSEGS",
    "PixelwiseMinInSEGS",
    "PixelwiseMinSEGSAndMASK",
    "PixelwiseMinSEGSAndSEGS",
    "pixelwise_in_segs",
    "pixelwise_segs_and_mask",
    "pixelwise_segs_and_segs",
]
