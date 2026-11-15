from __future__ import annotations

from typing import Any

import torch
from comfy_api.latest import io

from .segs_sampler_custom import (
    _blur_mask,
    _image_resize,
    _mask_tensor,
    _paste,
    _replace_crop,
)

SEGS = io.Custom("SEGS")


def seg_paste(
    cropped_image: torch.Tensor,
    segs: Any,
    index: int,
    feather: int,
    image: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, Any]:
    if not isinstance(cropped_image, torch.Tensor) or cropped_image.ndim != 4 or cropped_image.shape[0] != 1:
        shape = tuple(cropped_image.shape) if isinstance(cropped_image, torch.Tensor) else type(cropped_image).__name__
        raise ValueError(f"SEGPaste requires one cropped IMAGE in NHWC batch form, got {shape}.")
    if not isinstance(segs, (list, tuple)) or len(segs) != 2 or not isinstance(segs[1], (list, tuple)):
        raise TypeError("SEGPaste expects SEGS shaped as ((height, width), entries).")
    if not isinstance(segs[0], (list, tuple)) or len(segs[0]) != 2:
        raise TypeError("SEGPaste requires a two-value SEGS source-size header.")
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError(f"SEGPaste index must be an integer, got {index!r}.")
    entries = list(segs[1])
    if not 0 <= index < len(entries):
        raise IndexError(f"SEGPaste index {index} is outside the SEGS range 0..{len(entries) - 1}.")
    if isinstance(feather, bool) or not isinstance(feather, int) or not 0 <= feather <= 100:
        raise ValueError(f"SEGPaste feather must be an integer in 0..100, got {feather!r}.")

    source_height, source_width = int(segs[0][0]), int(segs[0][1])
    segment = entries[index]
    region = tuple(int(value) for value in segment.crop_region)
    if len(region) != 4:
        raise ValueError(f"SEGPaste SEG {index} crop_region must contain four coordinates.")
    x1, y1, x2, y2 = region
    if not (0 <= x1 < x2 <= source_width and 0 <= y1 < y2 <= source_height):
        raise ValueError(f"SEGPaste SEG {index} has invalid crop_region {region}.")

    crop_height = y2 - y1
    crop_width = x2 - x1
    resized = _image_resize(cropped_image, crop_width, crop_height)
    entries[index] = _replace_crop(segment, resized)

    output_image = None
    if image is not None:
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] != 1:
            shape = tuple(image.shape) if isinstance(image, torch.Tensor) else type(image).__name__
            raise ValueError(f"SEGPaste requires one destination IMAGE in NHWC batch form, got {shape}.")
        if tuple(image.shape[1:3]) != (source_height, source_width):
            raise ValueError(
                f"SEGPaste destination IMAGE size {tuple(image.shape[1:3])} must match "
                f"the SEGS source size {(source_height, source_width)}."
            )
        output_image = image.clone()
        mask = _mask_tensor(segment.cropped_mask, crop_height, crop_width)
        _paste(output_image, resized, region, _blur_mask(mask, feather))

    return output_image, (segs[0], entries)


class SEGPaste(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSEGPaste",
            display_name="SEGPaste",
            category="Utility Suite/SEGS",
            description="Resizes one processed SEG crop with Lanczos, updates SEGS, and optionally pastes it onto an image.",
            inputs=[
                io.Image.Input("cropped_image"),
                SEGS.Input("segs"),
                io.Int.Input("index", default=0, min=0, step=1),
                io.Int.Input("feather", default=5, min=0, max=100, step=1),
                io.Image.Input("image", optional=True),
            ],
            outputs=[io.Image.Output("image"), SEGS.Output("segs")],
        )

    @classmethod
    def execute(cls, cropped_image, segs, index, feather, image=None):
        return io.NodeOutput(*seg_paste(cropped_image, segs, index, feather, image))


__all__ = ["SEGPaste", "seg_paste"]
