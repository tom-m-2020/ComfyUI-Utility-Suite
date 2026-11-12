from __future__ import annotations

import math

import comfy.utils
import torch
from comfy_api.latest import io

UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]


def upscale_and_downscale(
    image: torch.Tensor,
    upscale_method: str,
    scale_by: float,
    cycle: int,
) -> torch.Tensor:
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] < 1:
        shape = tuple(image.shape) if isinstance(image, torch.Tensor) else type(image).__name__
        raise ValueError(f"Upscale and Downscale expects a nonempty NHWC IMAGE batch, got {shape}.")
    if upscale_method not in UPSCALE_METHODS:
        raise ValueError(f"Upscale and Downscale does not support upscale_method {upscale_method!r}.")
    if isinstance(scale_by, bool) or not isinstance(scale_by, (int, float)):
        raise TypeError(f"Upscale and Downscale scale_by must be a number, got {scale_by!r}.")
    if not math.isfinite(scale_by) or not 0.01 <= scale_by <= 8.0:
        raise ValueError(f"Upscale and Downscale scale_by must be within 0.01..8.0, got {scale_by!r}.")
    if isinstance(cycle, bool) or not isinstance(cycle, int) or not 1 <= cycle <= 5:
        raise ValueError(f"Upscale and Downscale cycle must be an integer within 1..5, got {cycle!r}.")

    original_height, original_width = image.shape[1:3]
    scaled_width = max(1, round(original_width * scale_by))
    scaled_height = max(1, round(original_height * scale_by))
    result = image.movedim(-1, 1)

    for _ in range(cycle):
        result = comfy.utils.common_upscale(
            result,
            scaled_width,
            scaled_height,
            upscale_method,
            "disabled",
        )
        result = comfy.utils.common_upscale(
            result,
            original_width,
            original_height,
            upscale_method,
            "disabled",
        )

    return result.movedim(1, -1)


class UpscaleAndDownscale(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteUpscaleAndDownscale",
            display_name="Upscale and Downscale",
            category="Utility Suite/Image",
            description="Repeatedly resizes an image by a scale factor and back to its original dimensions.",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("upscale_method", options=UPSCALE_METHODS, default="bilinear"),
                io.Float.Input("scale_by", default=8.0, min=0.01, max=8.0, step=0.01),
                io.Int.Input("cycle", default=1, min=1, max=5, step=1),
            ],
            outputs=[io.Image.Output("image")],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        upscale_method: str,
        scale_by: float,
        cycle: int,
    ) -> io.NodeOutput:
        return io.NodeOutput(upscale_and_downscale(image, upscale_method, scale_by, cycle))


__all__ = ["UPSCALE_METHODS", "UpscaleAndDownscale", "upscale_and_downscale"]
