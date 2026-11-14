from __future__ import annotations

import math
from typing import Literal

import comfy.utils
import torch
from comfy_api.latest import io

from .blur_mask_outward import blur_mask_outward
from .fit_dimensions_within_bounds import fit_dimensions_within_bounds
from .get_coordinate_by_percentage import get_coordinate_by_percentage
from .image_upscale_downscale import UPSCALE_METHODS

ResizeType = Literal["fit in canvas", "width", "height"]
SizeMode = Literal["pixel", "percent"]
MaskCombination = Literal["pixelwise min", "multiply"]

RESIZE_TYPES = ["fit in canvas", "width", "height"]
SIZE_MODES = ["pixel", "percent"]
MASK_COMBINATIONS = ["pixelwise min", "multiply"]


def _image(image: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        shape = tuple(image.shape) if isinstance(image, torch.Tensor) else type(image).__name__
        raise TypeError(f"Composite for Outpainting {name} must be an NHWC IMAGE, got {shape}.")
    if image.shape[0] < 1 or image.shape[1] < 1 or image.shape[2] < 1 or image.shape[3] < 1:
        raise ValueError(
            f"Composite for Outpainting {name} must have nonempty dimensions, got {tuple(image.shape)}."
        )
    return image


def _mask(mask: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"Composite for Outpainting {name} must be a MASK tensor.")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim != 3:
        raise ValueError(
            f"Composite for Outpainting {name} must be [H,W] or [B,H,W], got {tuple(mask.shape)}."
        )
    if any(dimension < 1 for dimension in mask.shape):
        raise ValueError(
            f"Composite for Outpainting {name} must have nonempty dimensions, got {tuple(mask.shape)}."
        )
    return mask


def _repeat_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    return comfy.utils.repeat_to_batch_size(tensor, batch_size)


def _resize_image(image: torch.Tensor, width: int, height: int, method: str) -> torch.Tensor:
    return comfy.utils.common_upscale(
        image.movedim(-1, 1), width, height, method, "disabled"
    ).movedim(1, -1)


def _resize_mask(mask: torch.Tensor, width: int, height: int, method: str) -> torch.Tensor:
    resized = comfy.utils.common_upscale(
        mask.unsqueeze(1), width, height, method, "disabled"
    ).squeeze(1)
    return resized.clamp(0.0, 1.0)


def _target_dimensions(
    canvas_width: int,
    canvas_height: int,
    source_width: int,
    source_height: int,
    resize: bool,
    resize_type: ResizeType,
    pixel_or_percent: SizeMode,
    target_size: int,
) -> tuple[int, int]:
    if not resize:
        return source_width, source_height
    if resize_type not in RESIZE_TYPES:
        raise ValueError(f"Unsupported Composite for Outpainting resize_type {resize_type!r}.")
    if pixel_or_percent not in SIZE_MODES:
        raise ValueError(
            f"Unsupported Composite for Outpainting pixel_or_percent {pixel_or_percent!r}."
        )
    if isinstance(target_size, bool) or not isinstance(target_size, int) or target_size < 0:
        raise ValueError(
            f"Composite for Outpainting target_size must be a nonnegative integer, got {target_size!r}."
        )

    if resize_type == "fit in canvas":
        width, height = fit_dimensions_within_bounds(
            canvas_width,
            canvas_height,
            source_width,
            source_height,
        )
        return max(1, width), max(1, height)

    if pixel_or_percent == "percent":
        if target_size > 200:
            raise ValueError(
                "Composite for Outpainting target_size cannot exceed 200 in percent mode."
            )
        canvas_size = canvas_width if resize_type == "width" else canvas_height
        resolved_size = round(canvas_size * (target_size * 0.01))
    else:
        resolved_size = target_size

    if resolved_size == 0:
        return source_width, source_height
    if resize_type == "width":
        return resolved_size, max(1, round(source_height * resolved_size / source_width))
    return max(1, round(source_width * resolved_size / source_height)), resolved_size


def _paste_image(
    canvas: torch.Tensor,
    subject: torch.Tensor,
    x: int,
    y: int,
) -> torch.Tensor:
    output = canvas.clone()
    subject = _repeat_batch(subject, output.shape[0]).to(
        device=output.device, dtype=output.dtype
    )
    if subject.shape[-1] > output.shape[-1]:
        subject = subject[..., : output.shape[-1]]
    elif subject.shape[-1] < output.shape[-1]:
        missing = output.shape[-1] - subject.shape[-1]
        subject = torch.nn.functional.pad(subject, (0, missing))
        if output.shape[-1] == 4:
            subject[..., -1] = 1.0

    destination_x = max(0, x)
    destination_y = max(0, y)
    source_x = max(0, -x)
    source_y = max(0, -y)
    width = min(output.shape[2] - destination_x, subject.shape[2] - source_x)
    height = min(output.shape[1] - destination_y, subject.shape[1] - source_y)
    if width > 0 and height > 0:
        output[
            :,
            destination_y : destination_y + height,
            destination_x : destination_x + width,
            :,
        ] = subject[:, source_y : source_y + height, source_x : source_x + width, :]
    return output


def _subtract_subject_mask(
    canvas_mask: torch.Tensor,
    subject_mask: torch.Tensor,
    x: int,
    y: int,
) -> torch.Tensor:
    output = canvas_mask.clone()
    subject_mask = _repeat_batch(subject_mask, output.shape[0]).to(
        device=output.device, dtype=output.dtype
    )
    destination_x = max(0, x)
    destination_y = max(0, y)
    source_x = max(0, -x)
    source_y = max(0, -y)
    width = min(output.shape[2] - destination_x, subject_mask.shape[2] - source_x)
    height = min(output.shape[1] - destination_y, subject_mask.shape[1] - source_y)
    if width > 0 and height > 0:
        destination = output[
            :,
            destination_y : destination_y + height,
            destination_x : destination_x + width,
        ]
        source = subject_mask[:, source_y : source_y + height, source_x : source_x + width]
        output[
            :,
            destination_y : destination_y + height,
            destination_x : destination_x + width,
        ] = destination - source
    return output.clamp(0.0, 1.0)


def composite_for_outpainting(
    canvas_image: torch.Tensor,
    subject_image: torch.Tensor,
    x_percent: int,
    y_percent: int,
    resize: bool,
    resize_type: ResizeType,
    pixel_or_percent: SizeMode,
    target_size: int,
    scale_method: str,
    blur_radius: float,
    tapered_corners: bool,
    combine_mask: MaskCombination,
    canvas_mask: torch.Tensor | None = None,
    subject_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    canvas_image = _image(canvas_image, "canvas_image")
    subject_image = _image(subject_image, "subject_image")
    if scale_method not in UPSCALE_METHODS:
        raise ValueError(f"Unsupported Composite for Outpainting scale_method {scale_method!r}.")
    if combine_mask not in MASK_COMBINATIONS:
        raise ValueError(f"Unsupported Composite for Outpainting combine_mask {combine_mask!r}.")
    if not isinstance(resize, bool):
        raise TypeError(f"Composite for Outpainting resize must be BOOLEAN, got {resize!r}.")
    if not isinstance(tapered_corners, bool):
        raise TypeError(
            f"Composite for Outpainting tapered_corners must be BOOLEAN, got {tapered_corners!r}."
        )
    for name, value in (("x_percent", x_percent), ("y_percent", y_percent)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
            raise ValueError(
                f"Composite for Outpainting {name} must be an integer within 0..100, got {value!r}."
            )
    if isinstance(blur_radius, bool) or not isinstance(blur_radius, (int, float)):
        raise TypeError(
            f"Composite for Outpainting blur_radius must be a number, got {blur_radius!r}."
        )
    if not math.isfinite(float(blur_radius)) or not 0 <= blur_radius <= 100:
        raise ValueError(
            f"Composite for Outpainting blur_radius must be finite and within 0..100, got {blur_radius!r}."
        )

    canvas_height, canvas_width = canvas_image.shape[1:3]
    source_height, source_width = subject_image.shape[1:3]
    target_width, target_height = _target_dimensions(
        canvas_width,
        canvas_height,
        source_width,
        source_height,
        resize,
        resize_type,
        pixel_or_percent,
        target_size,
    )
    if (target_width, target_height) != (source_width, source_height):
        processed_subject = _resize_image(
            subject_image, target_width, target_height, scale_method
        )
    else:
        processed_subject = subject_image

    if subject_mask is None:
        processed_subject_mask = torch.ones(
            (1, target_height, target_width),
            dtype=torch.float32,
            device=processed_subject.device,
        )
    else:
        processed_subject_mask = _mask(subject_mask, "subject_mask")
        if tuple(processed_subject_mask.shape[-2:]) != (target_height, target_width):
            processed_subject_mask = _resize_mask(
                processed_subject_mask,
                target_width,
                target_height,
                scale_method,
            )

    x, y = get_coordinate_by_percentage(
        canvas_width,
        canvas_height,
        target_width,
        target_height,
        x_percent,
        y_percent,
    )
    batch_size = canvas_image.shape[0]
    local_canvas_mask = torch.ones(
        (batch_size, canvas_height, canvas_width),
        dtype=torch.float32,
        device=processed_subject_mask.device,
    )
    local_canvas_mask = _subtract_subject_mask(
        local_canvas_mask,
        processed_subject_mask,
        x,
        y,
    )
    local_canvas_mask = blur_mask_outward(
        local_canvas_mask,
        float(blur_radius),
        tapered_corners,
        True,
    )

    if canvas_mask is None:
        output_mask = local_canvas_mask
    else:
        previous_mask = _mask(canvas_mask, "canvas_mask")
        if tuple(previous_mask.shape[-2:]) != (canvas_height, canvas_width):
            previous_mask = _resize_mask(
                previous_mask,
                canvas_width,
                canvas_height,
                scale_method,
            )
        previous_mask = _repeat_batch(previous_mask, batch_size).to(
            device=local_canvas_mask.device,
            dtype=local_canvas_mask.dtype,
        )
        if combine_mask == "pixelwise min":
            output_mask = torch.minimum(previous_mask, local_canvas_mask)
        else:
            output_mask = previous_mask * local_canvas_mask

    composited = _paste_image(canvas_image, processed_subject, x, y)
    subject_alpha = (1.0 - local_canvas_mask).to(
        device=canvas_image.device,
        dtype=canvas_image.dtype,
    ).unsqueeze(-1)
    masked = canvas_image + subject_alpha * (composited - canvas_image)
    return composited, masked, output_mask


class CompositeForOutpainting(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteCompositeForOutpainting",
            display_name="Composite for Outpainting",
            category="Utility Suite/Image",
            description=(
                "Places a subject on a canvas and creates a reusable outpainting mask with an optional outward blur."
            ),
            inputs=[
                io.Image.Input("canvas_image"),
                io.Image.Input("subject_image"),
                io.Mask.Input("canvas_mask", optional=True),
                io.Mask.Input("subject_mask", optional=True),
                io.Int.Input("x_percent", default=50, min=0, max=100, step=1),
                io.Int.Input("y_percent", default=50, min=0, max=100, step=1),
                io.Boolean.Input("resize", default=True),
                io.Combo.Input("resize_type", options=RESIZE_TYPES, default="fit in canvas"),
                io.Combo.Input("pixel_or_percent", options=SIZE_MODES, default="pixel"),
                io.Int.Input("target_size", default=0, min=0, max=32768, step=1),
                io.Combo.Input("scale_method", options=UPSCALE_METHODS, default="lanczos"),
                io.Float.Input("blur_radius", default=0.0, min=0.0, max=100.0, step=0.1),
                io.Boolean.Input("tapered_corners", default=False),
                io.Combo.Input(
                    "combine_mask",
                    options=MASK_COMBINATIONS,
                    default="pixelwise min",
                ),
            ],
            outputs=[
                io.Image.Output("canvas_image"),
                io.Image.Output("canvas_image_masked"),
                io.Mask.Output("canvas_mask"),
            ],
        )

    @classmethod
    def execute(
        cls,
        canvas_image: torch.Tensor,
        subject_image: torch.Tensor,
        x_percent: int,
        y_percent: int,
        resize: bool,
        resize_type: ResizeType,
        pixel_or_percent: SizeMode,
        target_size: int,
        scale_method: str,
        blur_radius: float,
        tapered_corners: bool,
        combine_mask: MaskCombination,
        canvas_mask: torch.Tensor | None = None,
        subject_mask: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            *composite_for_outpainting(
                canvas_image,
                subject_image,
                x_percent,
                y_percent,
                resize,
                resize_type,
                pixel_or_percent,
                target_size,
                scale_method,
                blur_radius,
                tapered_corners,
                combine_mask,
                canvas_mask,
                subject_mask,
            )
        )


__all__ = ["CompositeForOutpainting", "composite_for_outpainting"]
