from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from comfy_api.latest import io

from .segs_sampler_custom import _image_resize, processing_size

SEG_ELT = io.Custom("SEG_ELT")


def _image_tensor(value: Any) -> torch.Tensor:
    image = torch.as_tensor(value)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Guide Size (SEG_ELT) expects one NHWC IMAGE, got {tuple(image.shape)}.")
    return image


def _mask_tensor(value: Any) -> torch.Tensor:
    mask = torch.as_tensor(value)
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(f"Guide Size (SEG_ELT) expects one MASK as [H,W], [1,H,W], or [1,H,W,1], got {tuple(mask.shape)}.")
    return mask


def guide_size_seg_elt(
    seg_elt: Any,
    guide_size: int,
    guide_size_for: str,
    scale_to_guide_size: str,
    max_size: int,
    image: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image is None:
        image = getattr(seg_elt, "cropped_image", None)
        if image is None:
            raise ValueError("Guide Size (SEG_ELT) requires image or seg_elt.cropped_image.")
    if mask is None:
        mask = getattr(seg_elt, "cropped_mask", None)
        if mask is None:
            raise ValueError("Guide Size (SEG_ELT) requires mask or seg_elt.cropped_mask.")

    image = _image_tensor(image)
    mask = _mask_tensor(mask)
    image_height, image_width = image.shape[1:3]

    if guide_size_for == "crop_region":
        region = getattr(seg_elt, "crop_region", None)
    elif guide_size_for == "bbox":
        region = getattr(seg_elt, "bbox", None)
    else:
        raise ValueError(f"Unsupported guide_size_for {guide_size_for!r}.")
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        raise ValueError(f"Guide Size (SEG_ELT) requires a four-coordinate {guide_size_for}.")
    reference_width = int(region[2]) - int(region[0])
    reference_height = int(region[3]) - int(region[1])

    width, height = processing_size(
        image_width,
        image_height,
        reference_width,
        reference_height,
        guide_size,
        scale_to_guide_size,
        max_size,
        8,
    )
    reference_area = reference_width * reference_height
    guide_area = guide_size * guide_size
    if (scale_to_guide_size == "upscale" and guide_area <= reference_area) or (
        scale_to_guide_size == "downscale" and guide_area >= reference_area
    ):
        width, height = image_width, image_height

    resized_image = _image_resize(image, width, height)
    if mask.shape[-2:] == (height, width):
        resized_mask = mask
    else:
        resized_mask = F.interpolate(
            mask.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    return resized_image, resized_mask


class GuideSizeSEGELT(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteGuideSizeSEGELT",
            display_name="Guide Size (SEG_ELT)",
            category="Utility Suite/SEGS",
            description="Resizes a SEG element's image and mask to its guide-size sampling resolution.",
            inputs=[
                SEG_ELT.Input("seg_elt"),
                io.Image.Input("image", optional=True),
                io.Mask.Input("mask", optional=True),
                io.Int.Input(
                    "guide_size",
                    default=512,
                    min=1,
                    max=32768,
                    step=1,
                    tooltip="Square root of the target reference-region pixel count; this is not a width or height.",
                ),
                io.Combo.Input(
                    "guide_size_for",
                    options=["crop_region", "bbox"],
                    default="bbox",
                    tooltip="Selects the SEG region whose square-root pixel area establishes the scaling factor.",
                ),
                io.Combo.Input(
                    "scale_to_guide_size",
                    options=["scale", "upscale", "downscale", "none"],
                    default="upscale",
                    tooltip="Selects directional scaling. 'none' disables resizing and max_size enforcement.",
                ),
                io.Int.Input(
                    "max_size",
                    default=1024,
                    min=8,
                    max=32768,
                    step=1,
                    tooltip="Caps the square root of the output pixel count whenever scaling is performed.",
                ),
            ],
            outputs=[io.Image.Output("image"), io.Mask.Output("mask")],
        )

    @classmethod
    def execute(cls, **kwargs: Any) -> io.NodeOutput:
        return io.NodeOutput(*guide_size_seg_elt(**kwargs))


__all__ = ["GuideSizeSEGELT", "guide_size_seg_elt"]
