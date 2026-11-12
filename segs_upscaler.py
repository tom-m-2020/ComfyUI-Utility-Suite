from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io
from PIL import Image

from .segs_sampler_custom import _blur_mask, _paste, processing_size

SEGS = io.Custom("SEGS")
RESAMPLING_METHODS = ["lanczos", "nearest", "bilinear", "bicubic"]
PIL_RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def iteration_bounds(count: int, start_index: int, max_iteration: int) -> tuple[int, int]:
    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
        raise ValueError(f"SEGS Upscaler start_index must be a nonnegative integer, got {start_index!r}.")
    if isinstance(max_iteration, bool) or not isinstance(max_iteration, int) or max_iteration < -1:
        raise ValueError(f"SEGS Upscaler max_iteration must be -1 or greater, got {max_iteration!r}.")
    start = min(start_index, count)
    stop = count if max_iteration in {-1, 0} else min(count, start + max_iteration)
    return start, stop


def _resize_image(image: torch.Tensor, width: int, height: int, method: str) -> torch.Tensor:
    if method not in PIL_RESAMPLING:
        raise ValueError(f"SEGS Upscaler does not support resampling_method {method!r}.")
    if width <= 0 or height <= 0:
        raise ValueError(f"SEGS Upscaler resize dimensions must be positive, got {(width, height)}.")
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"SEGS Upscaler expects one NHWC IMAGE, got {tuple(image.shape)}.")
    if image.shape[1:3] == (height, width):
        return image

    arrays = np.clip(image.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    resized = [
        np.asarray(Image.fromarray(array).resize((width, height), resample=PIL_RESAMPLING[method])).copy()
        for array in arrays
    ]
    result = torch.from_numpy(np.stack(resized).astype(np.float32) / 255.0)
    return result.to(dtype=image.dtype)


def _upscale_with_model(upscale_model: Any, image: torch.Tensor) -> torch.Tensor:
    from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

    return ImageUpscaleWithModel.execute(upscale_model, image).result[0]


def _mask_tensor(mask: Any, height: int, width: int) -> torch.Tensor:
    value = torch.as_tensor(mask, dtype=torch.float32)
    if value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(
            "SEGS Upscaler expects each cropped_mask as [H,W], [1,H,W], or [1,H,W,1], "
            f"got {tuple(value.shape)}."
        )
    if value.shape[-2:] != (height, width):
        value = F.interpolate(value.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False)[:, 0]
    return value.clamp(0.0, 1.0)


def _restore_mask_type(mask: torch.Tensor, original: Any) -> Any:
    value = mask.detach().cpu()
    original_value = torch.as_tensor(original)
    if original_value.ndim == 2:
        value = value[0]
    elif original_value.ndim == 4 and original_value.shape[-1] == 1:
        value = value.unsqueeze(-1)
    if isinstance(original, np.ndarray):
        return value.numpy()
    return value.to(dtype=original_value.dtype)


def _replace_segment(segment: Any, **changes: Any) -> Any:
    if not hasattr(segment, "_replace"):
        raise TypeError("SEGS Upscaler SEG entries must support namedtuple-style _replace().")
    return segment._replace(**changes)


def _crop(image: torch.Tensor, region: tuple[int, int, int, int]) -> torch.Tensor:
    x1, y1, x2, y2 = region
    return image[:, y1:y2, x1:x2, :]


def _validate_inputs(image: torch.Tensor, segs: Any) -> tuple[Any, list[Any]]:
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] != 1:
        shape = tuple(image.shape) if isinstance(image, torch.Tensor) else type(image).__name__
        raise ValueError(f"SEGS Upscaler requires one IMAGE in NHWC batch form, got {shape}.")
    if not isinstance(segs, (list, tuple)) or len(segs) != 2 or not isinstance(segs[1], (list, tuple)):
        raise TypeError("SEGS Upscaler expects SEGS shaped as ((height, width), entries).")
    if not isinstance(segs[0], (list, tuple)) or len(segs[0]) != 2:
        raise TypeError("SEGS Upscaler requires a two-value SEGS source-size header.")
    source_height, source_width = int(segs[0][0]), int(segs[0][1])
    if (source_height, source_width) != tuple(image.shape[1:3]):
        raise ValueError(
            f"SEGS source size {(source_height, source_width)} must match IMAGE size {tuple(image.shape[1:3])}."
        )
    return segs[0], list(segs[1])


def _scale_segments(
    header: Any,
    entries: list[Any],
    canvas: torch.Tensor,
) -> tuple[tuple[int, int], list[Any], list[tuple[int, int, int, int]], list[torch.Tensor]]:
    source_height, source_width = int(header[0]), int(header[1])
    target_height, target_width = canvas.shape[1:3]
    width_ratio = target_width / source_width
    height_ratio = target_height / source_height
    output_entries = []
    regions = []
    masks = []

    for index, segment in enumerate(entries):
        crop_region = tuple(int(value) for value in segment.crop_region)
        bbox = tuple(int(value) for value in segment.bbox)
        if len(crop_region) != 4 or len(bbox) != 4:
            raise ValueError(f"SEGS Upscaler entry {index} requires four-coordinate crop_region and bbox values.")
        x1, y1, x2, y2 = crop_region
        if not (0 <= x1 < x2 <= source_width and 0 <= y1 < y2 <= source_height):
            raise ValueError(f"SEGS Upscaler entry {index} has invalid crop_region {crop_region}.")

        region = (
            int(x1 * width_ratio),
            int(y1 * height_ratio),
            int(x2 * width_ratio),
            int(y2 * height_ratio),
        )
        if region[2] <= region[0] or region[3] <= region[1]:
            raise ValueError(
                f"SEGS Upscaler entry {index} crop_region collapses to {region} at the requested scale."
            )
        scaled_bbox = (
            int(bbox[0] * width_ratio),
            int(bbox[1] * height_ratio),
            int(bbox[2] * width_ratio),
            int(bbox[3] * height_ratio),
        )
        crop_height = region[3] - region[1]
        crop_width = region[2] - region[0]
        mask = _mask_tensor(segment.cropped_mask, crop_height, crop_width)
        crop = _crop(canvas, region).clone()
        output_entries.append(
            _replace_segment(
                segment,
                cropped_image=crop.detach().cpu().numpy(),
                cropped_mask=_restore_mask_type(mask, segment.cropped_mask),
                crop_region=region,
                bbox=scaled_bbox,
            )
        )
        regions.append(region)
        masks.append(mask)

    return (target_height, target_width), output_entries, regions, masks


def _guide_resize(
    crop: torch.Tensor,
    guide_size: int,
    resampling_method: str,
) -> torch.Tensor:
    crop_height, crop_width = crop.shape[1:3]
    width, height = processing_size(
        crop_width,
        crop_height,
        crop_width,
        crop_height,
        guide_size,
        "scale",
        guide_size,
        8,
    )
    return _resize_image(crop, width, height, resampling_method)


def segs_upscaler(
    image: torch.Tensor,
    segs: Any,
    upscale_model: Any,
    start_index: int,
    max_iteration: int,
    rescale_factor: float,
    resampling_method: str,
    whole_upscale_first: bool,
    overscale_smoothing: bool,
    use_guide_size: bool,
    guide_size: int,
    feather: int,
) -> tuple[torch.Tensor, Any]:
    header, entries = _validate_inputs(image, segs)
    if isinstance(rescale_factor, bool) or not isinstance(rescale_factor, (int, float)):
        raise TypeError(f"SEGS Upscaler rescale_factor must be a number, got {rescale_factor!r}.")
    if not math.isfinite(rescale_factor) or rescale_factor <= 0:
        raise ValueError(f"SEGS Upscaler rescale_factor must be positive and finite, got {rescale_factor!r}.")
    if resampling_method not in RESAMPLING_METHODS:
        raise ValueError(f"SEGS Upscaler does not support resampling_method {resampling_method!r}.")
    for name, value in (
        ("whole_upscale_first", whole_upscale_first),
        ("overscale_smoothing", overscale_smoothing),
        ("use_guide_size", use_guide_size),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"SEGS Upscaler {name} must be a boolean, got {value!r}.")
    if isinstance(guide_size, bool) or not isinstance(guide_size, int) or guide_size < 8:
        raise ValueError(f"SEGS Upscaler guide_size must be an integer of at least 8, got {guide_size!r}.")
    if isinstance(feather, bool) or not isinstance(feather, int) or not 0 <= feather <= 100:
        raise ValueError(f"SEGS Upscaler feather must be an integer in 0..100, got {feather!r}.")

    start, stop = iteration_bounds(len(entries), start_index, max_iteration)
    original_height, original_width = image.shape[1:3]
    target_width = int(original_width * rescale_factor)
    target_height = int(original_height * rescale_factor)
    if target_width <= 0 or target_height <= 0:
        raise ValueError(
            f"SEGS Upscaler rescale_factor produces invalid canvas size {(target_width, target_height)}."
        )

    whole_source = _upscale_with_model(upscale_model, image) if whole_upscale_first else image
    if overscale_smoothing:
        whole_source = _resize_image(
            whole_source,
            target_width * 5,
            target_height * 5,
            resampling_method,
        )
    base_canvas = _resize_image(whole_source, target_width, target_height, resampling_method).cpu()
    output_header, output_entries, regions, masks = _scale_segments(header, entries, base_canvas)

    processed_indices = []
    for index in range(start, stop):
        mask = masks[index]
        if not torch.any(mask > 0):
            continue
        region = regions[index]
        crop = _crop(base_canvas, region).clone()
        model_input = _guide_resize(crop, guide_size, resampling_method) if use_guide_size else crop
        enhanced = _upscale_with_model(upscale_model, model_input)
        enhanced = _resize_image(
            enhanced,
            region[2] - region[0],
            region[3] - region[1],
            resampling_method,
        ).cpu()
        output_entries[index] = _replace_segment(
            output_entries[index],
            cropped_image=enhanced.detach().cpu().numpy(),
        )
        processed_indices.append(index)

    canvas = base_canvas.clone()
    for index in processed_indices:
        crop = torch.as_tensor(output_entries[index].cropped_image, dtype=canvas.dtype)
        _paste(canvas, crop, regions[index], _blur_mask(masks[index], feather))

    return canvas, (output_header, output_entries)


class SEGSUpscaler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSEGSUpscaler",
            display_name="SEGS Upscaler",
            category="Utility Suite/SEGS",
            description="Upscales a canvas, independently model-upscales selected SEGs, and composites them by mask.",
            inputs=[
                io.Image.Input("image"),
                SEGS.Input("segs"),
                io.UpscaleModel.Input("upscale_model"),
                io.Int.Input("start_index", default=0, min=0, step=1),
                io.Int.Input(
                    "max_iteration",
                    default=-1,
                    min=-1,
                    step=1,
                    tooltip="-1 or 0 processes all remaining SEGs; a positive value limits the number processed.",
                ),
                io.Float.Input("rescale_factor", default=2.0, min=0.01, max=100.0, step=0.01),
                io.Combo.Input("resampling_method", options=RESAMPLING_METHODS, default="lanczos"),
                io.Boolean.Input("whole_upscale_first", default=True),
                io.Boolean.Input(
                    "overscale_smoothing",
                    default=False,
                    tooltip="Resizes to 5x the requested canvas dimensions, then downsizes to the requested size.",
                ),
                io.Boolean.Input("use_guide_size", default=True),
                io.Int.Input(
                    "guide_size",
                    default=512,
                    min=8,
                    max=32768,
                    step=1,
                    tooltip="Square root of the target SEG input pixel count before model upscaling.",
                ),
                io.Int.Input("feather", default=5, min=0, max=100, step=1),
            ],
            outputs=[io.Image.Output("image"), SEGS.Output("segs")],
        )

    @classmethod
    def execute(cls, **kwargs: Any) -> io.NodeOutput:
        return io.NodeOutput(*segs_upscaler(**kwargs))


__all__ = ["SEGSUpscaler", "iteration_bounds", "segs_upscaler"]
