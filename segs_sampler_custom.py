from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from comfy_api.latest import io

from .segs_sampler_unsample import (
    SEGS_UNSAMPLE,
    SEGSUnsampleOptions,
    sample_with_optional_unsample,
)

SEGS = io.Custom("SEGS")


def processing_size(
    crop_width: int,
    crop_height: int,
    reference_width: int,
    reference_height: int,
    guide_size: int,
    scale_mode: str,
    max_size: int,
    multiple: int,
) -> tuple[int, int]:
    """Return a deterministic VAE-compatible processing size."""
    values = (crop_width, crop_height, reference_width, reference_height, guide_size, max_size, multiple)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("SEGS Sampler Custom resolution values must be positive integers.")
    if scale_mode == "none":
        return crop_width, crop_height
    if scale_mode not in {"scale", "upscale", "downscale"}:
        raise ValueError(f"Unsupported scale_to_guide_size {scale_mode!r}.")
    if max_size < multiple:
        raise ValueError(
            f"max_size {max_size} cannot contain one VAE-compatible spatial unit of {multiple} pixels."
        )

    requested = guide_size / math.sqrt(reference_width * reference_height)
    if scale_mode == "upscale":
        scale = max(1.0, requested)
    elif scale_mode == "downscale":
        scale = min(1.0, requested)
    else:
        scale = requested

    scale = min(scale, max_size / math.sqrt(crop_width * crop_height))
    target_width = crop_width * scale
    target_height = crop_height * scale

    def rounded(value: float) -> int:
        return max(multiple, math.floor(value / multiple + 0.5) * multiple)

    width = rounded(target_width)
    height = rounded(target_height)
    max_area = max_size * max_size
    while width * height > max_area:
        candidates = []
        if width > multiple:
            candidates.append((width - multiple, height))
        if height > multiple:
            candidates.append((width, height - multiple))
        if not candidates:
            raise ValueError(
                f"max_size {max_size} cannot fit the VAE spatial compression multiple {multiple}."
            )
        target_ratio = crop_width / crop_height
        width, height = min(
            candidates,
            key=lambda size: (abs(math.log((size[0] / size[1]) / target_ratio)), -size[0] * size[1]),
        )
    return width, height


def should_sample(reference_width: int, reference_height: int, guide_size: int, force_inpaint: str) -> bool:
    if force_inpaint not in {"disabled", "enabled"}:
        raise ValueError(f"Unsupported force_inpaint {force_inpaint!r}.")
    return force_inpaint == "enabled" or math.sqrt(reference_width * reference_height) < guide_size


def _image_resize(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if image.shape[1:3] == (height, width):
        return image
    from comfy.utils import common_upscale

    channels_first = image.movedim(-1, 1)
    return common_upscale(channels_first, width, height, "lanczos", "disabled").movedim(1, -1)


def _mask_tensor(mask: Any, height: int, width: int) -> torch.Tensor:
    value = torch.as_tensor(mask, dtype=torch.float32)
    if value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim == 2:
        value = value.unsqueeze(0)
    elif value.ndim == 3 and value.shape[0] > 1:
        value = torch.any(value > 0.1, dim=0, keepdim=True).float()
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(f"SEGS Sampler Custom expects each cropped_mask as [H,W] or [1,H,W], got {tuple(value.shape)}.")
    if value.shape[-2:] != (height, width):
        value = F.interpolate(value.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False)[:, 0]
    return value.clamp(0.0, 1.0)


def _blur_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel_size = radius * 2 + 1
    shortest = min(mask.shape[-2:])
    if shortest <= kernel_size:
        kernel_size = int(shortest / 2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            return mask
    half = kernel_size // 2
    coordinates = torch.arange(kernel_size, device=mask.device, dtype=mask.dtype) - half
    kernel = torch.exp(-(coordinates**2) / (2 * 10.0**2))
    kernel /= kernel.sum()
    value = mask.unsqueeze(1)
    padding_mode = "reflect" if min(value.shape[-2:]) > half else "replicate"
    value = F.pad(value, (half, half, 0, 0), mode=padding_mode)
    value = F.conv2d(value, kernel.view(1, 1, 1, -1))
    value = F.pad(value, (0, 0, half, half), mode=padding_mode)
    return F.conv2d(value, kernel.view(1, 1, -1, 1))[:, 0]


def _encode(vae: Any, image: torch.Tensor, tiled: bool) -> dict[str, torch.Tensor]:
    if tiled:
        samples = vae.encode_tiled(image, tile_x=512, tile_y=512, overlap=64, tile_t=64, overlap_t=8)
    else:
        samples = vae.encode(image)
    return {"samples": samples}


def _decode(vae: Any, latent: dict[str, torch.Tensor], tiled: bool) -> torch.Tensor:
    samples = latent["samples"]
    if tiled:
        compression = vae.spacial_compression_decode()
        image = vae.decode_tiled(
            samples,
            tile_x=512 // compression,
            tile_y=512 // compression,
            overlap=64 // compression,
            tile_t=None,
            overlap_t=None,
        )
    else:
        image = vae.decode(samples)
    if image.ndim == 5:
        image = image.reshape(-1, image.shape[-3], image.shape[-2], image.shape[-1])
    if image.ndim != 4:
        raise ValueError(f"SEGS Sampler Custom VAE decode returned unsupported shape {tuple(image.shape)}.")
    return image


def _sample_once(noise: Any, guider: Any, sampler: Any, sigmas: torch.Tensor, latent: dict[str, Any]) -> dict[str, Any]:
    from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

    return SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent).result[0]


def _replace_crop(segment: Any, cropped_image: torch.Tensor) -> Any:
    image = cropped_image.detach().cpu().numpy()
    if hasattr(segment, "_replace"):
        return segment._replace(cropped_image=image)
    raise TypeError("SEGS Sampler Custom SEG entries must support namedtuple-style _replace().")


def _crop(image: torch.Tensor, region: tuple[int, int, int, int]) -> torch.Tensor:
    x1, y1, x2, y2 = region
    return image[:, y1:y2, x1:x2, :]


def _paste(canvas: torch.Tensor, crop: torch.Tensor, region: tuple[int, int, int, int], mask: torch.Tensor) -> None:
    x1, y1, x2, y2 = region
    destination = canvas[:, y1:y2, x1:x2, :]
    source = crop.to(device=destination.device, dtype=destination.dtype)
    alpha = mask.to(device=destination.device, dtype=destination.dtype).unsqueeze(-1)
    destination.copy_(destination * (1.0 - alpha) + source * alpha)


def _validate(image: torch.Tensor, segs: Any) -> tuple[Any, list[Any], list[tuple[int, int, int, int]]]:
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("SEGS Sampler Custom requires one IMAGE in NHWC batch form.")
    if not isinstance(segs, (list, tuple)) or len(segs) != 2 or not isinstance(segs[1], (list, tuple)):
        raise TypeError("SEGS Sampler Custom expects SEGS shaped as ((height, width), entries).")
    source_height, source_width = (int(segs[0][0]), int(segs[0][1]))
    if (source_height, source_width) != tuple(image.shape[1:3]):
        raise ValueError(
            f"SEGS source size {(source_height, source_width)} must match IMAGE size {tuple(image.shape[1:3])}."
        )
    regions = []
    for index, segment in enumerate(segs[1]):
        region = tuple(int(value) for value in segment.crop_region)
        if len(region) != 4:
            raise ValueError(f"SEGS entry {index} crop_region must contain four coordinates.")
        x1, y1, x2, y2 = region
        if not (0 <= x1 < x2 <= source_width and 0 <= y1 < y2 <= source_height):
            raise ValueError(f"SEGS entry {index} has invalid crop_region {region}.")
        bbox = tuple(int(value) for value in segment.bbox)
        empty_bbox = bbox == (0, 0, 0, 0)
        empty_mask = empty_bbox and not torch.any(torch.as_tensor(segment.cropped_mask) != 0)
        if len(bbox) != 4 or not (empty_mask or (bbox[2] > bbox[0] and bbox[3] > bbox[1])):
            raise ValueError(f"SEGS entry {index} has invalid bbox {bbox}.")
        regions.append(region)
    return segs[0], list(segs[1]), regions


def _enhance(
    crop: torch.Tensor,
    segment: Any,
    noise: Any,
    guider: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    vae: Any,
    guide_size: int,
    guide_size_for: str,
    scale_mode: str,
    max_size: int,
    force_inpaint: str,
    noise_mask: str,
    noise_mask_feather: int,
    tiled_encode: str,
    tiled_decode: str,
    unsample: SEGSUnsampleOptions | None,
) -> torch.Tensor | None:
    crop_height, crop_width = crop.shape[1:3]
    if guide_size_for == "crop_region":
        reference_width, reference_height = crop_width, crop_height
    elif guide_size_for == "bbox":
        reference_width = int(segment.bbox[2]) - int(segment.bbox[0])
        reference_height = int(segment.bbox[3]) - int(segment.bbox[1])
    else:
        raise ValueError(f"Unsupported guide_size_for {guide_size_for!r}.")
    if not should_sample(reference_width, reference_height, guide_size, force_inpaint):
        return None

    compression = vae.spacial_compression_encode()
    if isinstance(compression, bool) or not isinstance(compression, int) or compression <= 0:
        raise ValueError(f"SEGS Sampler Custom requires a positive integer VAE compression ratio, got {compression!r}.")
    width, height = processing_size(
        crop_width,
        crop_height,
        reference_width,
        reference_height,
        guide_size,
        scale_mode,
        max_size,
        compression,
    )
    processed = _image_resize(crop, width, height)
    latent = _encode(vae, processed, tiled_encode == "enabled")
    if noise_mask == "enabled":
        mask = _mask_tensor(segment.cropped_mask, height, width)
        latent["noise_mask"] = _blur_mask(mask, noise_mask_feather)
    elif noise_mask != "disabled":
        raise ValueError(f"Unsupported noise_mask {noise_mask!r}.")
    latent = sample_with_optional_unsample(noise, guider, sampler, sigmas, latent, unsample, _sample_once)
    decoded = _decode(vae, latent, tiled_decode == "enabled")
    return _image_resize(decoded, crop_width, crop_height).cpu()


def segs_sampler_custom(
    image: torch.Tensor,
    segs: Any,
    noise: Any,
    guider: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    vae: Any,
    guide_size: int,
    guide_size_for: str,
    scale_to_guide_size: str,
    max_size: int,
    force_inpaint: str,
    feather: int,
    noise_mask: str,
    noise_mask_feather: int,
    tiled_encode: str,
    tiled_decode: str,
    operation: str,
    unsample: SEGSUnsampleOptions | None = None,
) -> tuple[torch.Tensor, Any]:
    for name, value, minimum, maximum in (
        ("guide_size", guide_size, 1, None),
        ("max_size", max_size, 1, None),
        ("feather", feather, 0, 100),
        ("noise_mask_feather", noise_mask_feather, 0, 100),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (
            maximum is not None and value > maximum
        ):
            interval = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
            raise ValueError(f"SEGS Sampler Custom {name} must be an integer in {interval}, got {value!r}.")
    header, entries, regions = _validate(image, segs)
    if operation not in {"sequential", "individual"}:
        raise ValueError(f"Unsupported operation {operation!r}.")
    canvas = image.clone()
    output_entries = []

    for segment, region in zip(entries, regions):
        original_height = region[3] - region[1]
        original_width = region[2] - region[0]
        original_mask = _mask_tensor(segment.cropped_mask, original_height, original_width)
        empty = not torch.any(original_mask > 0)
        if operation == "sequential":
            crop = _crop(canvas, region).clone()
        elif segment.cropped_image is not None:
            crop = torch.as_tensor(segment.cropped_image, dtype=image.dtype)
            if crop.ndim == 3:
                crop = crop.unsqueeze(0)
            if crop.ndim != 4 or crop.shape[0] != 1 or crop.shape[-1] != image.shape[-1]:
                raise ValueError(
                    "SEG cropped_image must contain one NHWC image with the same channel count as IMAGE, "
                    f"got {tuple(crop.shape)}."
                )
            if crop.shape[1:3] != (original_height, original_width):
                raise ValueError(
                    f"SEG cropped_image shape {tuple(crop.shape)} does not match crop_region size "
                    f"{(original_height, original_width)}."
                )
        else:
            crop = _crop(image, region).clone()

        enhanced = None if empty else _enhance(
            crop,
            segment,
            noise,
            guider,
            sampler,
            sigmas,
            vae,
            guide_size,
            guide_size_for,
            scale_to_guide_size,
            max_size,
            force_inpaint,
            noise_mask,
            noise_mask_feather,
            tiled_encode,
            tiled_decode,
            unsample,
        )
        result_crop = crop if enhanced is None else enhanced
        output_entries.append(_replace_crop(segment, result_crop))
        if operation == "sequential" and enhanced is not None:
            _paste(canvas, enhanced, region, _blur_mask(original_mask, feather))

    if operation == "individual":
        canvas = image.clone()
        for segment, region in zip(output_entries, regions):
            crop = torch.as_tensor(segment.cropped_image, dtype=image.dtype)
            _paste(canvas, crop, region, _blur_mask(_mask_tensor(segment.cropped_mask, region[3] - region[1], region[2] - region[0]), feather))
    else:
        output_entries = [_replace_crop(segment, _crop(canvas, region).clone()) for segment, region in zip(entries, regions)]

    return canvas, (header, output_entries)


class SEGSSamplerCustom(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        guide_tooltip = (
            "Square root of the target reference-region pixel count; this is not a width or height."
        )
        return io.Schema(
            node_id="UtilitySuiteSEGSSamplerCustom",
            display_name="SEGS Sampler Custom",
            category="Utility Suite/SEGS",
            description="Samples SEGS with ComfyUI NOISE, GUIDER, SAMPLER, and SIGMAS components.",
            inputs=[
                io.Image.Input("image"),
                SEGS.Input("segs"),
                io.Noise.Input("noise"),
                io.Guider.Input("guider"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Vae.Input("vae"),
                io.Int.Input("guide_size", default=512, min=1, max=32768, step=1, tooltip=guide_tooltip),
                io.Combo.Input(
                    "guide_size_for",
                    options=["crop_region", "bbox"],
                    default="bbox",
                    tooltip="Selects the region whose square-root pixel area establishes the scaling factor.",
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
                    min=1,
                    max=32768,
                    step=1,
                    tooltip="Caps the square root of the whole processing crop pixel count whenever scaling is performed.",
                ),
                io.Combo.Input("force_inpaint", options=["disabled", "enabled"], default="enabled"),
                io.Int.Input("feather", default=5, min=0, max=100, step=1),
                io.Combo.Input("noise_mask", options=["disabled", "enabled"], default="enabled"),
                io.Int.Input("noise_mask_feather", default=20, min=0, max=100, step=1),
                io.Combo.Input("tiled_encode", options=["disabled", "enabled"], default="disabled"),
                io.Combo.Input("tiled_decode", options=["disabled", "enabled"], default="disabled"),
                io.Combo.Input("operation", options=["sequential", "individual"], default="sequential"),
                SEGS_UNSAMPLE.Input("unsample", optional=True),
            ],
            outputs=[io.Image.Output("image"), SEGS.Output("segs")],
        )

    @classmethod
    def execute(cls, **kwargs: Any) -> io.NodeOutput:
        return io.NodeOutput(*segs_sampler_custom(**kwargs))


__all__ = ["SEGSSamplerCustom", "processing_size", "segs_sampler_custom", "should_sample"]
