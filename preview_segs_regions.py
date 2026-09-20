from __future__ import annotations

from typing import Any

import numpy as np
import torch
from comfy_api.latest import io
from PIL import Image, ImageDraw, ImageFont

from .segs_geometry import coverage_map, integer, validated_segs

SEGS = io.Custom("SEGS")
HATCH_PERIOD = 16
HATCH_WIDTH = 2
YELLOW_FILL_ALPHA = 0.22
RED_FILL_ALPHA = 0.26
INDEX_BLUE = (0.08, 0.32, 1.0)
INDEX_ALPHA = 0.74
INDEX_OUTLINE_ALPHA = 0.58
INDEX_FONT_SCALE = 0.18
INDEX_FONT_MIN = 12
INDEX_FONT_MAX = 72


def _array(value: Any, description: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.to(torch.float32)
        return value.numpy()
    if isinstance(value, np.ndarray):
        return value
    raise TypeError(f"{description} must be a NumPy array or Torch tensor, got {type(value).__name__}.")


def _image_crop(value: Any, index: int, height: int, width: int) -> np.ndarray:
    image = _array(value, f"SEGS entry {index} cropped_image")
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"SEGS entry {index} cropped_image must be [1,H,W,C], got {image.shape}.")
    if image.shape[1:3] != (height, width):
        raise ValueError(
            f"SEGS entry {index} cropped_image shape {image.shape[1:3]} does not match "
            f"crop_region extent {(height, width)}."
        )
    channels = image.shape[3]
    if channels == 1:
        return np.repeat(image[0].astype(np.float32, copy=False), 3, axis=2)
    if channels in (3, 4):
        return image[0, :, :, :3].astype(np.float32, copy=False)
    raise ValueError(f"SEGS entry {index} cropped_image must have 1, 3, or 4 channels, got {channels}.")


def _mask_crop(value: Any, index: int, height: int, width: int) -> np.ndarray:
    mask = _array(value, f"SEGS entry {index} cropped_mask")
    if mask.ndim == 3:
        if mask.shape[0] != 1:
            raise ValueError(f"SEGS entry {index} cropped_mask must be [H,W] or [1,H,W], got {mask.shape}.")
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"SEGS entry {index} cropped_mask must be [H,W] or [1,H,W], got {mask.shape}.")
    if mask.shape != (height, width):
        raise ValueError(
            f"SEGS entry {index} cropped_mask shape {mask.shape} does not match "
            f"crop_region extent {(height, width)}."
        )
    return mask.astype(np.float32, copy=False)


def _fallback_image(value: Any, height: int, width: int) -> np.ndarray:
    image = _array(value, "fallback_image_opt")
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Preview SEGS Regions requires fallback IMAGE [1,H,W,C], got {image.shape}.")
    if image.shape[1:3] != (height, width):
        raise ValueError(
            f"Preview SEGS Regions fallback IMAGE dimensions {image.shape[1:3]} do not match SEGS {(height, width)}."
        )
    channels = image.shape[3]
    if channels == 1:
        return np.repeat(image[0].astype(np.float32, copy=False), 3, axis=2).copy()
    if channels in (3, 4):
        return image[0, :, :, :3].astype(np.float32, copy=True)
    raise ValueError(f"Preview SEGS Regions fallback IMAGE must have 1, 3, or 4 channels, got {channels}.")


def _draw_boundaries(
    canvas: np.ndarray, crop_regions: list[tuple[int, int, int, int]], line_width: int
) -> None:
    blue = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    for x1, y1, x2, y2 in crop_regions:
        width = min(line_width, x2 - x1)
        height = min(line_width, y2 - y1)
        canvas[y1 : y1 + height, x1:x2] = blue
        canvas[y2 - height : y2, x1:x2] = blue
        canvas[y1:y2, x1 : x1 + width] = blue
        canvas[y1:y2, x2 - width : x2] = blue


def _index_label_specs(
    crop_regions: list[tuple[int, int, int, int]],
) -> list[tuple[str, tuple[float, float], int]]:
    specs = []
    for index, (x1, y1, x2, y2) in enumerate(crop_regions):
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        size = round(min(x2 - x1, y2 - y1) * INDEX_FONT_SCALE)
        specs.append((str(index), center, max(INDEX_FONT_MIN, min(INDEX_FONT_MAX, size))))
    return specs


def _draw_index_labels(canvas: np.ndarray, crop_regions: list[tuple[int, int, int, int]]) -> None:
    height, width = canvas.shape[:2]
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    blue = tuple(round(channel * 255) for channel in INDEX_BLUE) + (round(INDEX_ALPHA * 255),)
    outline = (0, 0, 0, round(INDEX_OUTLINE_ALPHA * 255))
    for text, center, font_size in _index_label_specs(crop_regions):
        font = ImageFont.load_default(size=font_size)
        draw.text(
            center,
            text,
            font=font,
            anchor="mm",
            fill=blue,
            stroke_width=max(1, font_size // 18),
            stroke_fill=outline,
        )
    pixels = np.asarray(overlay, dtype=np.float32) / 255.0
    alpha = pixels[:, :, 3:4]
    canvas[:] = canvas * (1.0 - alpha) + pixels[:, :, :3] * alpha


def preview_segs_regions(
    segs: Any,
    background: str,
    line_width: int,
    fallback_image_opt: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    if background not in ("image", "mask"):
        raise ValueError(f"Unsupported Preview SEGS Regions background {background!r}; expected 'image' or 'mask'.")
    line_width = integer(line_width, "line_width")
    if line_width <= 0:
        raise ValueError(f"Preview SEGS Regions line_width must be positive, got {line_width}.")

    source_shape, entries, crop_regions = validated_segs(segs, "Preview SEGS Regions")
    height = integer(source_shape[0], "SEGS source height")
    width = integer(source_shape[1], "SEGS source width")
    canvas = np.zeros((height, width, 3), dtype=np.float32)

    if background == "image":
        available = [index for index, segment in enumerate(entries) if getattr(segment, "cropped_image", None) is not None]
        if available:
            for index in available:
                x1, y1, x2, y2 = crop_regions[index]
                canvas[y1:y2, x1:x2] = _image_crop(entries[index].cropped_image, index, y2 - y1, x2 - x1)
        elif fallback_image_opt is not None:
            canvas = _fallback_image(fallback_image_opt, height, width)
    else:
        for index, segment in enumerate(entries):
            if getattr(segment, "cropped_mask", None) is None:
                continue
            x1, y1, x2, y2 = crop_regions[index]
            mask = np.clip(_mask_crop(segment.cropped_mask, index, y2 - y1, x2 - x1), 0.0, 1.0)
            target = canvas[y1:y2, x1:x2, 0]
            np.maximum(target, mask, out=target)
        canvas[:, :, 1] = canvas[:, :, 0]
        canvas[:, :, 2] = canvas[:, :, 0]

    coverage = coverage_map(height, width, crop_regions)
    yellow_area = coverage == 2
    red_area = coverage >= 3
    canvas[yellow_area] = canvas[yellow_area] * (1.0 - YELLOW_FILL_ALPHA) + np.array(
        (1.0, 1.0, 0.0), dtype=np.float32
    ) * YELLOW_FILL_ALPHA
    canvas[red_area] = canvas[red_area] * (1.0 - RED_FILL_ALPHA) + np.array(
        (1.0, 0.0, 0.0), dtype=np.float32
    ) * RED_FILL_ALPHA
    yy, xx = np.indices((height, width))
    hatch = (xx + yy) % HATCH_PERIOD < HATCH_WIDTH
    canvas[yellow_area & hatch] = (1.0, 1.0, 0.0)
    canvas[red_area & hatch] = (1.0, 0.0, 0.0)
    _draw_boundaries(canvas, crop_regions, line_width)
    _draw_index_labels(canvas, crop_regions)
    return torch.from_numpy(canvas.copy()).unsqueeze(0)


class PreviewSEGSRegions(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuitePreviewSEGSRegions",
            display_name="Preview SEGS Regions",
            category="Utility Suite/SEGS",
            description="Previews full-canvas SEGS crop geometry and overlap multiplicity.",
            inputs=[
                SEGS.Input("segs"),
                io.Image.Input("fallback_image_opt", optional=True),
                io.Combo.Input("background", options=["image", "mask"]),
                io.Int.Input("line_width", default=3, min=1, max=32768),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(
        cls,
        segs: Any,
        background: str,
        line_width: int,
        fallback_image_opt: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        return io.NodeOutput(preview_segs_regions(segs, background, line_width, fallback_image_opt))
