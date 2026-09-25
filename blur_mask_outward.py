from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np
import torch
from comfy_api.latest import io
from PIL import Image, ImageFilter


def _validate_radius(blur_radius: float) -> float:
    if isinstance(blur_radius, bool) or not isinstance(blur_radius, (int, float)):
        raise TypeError(f"Blur Mask Outward blur_radius must be FLOAT, got {type(blur_radius).__name__}.")
    radius = float(blur_radius)
    if not math.isfinite(radius) or radius < 0 or radius > 100:
        raise ValueError(f"Blur Mask Outward blur_radius must be finite and within 0..100, got {blur_radius}.")
    return radius


def _pillow_support_bound(radius: float) -> int:
    if radius == 0:
        return 0
    passes = 3
    sigma_squared = radius * radius / passes
    box_length = math.sqrt(12.0 * sigma_squared + 1.0)
    integer_radius = math.floor((box_length - 1.0) / 2.0)
    return passes * (integer_radius + 1)


def _pillow_blur_uint8(mask: np.ndarray, radius: float) -> np.ndarray:
    image = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8))
    if radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(image, dtype=np.float32) / 255.0


@lru_cache(maxsize=256)
def _minimum_straight_compensation(radius: float) -> int:
    if radius == 0:
        return 0
    support = _pillow_support_bound(radius)
    boundary = support + 2
    step = np.zeros((1, 2 * boundary), dtype=np.float32)
    step[:, :boundary] = 1.0
    blurred = _pillow_blur_uint8(step, radius)[0]
    attenuated = np.flatnonzero(blurred[:boundary] < 1.0)
    return 0 if not len(attenuated) else boundary - int(attenuated[0])


def _diamond_grow(mask: np.ndarray, amount: int) -> np.ndarray:
    if amount == 0:
        return mask
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    return cv2.dilate(mask, kernel, iterations=amount)


@lru_cache(maxsize=256)
def _minimum_tapered_compensation(radius: float) -> int:
    if radius == 0:
        return 0
    support = _pillow_support_bound(radius)
    upper = 2 * support
    center = upper + 2
    point = np.zeros((2 * center + 1, 2 * center + 1), dtype=np.float32)
    point[center, center] = 1.0

    def preserved(amount: int) -> bool:
        grown = _diamond_grow(point, amount)
        return _pillow_blur_uint8(grown, radius)[center, center] == 1.0

    low = _minimum_straight_compensation(radius)
    high = upper
    if not preserved(high):
        raise RuntimeError(f"Unable to derive tapered compensation for blur radius {radius}.")
    while low < high:
        middle = (low + high) // 2
        if preserved(middle):
            high = middle
        else:
            low = middle + 1
    return low


def _grow(mask: np.ndarray, amount: int, tapered_corners: bool) -> np.ndarray:
    if amount == 0:
        return mask
    if tapered_corners:
        return _diamond_grow(mask, amount)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * amount + 1, 2 * amount + 1))
    return cv2.dilate(mask, kernel, iterations=1)


def blur_mask_outward(
    mask: torch.Tensor,
    blur_radius: float,
    tapered_corners: bool = False,
    outward: bool = True,
) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"Blur Mask Outward expects a torch.Tensor MASK, got {type(mask).__name__}.")
    if mask.ndim == 2:
        batch = mask.unsqueeze(0)
        restore_rank_two = True
    elif mask.ndim == 3:
        batch = mask
        restore_rank_two = False
    else:
        raise ValueError(f"Blur Mask Outward expects MASK [H,W] or [B,H,W], got shape {tuple(mask.shape)}.")
    if any(dimension == 0 for dimension in batch.shape):
        raise ValueError(f"Blur Mask Outward requires nonempty dimensions, got shape {tuple(batch.shape)}.")
    if not isinstance(tapered_corners, bool):
        raise TypeError(
            f"Blur Mask Outward tapered_corners must be BOOLEAN, got {type(tapered_corners).__name__}."
        )
    if not isinstance(outward, bool):
        raise TypeError(f"Blur Mask Outward outward must be BOOLEAN, got {type(outward).__name__}.")

    radius = _validate_radius(blur_radius)
    if radius == 0:
        return mask.clone()

    source = batch.detach().to(device="cpu", dtype=torch.float32).numpy()
    if outward:
        grow_amount = (
            _minimum_tapered_compensation(radius)
            if tapered_corners
            else _minimum_straight_compensation(radius)
        )
        support = _pillow_support_bound(radius)
        padding = grow_amount + support
    else:
        grow_amount = 0
        padding = 0

    results = []
    for item in source:
        if outward:
            working = np.pad(item, padding, mode="edge")
            working = _grow(working, grow_amount, tapered_corners)
            blurred = _pillow_blur_uint8(working, radius)
            height, width = item.shape
            blurred = blurred[padding : padding + height, padding : padding + width]
            blurred[item == 1.0] = 1.0
        else:
            blurred = _pillow_blur_uint8(item, radius)
        results.append(torch.from_numpy(np.array(blurred, dtype=np.float32, copy=True)))

    output = torch.stack(results)
    return output[0] if restore_rank_two else output


class BlurMaskOutward(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteBlurMaskOutward",
            display_name="Blur Mask Outward",
            category="Utility Suite/Mask",
            inputs=[
                io.Mask.Input("mask"),
                io.Float.Input("blur_radius", default=0.0, min=0.0, max=100.0, step=0.1),
                io.Boolean.Input("tapered_corners", default=False),
                io.Boolean.Input("outward", default=True),
            ],
            outputs=[io.Mask.Output("mask", display_name="mask")],
        )

    @classmethod
    def execute(
        cls,
        mask: torch.Tensor,
        blur_radius: float,
        tapered_corners: bool,
        outward: bool,
    ) -> io.NodeOutput:
        return io.NodeOutput(blur_mask_outward(mask, blur_radius, tapered_corners, outward))


__all__ = [
    "BlurMaskOutward",
    "_minimum_straight_compensation",
    "_minimum_tapered_compensation",
    "blur_mask_outward",
]
