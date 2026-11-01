from __future__ import annotations

from collections import namedtuple

import cv2
import numpy as np
import torch
from comfy_api.latest import io

from .mask_bbox import quantized_mask_bounds

SEGS = io.Custom("SEGS")
SEG = namedtuple(
    "SEG",
    [
        "cropped_image",
        "cropped_mask",
        "confidence",
        "crop_region",
        "bbox",
        "label",
        "control_net_wrapper",
    ],
    defaults=[None],
)


def _mask_batch(mask: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor MASK, got {type(mask).__name__}."
        )
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"{name} expects MASK [B,H,W], got shape {tuple(mask.shape)}.")
    if any(dimension == 0 for dimension in mask.shape):
        raise ValueError(
            f"{name} requires nonempty dimensions, got shape {tuple(mask.shape)}."
        )
    return mask.detach().cpu().to(torch.float32)


def _normalize_axis(limit: int, start: int, size: float) -> tuple[int, int]:
    if start < 0:
        return 0, int(min(limit, size))
    if start + size > limit:
        return int(max(0, limit - size)), limit
    return start, int(min(limit, start + size))


def _crop_region(
    width: int,
    height: int,
    bbox: tuple[int, int, int, int],
    crop_width: float,
    crop_height: float,
) -> list[int]:
    x1, y1, x2, y2 = bbox
    center_x = x1 + (x2 - x1) / 2
    center_y = y1 + (y2 - y1) / 2
    start_x = int(center_x - crop_width / 2)
    start_y = int(center_y - crop_height / 2)
    crop_x1, crop_x2 = _normalize_axis(width, start_x, crop_width)
    crop_y1, crop_y2 = _normalize_axis(height, start_y, crop_height)
    return [crop_x1, crop_y1, crop_x2, crop_y2]


def _filled_contours(mask: np.ndarray) -> np.ndarray:
    quantized = (mask * 255.0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        quantized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(quantized)
    if hierarchy is not None:
        for index, contour in enumerate(contours):
            if hierarchy[0][index][3] == -1:
                cv2.drawContours(filled, [contour], 0, 255, -1)
    return filled.astype(np.float32) / 255.0


def batch_mask_to_segs_with_comparison(
    mask: torch.Tensor,
    crop_factor: float,
    bbox_fill: bool,
    contour_fill: bool,
    start_comparison: int,
    mask_base: torch.Tensor | None = None,
) -> tuple[tuple[int, int], list[SEG]]:
    targets = _mask_batch(mask, "Batch MASK to SEGS with Comparison mask")
    bases = None
    if mask_base is not None:
        bases = _mask_batch(mask_base, "Batch MASK to SEGS with Comparison mask_base")
        if bases.shape[1:] != targets.shape[1:]:
            raise ValueError(
                "Batch MASK to SEGS with Comparison requires mask and mask_base to share spatial dimensions, "
                f"got {tuple(targets.shape[1:])} and {tuple(bases.shape[1:])}."
            )

    height, width = targets.shape[1:]
    segments = []
    for index, target in enumerate(targets):
        bbox = quantized_mask_bounds(target)
        if bbox is None:
            bbox = (0, 0, 0, 0)
        target_width = bbox[2] - bbox[0]
        target_height = bbox[3] - bbox[1]

        if bases is None:
            crop_width = target_width * crop_factor
            crop_height = target_height * crop_factor
        else:
            base_index = index - start_comparison
            comparison_count = min(targets.shape[0], bases.shape[0])
            if 0 <= base_index < comparison_count:
                base_bbox = quantized_mask_bounds(bases[base_index])
                if base_bbox is None:
                    base_width = 0
                    base_height = 0
                else:
                    base_width = base_bbox[2] - base_bbox[0]
                    base_height = base_bbox[3] - base_bbox[1]
                crop_width = max(target_width, base_width * crop_factor)
                crop_height = max(target_height, base_height * crop_factor)
            else:
                crop_width = target_width
                crop_height = target_height

        if target_width == 0 or target_height == 0:
            crop_region = [0, 0, 1, 1]
        else:
            crop_region = _crop_region(
                width,
                height,
                bbox,
                crop_width,
                crop_height,
            )

        source = target.numpy()
        if contour_fill:
            source = _filled_contours(source)
        else:
            source = np.array(source, dtype=np.float32, copy=True)
        if bbox_fill and target_width > 0 and target_height > 0:
            source[bbox[1] : bbox[3], bbox[0] : bbox[2]] = 1.0
        cropped_mask = np.clip(
            source[
                crop_region[1] : crop_region[3],
                crop_region[0] : crop_region[2],
            ],
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        segments.append(
            SEG(
                None,
                cropped_mask,
                1.0,
                crop_region,
                bbox,
                "",
                None,
            )
        )

    return (height, width), segments


class BatchMaskToSEGSWithComparison(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteBatchMaskToSEGSWithComparison",
            display_name="Batch MASK to SEGS with Comparison",
            category="Utility Suite/SEGS",
            description="Creates one SEG per target mask with crop dimensions optionally compared against an offset base batch.",
            inputs=[
                io.Mask.Input("mask"),
                io.Mask.Input("mask_base", optional=True),
                io.Float.Input(
                    "crop_factor", default=3.0, min=1.0, max=100.0, step=0.1
                ),
                io.Boolean.Input(
                    "bbox_fill",
                    default=False,
                    label_on="enabled",
                    label_off="disabled",
                ),
                io.Boolean.Input(
                    "contour_fill",
                    default=False,
                    label_on="enabled",
                    label_off="disabled",
                ),
                io.Int.Input("start_comparison", default=0, min=-99, max=99, step=1),
            ],
            outputs=[SEGS.Output("segs", display_name="segs")],
        )

    @classmethod
    def execute(
        cls,
        mask: torch.Tensor,
        crop_factor: float,
        bbox_fill: bool,
        contour_fill: bool,
        start_comparison: int,
        mask_base: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            batch_mask_to_segs_with_comparison(
                mask,
                crop_factor,
                bbox_fill,
                contour_fill,
                start_comparison,
                mask_base,
            )
        )


__all__ = [
    "BatchMaskToSEGSWithComparison",
    "batch_mask_to_segs_with_comparison",
]
