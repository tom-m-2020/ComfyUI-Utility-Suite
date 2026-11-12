from __future__ import annotations

from typing import Any

import numpy as np
import torch
from comfy_api.latest import io

from .segs_geometry import validated_segs

SEGS = io.Custom("SEGS")


def _cropped_mask(
    segment: Any,
    crop: tuple[int, int, int, int],
    index: int,
    input_name: str,
) -> np.ndarray:
    if not hasattr(segment, "cropped_mask") or segment.cropped_mask is None:
        raise TypeError(
            f"Normalize Feathered SEGS {input_name} entry {index} has no cropped_mask."
        )
    mask = segment.cropped_mask
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    expected = (crop[3] - crop[1], crop[2] - crop[0])
    if mask.ndim != 2 or mask.shape != expected:
        raise ValueError(
            f"Normalize Feathered SEGS {input_name} entry {index} cropped_mask "
            f"shape {mask.shape} does not match crop_region extent {expected}."
        )
    if not np.isfinite(mask).all():
        raise ValueError(
            f"Normalize Feathered SEGS {input_name} entry {index} cropped_mask must be finite."
        )
    return np.clip(mask, 0.0, 1.0).astype(np.float32, copy=False)


def normalize_feathered_segs(
    feathered_segs: Any,
    reference_segs: Any,
) -> tuple[Any, list[Any]]:
    header, feathered_entries, feathered_crops = validated_segs(
        feathered_segs, "Normalize Feathered SEGS feathered_segs"
    )
    reference_header, reference_entries, reference_crops = validated_segs(
        reference_segs, "Normalize Feathered SEGS reference_segs"
    )
    if tuple(header) != tuple(reference_header):
        raise ValueError(
            "Normalize Feathered SEGS requires matching source shapes, "
            f"got {tuple(header)} and {tuple(reference_header)}."
        )
    if len(feathered_entries) != len(reference_entries):
        raise ValueError(
            "Normalize Feathered SEGS requires matching entry counts, "
            f"got {len(feathered_entries)} and {len(reference_entries)}."
        )
    for index, (feathered_crop, reference_crop) in enumerate(
        zip(feathered_crops, reference_crops)
    ):
        if feathered_crop != reference_crop:
            raise ValueError(
                "Normalize Feathered SEGS requires matching crop_region values "
                f"at each index; entry {index} has {feathered_crop} and {reference_crop}."
            )

    feathered_masks = [
        _cropped_mask(entry, crop, index, "feathered_segs")
        for index, (entry, crop) in enumerate(zip(feathered_entries, feathered_crops))
    ]
    reference_masks = [
        _cropped_mask(entry, crop, index, "reference_segs")
        for index, (entry, crop) in enumerate(zip(reference_entries, reference_crops))
    ]

    height, width = header
    feathered_sum = np.zeros((height, width), dtype=np.float32)
    reference_target = np.zeros((height, width), dtype=np.float32)
    for feathered_mask, reference_mask, crop in zip(
        feathered_masks, reference_masks, feathered_crops
    ):
        x1, y1, x2, y2 = crop
        feathered_sum[y1:y2, x1:x2] += feathered_mask
        np.maximum(
            reference_target[y1:y2, x1:x2],
            reference_mask,
            out=reference_target[y1:y2, x1:x2],
        )

    scale = np.divide(
        reference_target,
        feathered_sum,
        out=np.zeros_like(reference_target),
        where=feathered_sum > 0,
    )
    np.minimum(scale, 1.0, out=scale)

    output: list[Any] = []
    for index, (entry, crop, mask) in enumerate(
        zip(feathered_entries, feathered_crops, feathered_masks)
    ):
        if not hasattr(entry, "_replace"):
            raise TypeError(
                f"Normalize Feathered SEGS feathered entry {index} must support "
                "structural cropped_mask replacement."
            )
        x1, y1, x2, y2 = crop
        output.append(entry._replace(cropped_mask=mask * scale[y1:y2, x1:x2]))
    return header, output


class NormalizeFeatheredSEGS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteNormalizeFeatheredSEGS",
            display_name="Normalize Feathered SEGS",
            category="Utility Suite/SEGS",
            description="Caps overlapping feathered-mask sums to the reference SEGS envelope without increasing feather values.",
            inputs=[
                SEGS.Input("feathered_segs"),
                SEGS.Input("reference_segs"),
            ],
            outputs=[SEGS.Output("segs", display_name="segs")],
        )

    @classmethod
    def execute(cls, feathered_segs: Any, reference_segs: Any) -> io.NodeOutput:
        return io.NodeOutput(normalize_feathered_segs(feathered_segs, reference_segs))


__all__ = ["NormalizeFeatheredSEGS", "normalize_feathered_segs"]
