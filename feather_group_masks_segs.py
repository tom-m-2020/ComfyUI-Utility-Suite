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
) -> np.ndarray:
    mask = segment.cropped_mask
    if mask is None:
        return np.zeros((crop[3] - crop[1], crop[2] - crop[0]), dtype=np.float32)
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    expected = (crop[3] - crop[1], crop[2] - crop[0])
    if mask.ndim != 2 or mask.shape != expected:
        raise ValueError(
            f"Feather Group of Masks (SEGS) entry {index} cropped_mask shape "
            f"{mask.shape} does not match crop_region extent {expected}."
        )
    if not np.isfinite(mask).all():
        raise ValueError(
            f"Feather Group of Masks (SEGS) entry {index} cropped_mask must be finite."
        )
    return np.clip(mask, 0.0, 1.0).astype(np.float32, copy=False)


def _forward_ramp(support: np.ndarray, amount: int, axis: int) -> np.ndarray:
    factor = np.ones(support.shape, dtype=np.float32)
    if amount == 0:
        return factor
    run = np.zeros(support.shape[1 - axis], dtype=np.int32)
    length = support.shape[axis]
    for position in range(length):
        line = support[position, :] if axis == 0 else support[:, position]
        run = np.where(line, run + 1, 0)
        values = np.minimum(run / amount, 1.0)
        if axis == 0:
            factor[position, :] = np.where(line, values, 1.0)
        else:
            factor[:, position] = np.where(line, values, 1.0)
    return factor


def _directional_feather(
    support: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> np.ndarray:
    factor = np.ones(support.shape, dtype=np.float32)
    if left:
        factor *= _forward_ramp(support, left, axis=1)
    if right:
        factor *= np.flip(
            _forward_ramp(np.flip(support, axis=1), right, axis=1),
            axis=1,
        )
    if top:
        factor *= _forward_ramp(support, top, axis=0)
    if bottom:
        factor *= np.flip(
            _forward_ramp(np.flip(support, axis=0), bottom, axis=0),
            axis=0,
        )
    return factor


def feather_group_masks_segs(
    segs: Any,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> tuple[Any, list[Any]]:
    amounts = (left, top, right, bottom)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in amounts):
        raise TypeError("Feather Group of Masks (SEGS) feather amounts must be integers.")
    if any(value < 0 for value in amounts):
        raise ValueError(
            "Feather Group of Masks (SEGS) feather amounts must be nonnegative, "
            f"got {amounts}."
        )

    source_shape, entries, crops = validated_segs(
        segs, "Feather Group of Masks (SEGS)"
    )
    height, width = source_shape
    support = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in crops:
        support[y1:y2, x1:x2] = True
    feather = _directional_feather(support, left, top, right, bottom)

    output = []
    for index, (segment, crop) in enumerate(zip(entries, crops)):
        mask = _cropped_mask(segment, crop, index)
        x1, y1, x2, y2 = crop
        cropped_mask = mask * feather[y1:y2, x1:x2]
        try:
            output.append(segment._replace(cropped_mask=cropped_mask))
        except (AttributeError, TypeError) as error:
            raise TypeError(
                "Feather Group of Masks (SEGS) requires namedtuple-compatible SEG entries."
            ) from error
    return source_shape, output


class FeatherGroupOfMasksSEGS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteFeatherGroupOfMasksSEGS",
            display_name="Feather Group of Masks (SEGS)",
            category="Utility Suite/SEGS",
            description="Applies selected directional inner feathers to the union of all SEG crop regions as one group.",
            inputs=[
                SEGS.Input("segs"),
                io.Int.Input("left", default=0, min=0, max=16384, step=1),
                io.Int.Input("top", default=0, min=0, max=16384, step=1),
                io.Int.Input("right", default=0, min=0, max=16384, step=1),
                io.Int.Input("bottom", default=0, min=0, max=16384, step=1),
            ],
            outputs=[SEGS.Output("segs", display_name="segs")],
        )

    @classmethod
    def execute(
        cls,
        segs: Any,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            feather_group_masks_segs(segs, left, top, right, bottom)
        )


__all__ = ["FeatherGroupOfMasksSEGS", "feather_group_masks_segs"]
