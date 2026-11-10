from __future__ import annotations

import operator
from typing import Any

import numpy as np
import scipy.ndimage
import torch
from comfy_api.latest import io

from .segs_geometry import validated_segs

SEGS = io.Custom("SEGS")


def _slot_index(name: str, prefix: str) -> int:
    try:
        return int(name.removeprefix(prefix))
    except ValueError as error:
        raise ValueError(f"Combine Tiles (SEGS) received invalid input name {name!r}.") from error


def _cropped_mask(segment: Any, crop: tuple[int, int, int, int], index: int) -> np.ndarray:
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
            f"Combine Tiles (SEGS) entry {index} cropped_mask shape {mask.shape} "
            f"does not match crop_region extent {expected}."
        )
    if not np.isfinite(mask).all():
        raise ValueError(f"Combine Tiles (SEGS) entry {index} cropped_mask must be finite.")
    return np.clip(mask, 0.0, 1.0).astype(np.float32, copy=False)


def _inner_feather(envelope: np.ndarray, amount: int) -> np.ndarray:
    support = envelope > 0
    if amount == 0:
        return support.astype(np.float32)
    distance = scipy.ndimage.distance_transform_edt(support)
    return np.minimum(distance / amount, 1.0).astype(np.float32)


def combine_tiles_segs(
    segs_inputs: dict[str, Any],
    feather_inputs: dict[str, int] | None = None,
) -> tuple[Any, list[Any]]:
    if "segs0" not in segs_inputs:
        raise ValueError("Combine Tiles (SEGS) requires segs0.")

    feather_inputs = feather_inputs or {}
    unexpected = set(feather_inputs) - {
        f"feather{_slot_index(name, 'segs')}" for name in segs_inputs
    }
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ValueError(f"Combine Tiles (SEGS) has feather input without matching SEGS: {names}.")

    groups = []
    source_shape = None
    height = width = 0
    for group_index, name in enumerate(sorted(segs_inputs, key=lambda value: _slot_index(value, "segs"))):
        slot = _slot_index(name, "segs")
        shape, entries, crops = validated_segs(segs_inputs[name], "Combine Tiles (SEGS)")
        if source_shape is None:
            source_shape = shape
            height, width = shape
        elif tuple(shape) != tuple(source_shape):
            raise ValueError(
                "Combine Tiles (SEGS) requires matching SEGS source shapes, "
                f"got {tuple(source_shape)} and {tuple(shape)} for segs{slot}."
            )

        feather = feather_inputs.get(f"feather{slot}", 0)
        if isinstance(feather, bool):
            raise TypeError(f"Combine Tiles (SEGS) feather{slot} must be an integer, got bool.")
        try:
            feather = operator.index(feather)
        except TypeError as error:
            raise TypeError(
                f"Combine Tiles (SEGS) feather{slot} must be an integer, "
                f"got {type(feather).__name__}."
            ) from error
        if feather < 0:
            raise ValueError(
                f"Combine Tiles (SEGS) feather{slot} must be nonnegative, got {feather}."
            )

        masks = [
            _cropped_mask(segment, crop, index)
            for index, (segment, crop) in enumerate(zip(entries, crops))
        ]
        envelope = np.zeros((height, width), dtype=np.float32)
        for mask, crop in zip(masks, crops):
            x1, y1, x2, y2 = crop
            np.maximum(envelope[y1:y2, x1:x2], mask, out=envelope[y1:y2, x1:x2])
        feather_field = _inner_feather(envelope, feather)
        groups.append((group_index, entries, crops, masks, feather_field, envelope * feather_field))

    contributions: list[np.ndarray] = []
    total = np.zeros((height, width), dtype=np.float32)
    for *_, desired in groups:
        target = np.maximum(total, desired)
        remaining = target - desired
        scale = np.divide(
            remaining,
            total,
            out=np.zeros_like(total),
            where=total > 0,
        )
        for index in range(len(contributions)):
            contributions[index] *= scale
        contributions.append(desired.copy())
        total = target

    output = []
    for (_, entries, crops, masks, feather_field, desired), contribution in zip(
        groups, contributions
    ):
        allocation = np.divide(
            contribution,
            desired,
            out=np.zeros_like(desired),
            where=desired > 0,
        )
        group_scale = feather_field * allocation
        for segment, crop, mask in zip(entries, crops, masks):
            x1, y1, x2, y2 = crop
            cropped_mask = mask * group_scale[y1:y2, x1:x2]
            try:
                output.append(segment._replace(cropped_mask=cropped_mask))
            except (AttributeError, TypeError) as error:
                raise TypeError(
                    "Combine Tiles (SEGS) requires namedtuple-compatible SEG entries."
                ) from error

    return source_shape, output


class CombineTilesSEGS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        segs = io.Autogrow.TemplatePrefix(
            SEGS.Input("segs"),
            prefix="segs",
            min=1,
            max=50,
        )
        feathers = io.Autogrow.TemplatePrefix(
            io.Int.Input("feather", min=0, max=16384, step=1),
            prefix="feather",
            min=1,
            max=50,
        )
        return io.Schema(
            node_id="UtilitySuiteCombineTilesSEGS",
            display_name="Combine Tiles (SEGS)",
            category="Utility Suite/SEGS",
            description="Combines SEGS groups with group-boundary inner feathers and deficit-free crossfades between groups.",
            inputs=[
                io.Autogrow.Input("segs_inputs", template=segs),
                io.Autogrow.Input("feather_inputs", template=feathers, optional=True),
            ],
            outputs=[SEGS.Output("segs", display_name="segs")],
        )

    @classmethod
    def execute(
        cls,
        segs_inputs: io.Autogrow.Type,
        feather_inputs: io.Autogrow.Type | None = None,
    ) -> io.NodeOutput:
        return io.NodeOutput(combine_tiles_segs(segs_inputs, feather_inputs))


__all__ = ["CombineTilesSEGS", "combine_tiles_segs"]
