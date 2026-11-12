from __future__ import annotations

from typing import Any

import numpy as np
from comfy_api.latest import io

from .combine_tiles_segs import SEGS, _cropped_mask
from .segs_geometry import validated_segs


def combine_feather_segs(
    segs_destination: Any,
    segs_source: Any,
) -> tuple[Any, list[Any]]:
    destination_shape, destination_entries, destination_crops = validated_segs(
        segs_destination, "Combine Feather (SEGS) segs_destination"
    )
    source_shape, source_entries, source_crops = validated_segs(
        segs_source, "Combine Feather (SEGS) segs_source"
    )
    if tuple(destination_shape) != tuple(source_shape):
        raise ValueError(
            "Combine Feather (SEGS) requires matching SEGS source shapes, "
            f"got {tuple(destination_shape)} and {tuple(source_shape)}."
        )
    if not source_entries:
        return destination_shape, list(destination_entries)

    height, width = destination_shape
    source_max = np.zeros((height, width), dtype=np.float32)
    for index, (entry, crop) in enumerate(zip(source_entries, source_crops)):
        mask = _cropped_mask(entry, crop, index)
        x1, y1, x2, y2 = crop
        np.maximum(source_max[y1:y2, x1:x2], mask, out=source_max[y1:y2, x1:x2])

    output = []
    for index, (entry, crop) in enumerate(zip(destination_entries, destination_crops)):
        mask = _cropped_mask(entry, crop, index)
        x1, y1, x2, y2 = crop
        combined_mask = np.minimum(mask, source_max[y1:y2, x1:x2])
        output.append(entry._replace(cropped_mask=combined_mask))

    return destination_shape, output


class CombineFeatherSEGS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteCombineFeatherSEGS",
            display_name="Combine Feather (SEGS)",
            category="Utility Suite/SEGS",
            description="Intersects destination masks with the source group's pixelwise maximum mask while preserving destination SEGS structure.",
            inputs=[
                SEGS.Input("segs_destination"),
                SEGS.Input("segs_source"),
            ],
            outputs=[SEGS.Output("segs", display_name="segs")],
        )

    @classmethod
    def execute(cls, segs_destination: Any, segs_source: Any) -> io.NodeOutput:
        return io.NodeOutput(combine_feather_segs(segs_destination, segs_source))


__all__ = ["CombineFeatherSEGS", "combine_feather_segs"]
