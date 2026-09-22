from __future__ import annotations

from typing import Any

from comfy_api.latest import io

from .filter_tile_segs import _mask_array
from .segs_geometry import validated_segs

SEGS = io.Custom("SEGS")


def invert_mask_segs(segs: Any) -> tuple[Any, list[Any]]:
    header, entries, crops = validated_segs(segs, "Invert Mask (SEGS)")
    result: list[Any] = []
    for index, (entry, crop) in enumerate(zip(entries, crops)):
        if not hasattr(entry, "_replace") or not hasattr(entry, "cropped_mask"):
            raise TypeError(f"SEGS entry {index} must support structural cropped_mask replacement.")
        x1, y1, x2, y2 = crop
        original = _mask_array(entry, index, y2 - y1, x2 - x1)
        result.append(entry._replace(cropped_mask=1.0 - original))
    return header, result


class InvertMaskSEGS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteInvertMaskSEGS",
            display_name="Invert Mask (SEGS)",
            category="Utility Suite/SEGS",
            description="Inverts each SEG cropped_mask continuously as 1.0 - mask.",
            inputs=[SEGS.Input("segs")],
            outputs=[SEGS.Output("segs", display_name="segs")],
        )

    @classmethod
    def execute(cls, segs: Any) -> io.NodeOutput:
        return io.NodeOutput(invert_mask_segs(segs))
