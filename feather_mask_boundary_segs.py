from __future__ import annotations

from typing import Any

import torch
from comfy_api.latest import io

from .feather_mask_boundary import feather_mask_from_boundary
from .filter_tile_segs import _mask_array
from .segs_geometry import validated_segs

SEGS = io.Custom("SEGS")


def feather_mask_from_boundary_segs(
    segs: Any,
    left: int,
    top: int,
    right: int,
    bottom: int,
    outward: bool,
    tapered_corners: bool,
) -> tuple[Any, list[Any]]:
    header, entries, crops = validated_segs(segs, "Feather Mask from Boundary (SEGS)")
    output: list[Any] = []
    for index, (entry, crop) in enumerate(zip(entries, crops)):
        if not hasattr(entry, "_replace") or not hasattr(entry, "cropped_mask"):
            raise TypeError(f"SEGS entry {index} must support structural cropped_mask replacement.")
        x1, y1, x2, y2 = crop
        mask = _mask_array(entry, index, y2 - y1, x2 - x1)
        mask_tensor = torch.from_numpy(mask)
        feathered = feather_mask_from_boundary(
            mask_tensor,
            left,
            top,
            right,
            bottom,
            outward,
            tapered_corners,
        )
        output.append(entry._replace(cropped_mask=feathered.numpy()))
    return header, output


class FeatherMaskFromBoundarySEGS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteFeatherMaskFromBoundarySEGS",
            display_name="Feather Mask from Boundary (SEGS)",
            category="Utility Suite/SEGS",
            description="Applies Feather Mask from Boundary independently to every SEG cropped_mask.",
            inputs=[
                SEGS.Input("segs"),
                io.Int.Input("left", default=0, min=0, max=16384, step=1),
                io.Int.Input("top", default=0, min=0, max=16384, step=1),
                io.Int.Input("right", default=0, min=0, max=16384, step=1),
                io.Int.Input("bottom", default=0, min=0, max=16384, step=1),
                io.Boolean.Input("outward", default=False),
                io.Boolean.Input("tapered_corners", default=False),
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
        outward: bool,
        tapered_corners: bool,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            feather_mask_from_boundary_segs(
                segs,
                left,
                top,
                right,
                bottom,
                outward,
                tapered_corners,
            )
        )


__all__ = ["FeatherMaskFromBoundarySEGS", "feather_mask_from_boundary_segs"]
