from __future__ import annotations

from typing import Any

from comfy_api.latest import io

from .filter_tile_segs import SEGS, partition_tile_segs
from .restore_tile_segs_gaps import restore_tile_segs_gaps


def remove_redundant_tile_segs(
    segs: Any,
    mode: str,
    min_overlap_x: int,
    min_overlap_y: int,
) -> tuple[Any, list[Any]]:
    kept_segs, excluded_segs = partition_tile_segs(segs, mode)
    return restore_tile_segs_gaps(
        kept_segs,
        excluded_segs,
        mode,
        min_overlap_x,
        min_overlap_y,
    )


class RemoveRedundantTileSEGS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteRemoveRedundantTileSEGS",
            display_name="Remove Redundant Tile SEGS",
            category="Utility Suite/SEGS",
            description="Filters redundant tile SEGS, then restores filtered tiles needed for minimum overlap continuity.",
            inputs=[
                SEGS.Input("segs"),
                io.Combo.Input("mode", options=["mask", "bbox"]),
                io.Int.Input("min_overlap_x", default=128, min=0, max=32768),
                io.Int.Input("min_overlap_y", default=128, min=0, max=32768),
            ],
            outputs=[SEGS.Output("segs", display_name="segs")],
        )

    @classmethod
    def execute(
        cls,
        segs: Any,
        mode: str,
        min_overlap_x: int,
        min_overlap_y: int,
    ) -> io.NodeOutput:
        return io.NodeOutput(remove_redundant_tile_segs(segs, mode, min_overlap_x, min_overlap_y))
