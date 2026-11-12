from __future__ import annotations

from typing import Any

from comfy_api.latest import io

from .combine_tiles_segs import SEGS, combine_tiles_segs
from .invert_mask_segs import invert_mask_segs
from .segs_geometry import validated_segs


def combine_feather_segs(
    segs_destination: Any,
    segs_source: Any,
) -> tuple[Any, list[Any]]:
    _, destination_entries, _ = validated_segs(
        segs_destination, "Combine Feather (SEGS) segs_destination"
    )
    inverted_source = invert_mask_segs(segs_source)
    combined = combine_tiles_segs(
        {"segs0": segs_destination, "segs1": inverted_source}
    )
    return combined[0], combined[1][: len(destination_entries)]


class CombineFeatherSEGS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteCombineFeatherSEGS",
            display_name="Combine Feather (SEGS)",
            category="Utility Suite/SEGS",
            description="Combines destination SEGS with inverted source masks using zero-feather Combine Tiles semantics, then removes source entries.",
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
