from __future__ import annotations

from typing import Any

from comfy_api.latest import io

from .segs_geometry import integer, validated_segs

SEGS = io.Custom("SEGS")


def seg_from_segs(segs: Any, index: int, length: int) -> tuple[Any, list[Any]]:
    header, entries, _ = validated_segs(segs, "SEG From SEGS")
    index = integer(index, "index")
    length = integer(length, "length")
    if index < 0:
        raise ValueError(f"SEG From SEGS index must be nonnegative, got {index}.")
    if length < 1:
        raise ValueError(f"SEG From SEGS length must be at least one, got {length}.")

    count = len(entries)
    if count == 0:
        return header, []
    index = min(index, count - 1)
    length = min(count - index, length)
    return header, list(entries[index : index + length])


class SEGFromSEGS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSEGFromSEGS",
            display_name="SEG From SEGS",
            category="Utility Suite/SEGS",
            description="Selects a contiguous zero-based range of SEG entries while preserving the SEGS header.",
            inputs=[
                SEGS.Input("segs"),
                io.Int.Input("index", default=0, min=0, max=32768),
                io.Int.Input("length", default=1, min=1, max=32768),
            ],
            outputs=[SEGS.Output("segs", display_name="segs")],
        )

    @classmethod
    def execute(cls, segs: Any, index: int, length: int) -> io.NodeOutput:
        return io.NodeOutput(seg_from_segs(segs, index, length))
