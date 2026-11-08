from __future__ import annotations

from typing import Any

from comfy_api.latest import io

from .segs_geometry import integer

SEGS = io.Custom("SEGS")
SEGS_INDICES = io.Custom("SEGS_INDICES")


def _select_segs_range(
    segs: Any, index: int, length: int
) -> tuple[Any, list[Any], list[int]]:
    if not isinstance(segs, (list, tuple)) or len(segs) != 2:
        raise TypeError("SEG From SEGS expects SEGS shaped as (header, segment_entries).")
    header, entries = segs
    if not isinstance(entries, (list, tuple)):
        raise TypeError("SEG From SEGS segment entries must be a list or tuple.")
    index = integer(index, "index")
    length = integer(length, "length")
    if index < 0:
        raise ValueError(f"SEG From SEGS index must be nonnegative, got {index}.")
    if length < 1:
        raise ValueError(f"SEG From SEGS length must be at least one, got {length}.")

    count = len(entries)
    if count == 0:
        return header, [], []
    index = min(index, count - 1)
    length = min(count - index, length)
    stop = index + length
    return header, list(entries[index:stop]), list(range(index, stop))


def seg_from_segs(segs: Any, index: int, length: int) -> tuple[Any, list[Any]]:
    header, entries, _ = _select_segs_range(segs, index, length)
    return header, entries


def seg_from_segs_with_indices(
    segs: Any, index: int, length: int
) -> tuple[tuple[Any, list[Any]], list[int]]:
    header, entries, indices = _select_segs_range(segs, index, length)
    return (header, entries), indices


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
            outputs=[
                SEGS.Output("segs", display_name="segs"),
                SEGS_INDICES.Output("index", display_name="index"),
            ],
        )

    @classmethod
    def execute(cls, segs: Any, index: int, length: int) -> io.NodeOutput:
        selected, indices = seg_from_segs_with_indices(segs, index, length)
        return io.NodeOutput(selected, indices)
