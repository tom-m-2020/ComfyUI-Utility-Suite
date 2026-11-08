from __future__ import annotations

from typing import Any

from comfy_api.latest import io

from .segs_indices import parse_index_csv, validate_segs_indices

SEGS = io.Custom("SEGS")
SEGS_INDICES = io.Custom("SEGS_INDICES")


def resolve_filter_indices(index_text: str, index_input: Any, count: int) -> list[int]:
    if not isinstance(index_text, str):
        raise TypeError("SEGS Filter (index) manual index must be a string.")
    if index_text.strip():
        indices = parse_index_csv(index_text, "SEGS Filter (index)")
    elif index_input is not None:
        indices = index_input
    else:
        raise ValueError("SEGS Filter (index) requires a SEGS_INDICES input or manually entered indices.")
    return validate_segs_indices(indices, count, "SEGS Filter (index)")


def filter_segs_by_index(
    segs: Any, indices: list[int]
) -> tuple[tuple[Any, list[Any]], tuple[Any, list[Any]], list[int]]:
    selected_positions = set(indices)
    filtered = [segs[1][index] for index in indices]
    remained = [seg for index, seg in enumerate(segs[1]) if index not in selected_positions]
    return (segs[0], filtered), (segs[0], remained), list(indices)


class SEGSFilterIndex(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSEGSFilterIndex",
            display_name="SEGS Filter (index)",
            category="Utility Suite/SEGS",
            description="Partitions SEGS by selected original positions and returns their paired indices.",
            inputs=[
                SEGS.Input("segs"),
                SEGS_INDICES.Input("index", display_name="index", optional=True),
                io.String.Input(
                    "index_text",
                    display_name="index",
                    default="",
                    multiline=False,
                    tooltip="Optional comma-separated SEG positions. A nonempty value overrides the index socket.",
                ),
            ],
            outputs=[
                SEGS.Output("filtered_SEGS"),
                SEGS.Output("remained_SEGS"),
                SEGS_INDICES.Output("index"),
            ],
        )

    @classmethod
    def execute(
        cls,
        segs: Any,
        index: Any = None,
        index_text: str = "",
    ) -> io.NodeOutput:
        indices = resolve_filter_indices(index_text, index, len(segs[1]))
        return io.NodeOutput(*filter_segs_by_index(segs, indices))
