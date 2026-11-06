from __future__ import annotations

from typing import Any

from comfy_api.latest import io

SEGS = io.Custom("SEGS")
SEGS_INDICES = io.Custom("SEGS_INDICES")


def _segs_parts(segs: Any, input_name: str) -> tuple[Any, list[Any] | tuple[Any, ...]]:
    if not isinstance(segs, (list, tuple)) or len(segs) != 2:
        raise TypeError(f"{input_name} must be a (source-size header, entries) SEGS pair.")
    header, entries = segs
    if not isinstance(header, (list, tuple)) or len(header) != 2:
        raise TypeError(f"{input_name} source-size header must contain height and width.")
    if not isinstance(entries, (list, tuple)):
        raise TypeError(f"{input_name} entries must be a list or tuple.")
    return header, entries


def insert_segs_by_index(segs_destination: Any, segs_source: Any, index: Any) -> tuple[Any, list[Any]]:
    destination_header, destination = _segs_parts(segs_destination, "segs_destination")
    source_header, source = _segs_parts(segs_source, "segs_source")

    if tuple(destination_header) != tuple(source_header):
        raise ValueError(
            "SEGS Insert by Index requires matching source-size headers; "
            f"got {tuple(destination_header)} and {tuple(source_header)}."
        )
    if not isinstance(index, list):
        raise TypeError("index must be a list of integers.")
    if len(index) != len(source):
        raise ValueError(
            f"SEGS Insert by Index received {len(source)} source entries but {len(index)} indices."
        )

    total = len(destination) + len(source)
    occupied = [False] * total
    result: list[Any] = [None] * total
    for source_position, output_position in enumerate(index):
        if isinstance(output_position, bool) or not isinstance(output_position, int):
            raise TypeError(f"index[{source_position}] must be an integer, got {type(output_position).__name__}.")
        if output_position < 0 or output_position >= total:
            raise ValueError(
                f"index[{source_position}]={output_position} is outside the valid final range 0..{total - 1}."
            )
        if occupied[output_position]:
            raise ValueError(f"index contains duplicate final position {output_position}.")
        occupied[output_position] = True
        result[output_position] = source[source_position]

    destination_entries = iter(destination)
    for output_position in range(total):
        if not occupied[output_position]:
            result[output_position] = next(destination_entries)

    return destination_header, result


class SEGSInsertByIndex(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSEGSInsertByIndex",
            display_name="SEGS Insert by Index",
            category="Utility Suite/SEGS",
            description="Reconstructs SEGS by inserting source entries at specified final positions.",
            inputs=[
                SEGS.Input("segs_destination"),
                SEGS.Input("segs_source"),
                SEGS_INDICES.Input("index"),
            ],
            outputs=[SEGS.Output("SEGS")],
        )

    @classmethod
    def execute(cls, segs_destination: Any, segs_source: Any, index: Any) -> io.NodeOutput:
        return io.NodeOutput(insert_segs_by_index(segs_destination, segs_source, index))
