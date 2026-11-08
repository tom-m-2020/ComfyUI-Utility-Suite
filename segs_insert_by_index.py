from __future__ import annotations

from typing import Any

from comfy_api.latest import io

from .segs_indices import parse_index_csv as _parse_index_csv

SEGS = io.Custom("SEGS")
SEGS_INDICES = io.Custom("SEGS_INDICES")


def parse_index_csv(index_text: str) -> list[int]:
    return _parse_index_csv(index_text, "SEGS Insert by Index")


def resolve_indices(index_text: str, index_input: Any) -> Any:
    if not isinstance(index_text, str):
        raise TypeError("SEGS Insert by Index manual index must be a string.")
    if index_text.strip():
        return parse_index_csv(index_text)
    if index_input is None:
        raise ValueError("SEGS Insert by Index requires a SEGS_INDICES input or manually entered indices.")
    return index_input


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
                SEGS_INDICES.Input("index", display_name="index", optional=True),
                io.String.Input(
                    "index_text",
                    display_name="index",
                    default="",
                    multiline=False,
                    tooltip="Optional comma-separated final positions. A nonempty value overrides the index socket.",
                ),
            ],
            outputs=[SEGS.Output("SEGS")],
        )

    @classmethod
    def execute(
        cls,
        segs_destination: Any,
        segs_source: Any,
        index: Any = None,
        index_text: str = "",
    ) -> io.NodeOutput:
        indices = resolve_indices(index_text, index)
        return io.NodeOutput(insert_segs_by_index(segs_destination, segs_source, indices))
