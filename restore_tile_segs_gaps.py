from __future__ import annotations

from collections import defaultdict
from itertools import pairwise
from typing import Any

from comfy_api.latest import io

from .filter_tile_segs import partition_tile_segs
from .segs_geometry import integer, validated_segs

SEGS = io.Custom("SEGS")


def _axis_overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int], axis: int) -> int:
    if axis == 0:
        return min(first[2], second[2]) - max(first[0], second[0])
    return min(first[3], second[3]) - max(first[1], second[1])


def _geometry_order(item: tuple[Any, tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    _, (x1, y1, x2, y2) = item
    return y1, x1, y2, x2


def restore_tile_segs_gaps(
    segs: Any,
    excluded_segs: Any,
    mode: str,
    min_overlap_x: int,
    min_overlap_y: int,
) -> tuple[Any, list[Any]]:
    if mode not in ("mask", "bbox"):
        raise ValueError(f"Unsupported Restore Tile SEGS Gaps mode {mode!r}; expected 'mask' or 'bbox'.")
    min_overlap_x = integer(min_overlap_x, "min_overlap_x")
    min_overlap_y = integer(min_overlap_y, "min_overlap_y")
    if min_overlap_x < 0 or min_overlap_y < 0:
        raise ValueError(
            "Restore Tile SEGS Gaps requires nonnegative minimum overlaps, "
            f"got {(min_overlap_x, min_overlap_y)}."
        )

    source_shape, kept_entries, kept_crops = validated_segs(segs, "Restore Tile SEGS Gaps")
    excluded_shape, excluded_entries, excluded_crops = validated_segs(
        excluded_segs, "Restore Tile SEGS Gaps"
    )
    if tuple(source_shape) != tuple(excluded_shape):
        raise ValueError(
            "Restore Tile SEGS Gaps requires matching SEGS source shapes, "
            f"got {tuple(source_shape)} and {tuple(excluded_shape)}."
        )

    kept_ids = {id(entry) for entry in kept_entries}
    excluded_ids = {id(entry) for entry in excluded_entries}
    if len(kept_ids) != len(kept_entries) or len(excluded_ids) != len(excluded_entries):
        raise ValueError("Restore Tile SEGS Gaps requires each input SEG object to occur only once.")
    if kept_ids & excluded_ids:
        raise ValueError("Restore Tile SEGS Gaps requires disjoint kept and excluded SEG objects.")

    original = list(zip(kept_entries, kept_crops)) + list(zip(excluded_entries, excluded_crops))
    crop_keys = [crop for _, crop in original]
    if len(set(crop_keys)) != len(crop_keys):
        raise ValueError("Restore Tile SEGS Gaps cannot reconstruct ordering with duplicate crop_regions.")
    original.sort(key=_geometry_order)

    rows: dict[tuple[int, int], list[tuple[Any, tuple[int, int, int, int]]]] = defaultdict(list)
    columns: dict[tuple[int, int], list[tuple[Any, tuple[int, int, int, int]]]] = defaultdict(list)
    for item in original:
        _, (x1, y1, x2, y2) = item
        rows[(y1, y2)].append(item)
        columns[(x1, x2)].append(item)
    for row in rows.values():
        row.sort(key=lambda item: (item[1][0], item[1][2]))
    for column in columns.values():
        column.sort(key=lambda item: (item[1][1], item[1][3]))

    candidate_ids: set[int] = set()

    def collect(
        sequence: list[tuple[Any, tuple[int, int, int, int]]],
        axis: int,
        required_overlap: int,
    ) -> None:
        if required_overlap == 0:
            return
        survivor_positions = [index for index, (entry, _) in enumerate(sequence) if id(entry) in kept_ids]
        for first_pos, second_pos in pairwise(survivor_positions):
            intermediates = sequence[first_pos + 1 : second_pos]
            if not intermediates:
                continue
            if _axis_overlap(sequence[first_pos][1], sequence[second_pos][1], axis) >= required_overlap:
                continue
            for entry, crop in intermediates:
                if id(entry) not in excluded_ids:
                    continue
                if axis == 0:
                    for related, _ in columns[(crop[0], crop[2])]:
                        if id(related) in excluded_ids:
                            candidate_ids.add(id(related))
                else:
                    for related, _ in rows[(crop[1], crop[3])]:
                        if id(related) in excluded_ids:
                            candidate_ids.add(id(related))

    for row in rows.values():
        collect(row, 0, min_overlap_x)
    for column in columns.values():
        collect(column, 1, min_overlap_y)

    candidates = [entry for entry, _ in original if id(entry) in candidate_ids]
    restored_ids: set[int] = set()
    if candidates:
        candidate_survivors, _ = partition_tile_segs((source_shape, candidates), mode)
        restored_ids = {id(entry) for entry in candidate_survivors[1]}

    output = [entry for entry, _ in original if id(entry) in kept_ids or id(entry) in restored_ids]
    return source_shape, output


class RestoreTileSEGSGaps(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteRestoreTileSEGSGaps",
            display_name="Restore Tile SEGS Gaps",
            category="Utility Suite/SEGS",
            description="Reconsiders excluded tile SEGS that bridge insufficient survivor continuity.",
            inputs=[
                SEGS.Input("segs"),
                SEGS.Input("excluded_segs"),
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
        excluded_segs: Any,
        mode: str,
        min_overlap_x: int,
        min_overlap_y: int,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            restore_tile_segs_gaps(segs, excluded_segs, mode, min_overlap_x, min_overlap_y)
        )
