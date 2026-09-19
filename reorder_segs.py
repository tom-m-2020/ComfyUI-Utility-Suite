from __future__ import annotations

from typing import Any, Literal

from comfy_api.latest import io

from .segs_geometry import validated_segs

SEGS = io.Custom("SEGS")
Direction = Literal["clockwise", "counter_clockwise"]
ORDERS = [
    "left_to_right_top_to_bottom",
    "clockwise",
    "counter_clockwise",
    "clockwise_outward",
    "counter_clockwise_outward",
]


def rectangular_ring(
    top: int,
    left: int,
    bottom: int,
    right: int,
    direction: Direction,
) -> list[tuple[int, int]]:
    if top > bottom or left > right:
        return []
    if top == bottom:
        return [(top, column) for column in range(left, right + 1)]
    if left == right:
        return [(row, left) for row in range(top, bottom + 1)]
    if direction == "clockwise":
        return (
            [(top, column) for column in range(left, right + 1)]
            + [(row, right) for row in range(top + 1, bottom + 1)]
            + [(bottom, column) for column in range(right - 1, left - 1, -1)]
            + [(row, left) for row in range(bottom - 1, top, -1)]
        )
    if direction == "counter_clockwise":
        return (
            [(row, left) for row in range(top, bottom + 1)]
            + [(bottom, column) for column in range(left + 1, right + 1)]
            + [(row, right) for row in range(bottom - 1, top - 1, -1)]
            + [(top, column) for column in range(right - 1, left, -1)]
        )
    raise ValueError(f"Unsupported ring direction {direction!r}.")


def inward_spiral(rows: int, columns: int, direction: Direction) -> list[tuple[int, int]]:
    if rows <= 0 or columns <= 0:
        return []
    result: list[tuple[int, int]] = []
    top, left, bottom, right = 0, 0, rows - 1, columns - 1
    while top <= bottom and left <= right:
        result.extend(rectangular_ring(top, left, bottom, right, direction))
        top += 1
        left += 1
        bottom -= 1
        right -= 1
    return result


def outward_spiral(rows: int, columns: int, direction: Direction) -> list[tuple[int, int]]:
    if rows <= 0 or columns <= 0:
        return []
    top = (rows - 1) // 2
    bottom = rows // 2
    left = (columns - 1) // 2
    right = columns // 2
    result: list[tuple[int, int]] = []
    emitted: set[tuple[int, int]] = set()
    while True:
        for coordinate in rectangular_ring(top, left, bottom, right, direction):
            if coordinate not in emitted:
                result.append(coordinate)
                emitted.add(coordinate)
        if top == 0 and left == 0 and bottom == rows - 1 and right == columns - 1:
            return result
        top = max(0, top - 1)
        left = max(0, left - 1)
        bottom = min(rows - 1, bottom + 1)
        right = min(columns - 1, right + 1)


def grid_coordinates(rows: int, columns: int, order: str) -> list[tuple[int, int]]:
    if order == "left_to_right_top_to_bottom":
        return [(row, column) for row in range(rows) for column in range(columns)]
    if order == "clockwise":
        return inward_spiral(rows, columns, "clockwise")
    if order == "counter_clockwise":
        return inward_spiral(rows, columns, "counter_clockwise")
    if order == "clockwise_outward":
        return outward_spiral(rows, columns, "clockwise")
    if order == "counter_clockwise_outward":
        return outward_spiral(rows, columns, "counter_clockwise")
    raise ValueError(f"Unsupported Reorder SEGS order {order!r}; expected one of {ORDERS}.")


def _ordered_spans(
    spans: set[tuple[int, int]], axis_name: str
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
    ordered = sorted(spans, key=lambda span: (span[0], span[1]))
    starts: dict[int, tuple[int, int]] = {}
    for span in ordered:
        if span[0] in starts and starts[span[0]] != span:
            raise ValueError(
                f"Reorder SEGS has ambiguous {axis_name} spans starting at {span[0]}: "
                f"{starts[span[0]]} and {span}."
            )
        starts[span[0]] = span
    return ordered, {span: index for index, span in enumerate(ordered)}


def reorder_segs(segs: Any, order: str) -> tuple[Any, list[Any]]:
    source_shape, entries, crops = validated_segs(segs, "Reorder SEGS")
    if order not in ORDERS:
        raise ValueError(f"Unsupported Reorder SEGS order {order!r}; expected one of {ORDERS}.")
    if not entries:
        return source_shape, []

    row_spans, row_indices = _ordered_spans({(crop[1], crop[3]) for crop in crops}, "row")
    column_spans, column_indices = _ordered_spans({(crop[0], crop[2]) for crop in crops}, "column")
    cells: dict[tuple[int, int], Any] = {}
    crop_regions: set[tuple[int, int, int, int]] = set()
    for entry, crop in zip(entries, crops):
        if crop in crop_regions:
            raise ValueError(f"Reorder SEGS found duplicate crop_region {crop}.")
        crop_regions.add(crop)
        cell = (row_indices[(crop[1], crop[3])], column_indices[(crop[0], crop[2])])
        if cell in cells:
            raise ValueError(f"Reorder SEGS found multiple entries in logical cell {cell}.")
        cells[cell] = entry

    traversal = grid_coordinates(len(row_spans), len(column_spans), order)
    return source_shape, [cells[cell] for cell in traversal if cell in cells]


class ReorderSEGS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteReorderSEGS",
            display_name="Reorder SEGS",
            category="Utility Suite/SEGS",
            description="Reorders tile-grid SEGS without changing their objects or fields.",
            inputs=[SEGS.Input("segs"), io.Combo.Input("order", options=ORDERS)],
            outputs=[SEGS.Output("segs", display_name="segs")],
        )

    @classmethod
    def execute(cls, segs: Any, order: str) -> io.NodeOutput:
        return io.NodeOutput(reorder_segs(segs, order))
