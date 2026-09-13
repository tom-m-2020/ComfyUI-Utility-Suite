from __future__ import annotations

import operator
from typing import Any

from comfy_api.latest import io


def _coordinate(value: Any, entry_index: int, coordinate_index: int) -> int:
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(
            f"SEGS entry {entry_index} bbox coordinate {coordinate_index} must be an integer, "
            f"got {type(value).__name__}."
        ) from error


def segs_to_bboxes(
    segs: Any,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]], list[dict[str, int]]]:
    if not isinstance(segs, (list, tuple)) or len(segs) != 2:
        raise TypeError("SEGS to BBOX expects SEGS shaped as (source_size, segment_entries).")
    entries = segs[1]
    if not isinstance(entries, (list, tuple)):
        raise TypeError("SEGS segment entries must be a list or tuple.")

    xyxy: list[tuple[int, int, int, int]] = []
    xywh: list[tuple[int, int, int, int]] = []
    bounding_boxes: list[dict[str, int]] = []
    for entry_index, segment in enumerate(entries):
        if not hasattr(segment, "bbox"):
            raise TypeError(f"SEGS entry {entry_index} has no bbox field.")
        bbox = segment.bbox
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise TypeError(f"SEGS entry {entry_index} bbox must contain exactly four coordinates.")
        x1, y1, x2, y2 = (
            _coordinate(value, entry_index, coordinate_index) for coordinate_index, value in enumerate(bbox)
        )
        if x2 < x1 or y2 < y1:
            raise ValueError(f"SEGS entry {entry_index} bbox has reversed coordinates: {(x1, y1, x2, y2)}.")
        width = x2 - x1
        height = y2 - y1
        xyxy.append((x1, y1, x2, y2))
        xywh.append((x1, y1, width, height))
        bounding_boxes.append({"x": x1, "y": y1, "width": width, "height": height})
    return xyxy, xywh, bounding_boxes


class SEGSToBBOX(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSEGSToBBOX",
            display_name="SEGS to BBOX",
            category="Utility Suite/SEGS",
            description="Extracts authoritative Impact-style SEG bbox metadata as xyxy, xywh, and core boxes.",
            inputs=[io.Custom("SEGS").Input("segs")],
            outputs=[
                io.BBOX.Output("xyxy", display_name="xyxy", is_output_list=True),
                io.BBOX.Output("xywh", display_name="xywh", is_output_list=True),
                io.BoundingBox.Output("bounding_box", display_name="bounding_box", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, segs: Any) -> io.NodeOutput:
        return io.NodeOutput(*segs_to_bboxes(segs))
