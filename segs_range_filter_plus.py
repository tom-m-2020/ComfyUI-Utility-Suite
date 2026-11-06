from __future__ import annotations

import logging
import sys
from typing import Any

from comfy_api.latest import io

SEGS = io.Custom("SEGS")
SEGS_INDICES = io.Custom("SEGS_INDICES")

TARGETS = [
    "area(=w*h)",
    "width",
    "height",
    "x1",
    "y1",
    "x2",
    "y2",
    "length_percent",
    "confidence(0-100)",
]


def filter_segs_range_plus(
    segs: Any,
    target: str,
    mode: bool,
    min_value: int,
    max_value: int,
) -> tuple[tuple[Any, list[Any]], tuple[Any, list[Any]], list[int]]:
    filtered_segs = []
    remained_segs = []
    indices = []

    for index, seg in enumerate(segs[1]):
        x1 = seg.crop_region[0]
        y1 = seg.crop_region[1]
        x2 = seg.crop_region[2]
        y2 = seg.crop_region[3]

        if target == "area(=w*h)":
            value = (y2 - y1) * (x2 - x1)
        elif target == "length_percent":
            h = y2 - y1
            w = x2 - x1
            value = max(h / w, w / h) * 100
        elif target == "width":
            value = x2 - x1
        elif target == "height":
            value = y2 - y1
        elif target == "x1":
            value = x1
        elif target == "x2":
            value = x2
        elif target == "y1":
            value = y1
        elif target == "y2":
            value = y2
        elif target == "confidence(0-100)":
            value = seg.confidence * 100
        else:
            raise Exception(f"[Impact Pack] SEGSRangeFilter - Unexpected target '{target}'")  # noqa: TRY002

        if mode and min_value <= value <= max_value:
            logging.info(f"[in] value={value} / {mode}, {min_value}, {max_value}")  # noqa: LOG015
            filtered_segs.append(seg)
            indices.append(index)
        elif not mode and (value < min_value or value > max_value):
            logging.info(f"[out] value={value} / {mode}, {min_value}, {max_value}")  # noqa: LOG015
            filtered_segs.append(seg)
            indices.append(index)
        else:
            remained_segs.append(seg)
            logging.info(f"[filter] value={value} / {mode}, {min_value}, {max_value}")  # noqa: LOG015

    return (segs[0], filtered_segs), (segs[0], remained_segs), indices


class SEGSRangeFilterPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSEGSRangeFilterPlus",
            display_name="SEGS Filter (range) Plus",
            category="Utility Suite/SEGS",
            description="Impact-compatible SEGS range filtering with original selected indices.",
            inputs=[
                SEGS.Input("segs"),
                io.Combo.Input("target", options=TARGETS),
                io.Boolean.Input("mode", default=True, label_on="inside", label_off="outside"),
                io.Int.Input("min_value", default=0, min=0, max=sys.maxsize, step=1),
                io.Int.Input("max_value", default=67108864, min=0, max=sys.maxsize, step=1),
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
        target: str,
        mode: bool,
        min_value: int,
        max_value: int,
    ) -> io.NodeOutput:
        return io.NodeOutput(*filter_segs_range_plus(segs, target, mode, min_value, max_value))
