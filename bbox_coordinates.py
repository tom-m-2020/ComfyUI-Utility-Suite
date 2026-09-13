from __future__ import annotations

from numbers import Real
from typing import Any

from comfy_api.latest import io


def convert_bbox(bbox: Any, conversion: str) -> tuple[Real, Real, Real, Real]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise TypeError("BBOX xywh to xyxy expects one bbox containing exactly four numeric values.")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in bbox):
        raise TypeError("BBOX xywh to xyxy expects one bbox containing exactly four numeric values.")

    a, b, c, d = bbox
    if conversion == "xywh to xyxy":
        if c < 0 or d < 0:
            raise ValueError(f"xywh width and height must be nonnegative, got {(c, d)}.")
        return a, b, a + c, b + d
    if conversion == "xyxy to xywh":
        if c < a or d < b:
            raise ValueError(f"xyxy coordinates must not be reversed, got {(a, b, c, d)}.")
        return a, b, c - a, d - b
    raise ValueError(f"Unsupported bbox conversion: {conversion!r}.")


class BBOXCoordinates(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteBBOXCoordinates",
            display_name="BBOX xywh to xyxy",
            category="Utility Suite/BBOX",
            description="Converts one ordinary BBOX between exclusive-end xywh and xyxy coordinates.",
            inputs=[
                io.BBOX.Input("bbox"),
                io.Combo.Input("conversion", options=["xywh to xyxy", "xyxy to xywh"]),
            ],
            outputs=[io.BBOX.Output("bbox", display_name="bbox")],
        )

    @classmethod
    def execute(cls, bbox: Any, conversion: str) -> io.NodeOutput:
        return io.NodeOutput(convert_bbox(bbox, conversion))
