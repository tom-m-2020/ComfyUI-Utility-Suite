from __future__ import annotations

from typing import Any

from comfy_api.latest import io


def masks_from_list(masks: list[Any], start: int, length: int) -> list[Any]:
    length = min(length, len(masks))
    start = min(start, len(masks) - 1)
    length = min(len(masks) - start, length)
    return list(masks[start : start + length])


class MaskFromList(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteMaskFromList",
            display_name="Mask From List",
            category="Utility Suite/Mask",
            description="Selects a contiguous range of MASK list items without inspecting their tensor batches.",
            is_input_list=True,
            inputs=[
                io.Mask.Input("masks"),
                io.Int.Input("start", default=0, min=0, step=1),
                io.Int.Input("length", default=1, min=1, step=1),
            ],
            outputs=[io.Mask.Output(display_name="MASK", is_output_list=True)],
        )

    @classmethod
    def execute(cls, masks: list[Any], start: list[int], length: list[int]) -> io.NodeOutput:
        return io.NodeOutput(masks_from_list(masks, start[0], length[0]))
