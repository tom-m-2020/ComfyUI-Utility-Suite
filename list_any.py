from __future__ import annotations

from typing import Any

from comfy_api.latest import io


def extend_comfy_items(accumulator_items: list[Any], new_items: list[Any]) -> list[Any]:
    if not isinstance(accumulator_items, (list, tuple)) or not isinstance(new_items, (list, tuple)):
        raise TypeError("List Any (Append) requires Comfy whole-list input containers.")
    if not accumulator_items or (len(accumulator_items) == 1 and accumulator_items[0] is None):
        return list(new_items)
    return [*accumulator_items, *new_items]


class ListAnyAppend(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteListAnyAppend",
            display_name="List Any (Append)",
            category="Utility Suite/Utilities",
            description="Extends an accumulated Comfy list with incoming list items without interpreting them.",
            is_input_list=True,
            inputs=[
                io.AnyType.Input("any_1"),
                io.AnyType.Input("any_2"),
            ],
            outputs=[io.AnyType.Output("list", display_name="list", is_output_list=True)],
        )

    @classmethod
    def execute(cls, any_1: list[Any], any_2: list[Any]) -> io.NodeOutput:
        return io.NodeOutput(extend_comfy_items(any_1, any_2))
