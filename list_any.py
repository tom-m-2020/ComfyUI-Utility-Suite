from __future__ import annotations

from typing import Any

from comfy_api.latest import io


def append_any(accumulator: Any, item: Any) -> list[Any]:
    if accumulator is None:
        return [item]
    if isinstance(accumulator, list):
        return [*accumulator, item]
    return [accumulator, item]


def _unwrap_new_item(values: list[Any]) -> Any:
    if len(values) == 1:
        return values[0]
    return list(values)


class ListAnyAppend(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteListAnyAppend",
            display_name="List Any (Append)",
            category="Utility Suite/Utilities",
            description="Appends one arbitrary value to a new or existing Comfy list without interpreting it.",
            is_input_list=True,
            inputs=[
                io.AnyType.Input("any_1"),
                io.AnyType.Input("any_2"),
            ],
            outputs=[io.AnyType.Output("list", display_name="list", is_output_list=True)],
        )

    @classmethod
    def execute(cls, any_1: list[Any], any_2: list[Any]) -> io.NodeOutput:
        accumulator = None if not any_1 or (len(any_1) == 1 and any_1[0] is None) else list(any_1)
        return io.NodeOutput(append_any(accumulator, _unwrap_new_item(any_2)))
