from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from comfy_api.latest import io

UTILITY_ANY_LIST = io.Custom("UTILITY_ANY_LIST")


@dataclass(frozen=True)
class UtilityAnyList:
    items: tuple[Any, ...] = ()

    def append_items(self, items: list[Any] | tuple[Any, ...]) -> UtilityAnyList:
        return UtilityAnyList((*self.items, *items))


def _input_accumulator(values: list[Any] | tuple[Any, ...] | None) -> UtilityAnyList:
    if values is None or not values or (len(values) == 1 and values[0] is None):
        return UtilityAnyList()
    if len(values) != 1:
        raise ValueError(f"accumulator requires exactly one opaque value; received {len(values)} items.")
    accumulator = values[0]
    if not isinstance(accumulator, UtilityAnyList):
        raise TypeError(f"accumulator must be UTILITY_ANY_LIST, got {type(accumulator).__name__}.")
    return accumulator


def append_list_items(accumulator: UtilityAnyList, items: list[Any] | tuple[Any, ...]) -> UtilityAnyList:
    if not isinstance(items, (list, tuple)):
        raise TypeError("items must be delivered as a Comfy transport list.")
    return accumulator.append_items(items)


class ListAccumulatorAppend(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteListAccumulatorAppend",
            display_name="List Accumulator Append",
            category="Utility Suite/Utilities",
            description="Appends outer Comfy-list items to one immutable loop-safe accumulator.",
            is_input_list=True,
            inputs=[
                UTILITY_ANY_LIST.Input("accumulator", optional=True),
                io.AnyType.Input("items"),
            ],
            outputs=[UTILITY_ANY_LIST.Output("accumulator", display_name="accumulator")],
        )

    @classmethod
    def execute(
        cls,
        items: list[Any],
        accumulator: list[UtilityAnyList] | None = None,
    ) -> io.NodeOutput:
        return io.NodeOutput(append_list_items(_input_accumulator(accumulator), items))


class ListAccumulatorToList(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteListAccumulatorToList",
            display_name="List Accumulator To List",
            category="Utility Suite/Utilities",
            description="Converts a loop-safe accumulator into a genuine Comfy list.",
            inputs=[UTILITY_ANY_LIST.Input("accumulator")],
            outputs=[io.AnyType.Output("items", display_name="items", is_output_list=True)],
        )

    @classmethod
    def execute(cls, accumulator: UtilityAnyList) -> io.NodeOutput:
        if not isinstance(accumulator, UtilityAnyList):
            raise TypeError(f"accumulator must be UTILITY_ANY_LIST, got {type(accumulator).__name__}.")
        return io.NodeOutput(list(accumulator.items))
