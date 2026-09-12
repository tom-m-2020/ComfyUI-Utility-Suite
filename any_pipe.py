from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from comfy_api.latest import io

UTILITY_ANY_PIPE = io.Custom("UTILITY_ANY_PIPE")


@dataclass(frozen=True)
class PipeEntry:
    key: str
    kind: Literal["scalar", "list"]
    value: Any


@dataclass(frozen=True)
class UtilityAnyPipe:
    entries: tuple[PipeEntry, ...] = ()

    def get(self, key: str) -> PipeEntry:
        for entry in self.entries:
            if entry.key == key:
                return entry
        raise KeyError(f"UTILITY_ANY_PIPE has no field {key!r}.")

    def set(self, entry: PipeEntry) -> UtilityAnyPipe:
        for index, current in enumerate(self.entries):
            if current.key == entry.key:
                return UtilityAnyPipe((*self.entries[:index], entry, *self.entries[index + 1 :]))
        return UtilityAnyPipe((*self.entries, entry))


def _single_transport(values: list[Any], name: str) -> Any:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be delivered as a Comfy transport list.")
    if len(values) != 1:
        raise ValueError(f"{name} requires exactly one scalar execution value; received {len(values)} items.")
    return values[0]


def _input_pipe(values: list[Any] | None) -> UtilityAnyPipe:
    if values is None or not values or (len(values) == 1 and values[0] is None):
        return UtilityAnyPipe()
    pipe = _single_transport(values, "pipe")
    if not isinstance(pipe, UtilityAnyPipe):
        raise TypeError(f"pipe must be UTILITY_ANY_PIPE, got {type(pipe).__name__}.")
    return pipe


def _input_key(values: list[Any]) -> str:
    key = _single_transport(values, "key")
    if not isinstance(key, str):
        raise TypeError(f"key must be STRING, got {type(key).__name__}.")
    return key


def set_scalar(pipe: UtilityAnyPipe, key: str, value: Any) -> UtilityAnyPipe:
    return pipe.set(PipeEntry(key=key, kind="scalar", value=value))


def set_list(pipe: UtilityAnyPipe, key: str, items: list[Any]) -> UtilityAnyPipe:
    return pipe.set(PipeEntry(key=key, kind="list", value=tuple(items)))


def get_scalar(pipe: UtilityAnyPipe, key: str) -> Any:
    entry = pipe.get(key)
    if entry.kind != "scalar":
        raise TypeError(f"UTILITY_ANY_PIPE field {key!r} is a list; use Pipe Get List.")
    return entry.value


def get_list(pipe: UtilityAnyPipe, key: str) -> list[Any]:
    entry = pipe.get(key)
    if entry.kind != "list":
        raise TypeError(f"UTILITY_ANY_PIPE field {key!r} is scalar; use Pipe Get Any.")
    return list(entry.value)


def _optional_field(pipe: UtilityAnyPipe, key: str, kind: Literal["scalar", "list"]) -> Any:
    try:
        entry = pipe.get(key)
    except KeyError:
        return None if kind == "scalar" else []
    if entry.kind != kind:
        raise TypeError(f"UTILITY_ANY_PIPE field {key!r} is {entry.kind}, expected {kind}.")
    return entry.value if kind == "scalar" else list(entry.value)


class PipeSetAny(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuitePipeSetAny",
            display_name="Pipe Set Any",
            category="Utility Suite/Pipe",
            description="Stores exactly one scalar execution value in an immutable utility pipe.",
            is_input_list=True,
            inputs=[
                UTILITY_ANY_PIPE.Input("pipe", optional=True),
                io.String.Input("key", default="value"),
                io.AnyType.Input("value"),
            ],
            outputs=[UTILITY_ANY_PIPE.Output("pipe", display_name="pipe")],
        )

    @classmethod
    def execute(cls, key: list[str], value: list[Any], pipe: list[UtilityAnyPipe] | None = None) -> io.NodeOutput:
        return io.NodeOutput(set_scalar(_input_pipe(pipe), _input_key(key), _single_transport(value, "value")))


class PipeSetList(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuitePipeSetList",
            display_name="Pipe Set List",
            category="Utility Suite/Pipe",
            description="Stores a complete outer Comfy list in an immutable utility pipe.",
            is_input_list=True,
            inputs=[
                UTILITY_ANY_PIPE.Input("pipe", optional=True),
                io.String.Input("key", default="items"),
                io.AnyType.Input("items"),
            ],
            outputs=[UTILITY_ANY_PIPE.Output("pipe", display_name="pipe")],
        )

    @classmethod
    def execute(cls, key: list[str], items: list[Any], pipe: list[UtilityAnyPipe] | None = None) -> io.NodeOutput:
        return io.NodeOutput(set_list(_input_pipe(pipe), _input_key(key), items))


class PipeGetAny(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuitePipeGetAny",
            display_name="Pipe Get Any",
            category="Utility Suite/Pipe",
            description="Gets a scalar field from a utility pipe.",
            inputs=[
                UTILITY_ANY_PIPE.Input("pipe"),
                io.String.Input("key", default="value"),
            ],
            outputs=[io.AnyType.Output("value", display_name="value")],
        )

    @classmethod
    def execute(cls, pipe: UtilityAnyPipe, key: str) -> io.NodeOutput:
        return io.NodeOutput(get_scalar(pipe, key))


class PipeGetList(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuitePipeGetList",
            display_name="Pipe Get List",
            category="Utility Suite/Pipe",
            description="Gets a Comfy-list field from a utility pipe.",
            inputs=[
                UTILITY_ANY_PIPE.Input("pipe"),
                io.String.Input("key", default="items"),
            ],
            outputs=[io.AnyType.Output("items", display_name="items", is_output_list=True)],
        )

    @classmethod
    def execute(cls, pipe: UtilityAnyPipe, key: str) -> io.NodeOutput:
        return io.NodeOutput(get_list(pipe, key))


class PipeToEditAny(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuitePipeToEditAny",
            display_name="Pipe To/Edit Any",
            category="Utility Suite/Pipe",
            description="Edits six scalar and three independent Comfy-list fields in an immutable utility pipe.",
            is_input_list=True,
            inputs=[
                UTILITY_ANY_PIPE.Input("pipe", optional=True),
                *[io.AnyType.Input(f"any_{index}", optional=True) for index in range(1, 7)],
                *[io.AnyType.Input(f"list_{index}", optional=True) for index in range(1, 4)],
            ],
            outputs=[UTILITY_ANY_PIPE.Output("pipe", display_name="pipe")],
        )

    @classmethod
    def execute(
        cls,
        pipe: list[UtilityAnyPipe] | None = None,
        **inputs: list[Any],
    ) -> io.NodeOutput:
        result = _input_pipe(pipe)
        for index in range(1, 7):
            key = f"any_{index}"
            if key in inputs:
                result = set_scalar(result, key, _single_transport(inputs[key], key))
        for index in range(1, 4):
            key = f"list_{index}"
            if key in inputs:
                result = set_list(result, key, inputs[key])
        return io.NodeOutput(result)


class PipeFromAny(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuitePipeFromAny",
            display_name="Pipe From Any",
            category="Utility Suite/Pipe",
            description="Extracts six scalar and three independent Comfy-list fields from a utility pipe.",
            inputs=[UTILITY_ANY_PIPE.Input("pipe")],
            outputs=[
                UTILITY_ANY_PIPE.Output("pipe", display_name="pipe"),
                *[io.AnyType.Output(f"any_{index}", display_name=f"any_{index}") for index in range(1, 7)],
                *[
                    io.AnyType.Output(f"list_{index}", display_name=f"list_{index}", is_output_list=True)
                    for index in range(1, 4)
                ],
            ],
        )

    @classmethod
    def execute(cls, pipe: UtilityAnyPipe) -> io.NodeOutput:
        if not isinstance(pipe, UtilityAnyPipe):
            raise TypeError(f"pipe must be UTILITY_ANY_PIPE, got {type(pipe).__name__}.")
        scalars = [_optional_field(pipe, f"any_{index}", "scalar") for index in range(1, 7)]
        lists = [_optional_field(pipe, f"list_{index}", "list") for index in range(1, 4)]
        return io.NodeOutput(pipe, *scalars, *lists)
