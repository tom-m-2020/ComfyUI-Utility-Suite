from __future__ import annotations

from typing import Any

import torch
from comfy_api.latest import io


def _source_output_kind(
    dynprompt: Any,
    node_id: Any,
    output_index: int,
    visiting: set[Any] | None = None,
) -> str | None:
    import nodes

    if not isinstance(output_index, int) or output_index < 0:
        return None
    if visiting is None:
        visiting = set()
    if node_id in visiting:
        return None
    visiting = {*visiting, node_id}

    source_node = dynprompt.get_node(node_id)
    source_class = nodes.NODE_CLASS_MAPPINGS[source_node["class_type"]]

    if issubclass(source_class, io.ComfyNode):
        schema = source_class.GET_SCHEMA()
        outputs = schema.outputs
        if output_index >= len(outputs):
            return None
        output = outputs[output_index]
        if output.is_output_list:
            return "list"
        output_type = output.io_type
        consumes_lists = schema.is_input_list
    else:
        output_is_list = getattr(source_class, "OUTPUT_IS_LIST", ())
        if output_index < len(output_is_list) and output_is_list[output_index]:
            return "list"
        return_types = getattr(source_class, "RETURN_TYPES", ())
        output_type = return_types[output_index] if output_index < len(return_types) else None
        consumes_lists = getattr(source_class, "INPUT_IS_LIST", False)

    if not consumes_lists:
        for input_value in source_node.get("inputs", {}).values():
            if not isinstance(input_value, (list, tuple)) or len(input_value) != 2:
                continue
            if _source_output_kind(dynprompt, input_value[0], input_value[1], visiting) == "list":
                return "list"

    if output_type in ("IMAGE", "MASK"):
        return "batch"
    return None


def _connected_output_kind(hidden: Any) -> str | None:
    dynprompt = getattr(hidden, "dynprompt", None)
    unique_id = getattr(hidden, "unique_id", None)
    if dynprompt is None or unique_id is None:
        return None

    node = dynprompt.get_node(unique_id)
    link = node.get("inputs", {}).get("value")
    if not isinstance(link, (list, tuple)) or len(link) != 2:
        return None
    return _source_output_kind(dynprompt, link[0], link[1])


def classify_list_or_batch(values: list[Any], declared_kind: str | None = None) -> tuple[str, int]:
    if not isinstance(values, (list, tuple)):
        raise TypeError("List / Batch Inspector must receive Comfy's whole-list input wrapper.")
    if len(values) > 1:
        return "list", len(values)
    if declared_kind == "list":
        return "list", 0 if not values or all(value is None for value in values) else len(values)
    if declared_kind == "batch":
        if not values or values[0] is None:
            return "batch", 0
        value = values[0]
        if isinstance(value, torch.Tensor) and value.ndim in (3, 4):
            return "batch", value.shape[0]
        return "batch", 0
    if not values:
        return "list", 0

    value = values[0]
    if isinstance(value, (list, tuple)):
        return "list", len(value)
    if isinstance(value, torch.Tensor) and value.ndim in (3, 4):
        return "batch", value.shape[0]
    return "unknown", 0


class ListBatchInspector(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteListBatchInspector",
            display_name="List / Batch Inspector",
            category="Utility Suite/Utilities",
            description="Reports whether an input is a Comfy list, an IMAGE/MASK tensor batch, or unknown.",
            is_input_list=True,
            hidden=[io.Hidden.dynprompt, io.Hidden.unique_id],
            inputs=[io.AnyType.Input("value")],
            outputs=[
                io.String.Output("kind", display_name="kind"),
                io.Int.Output("count", display_name="count"),
            ],
        )

    @classmethod
    def execute(cls, value: list[Any]) -> io.NodeOutput:
        kind, count = classify_list_or_batch(value, _connected_output_kind(cls.hidden))
        return io.NodeOutput(kind, count)
