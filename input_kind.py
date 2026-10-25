from __future__ import annotations

from typing import Any

import torch
from comfy_api.latest import io


def _connected_output_kind(hidden: Any) -> str | None:
    dynprompt = getattr(hidden, "dynprompt", None)
    unique_id = getattr(hidden, "unique_id", None)
    if dynprompt is None or unique_id is None:
        return None

    node = dynprompt.get_node(unique_id)
    link = node.get("inputs", {}).get("value")
    if not isinstance(link, (list, tuple)) or len(link) != 2:
        return None

    import nodes

    source_node = dynprompt.get_node(link[0])
    source_class = nodes.NODE_CLASS_MAPPINGS[source_node["class_type"]]
    output_index = link[1]
    if not isinstance(output_index, int) or output_index < 0:
        return None

    if issubclass(source_class, io.ComfyNode):
        outputs = source_class.GET_SCHEMA().outputs
        if output_index >= len(outputs):
            return None
        output = outputs[output_index]
        if output.is_output_list:
            return "list"
        if output.io_type in ("IMAGE", "MASK"):
            return "batch"
        return None

    output_is_list = getattr(source_class, "OUTPUT_IS_LIST", ())
    if output_index < len(output_is_list) and output_is_list[output_index]:
        return "list"
    return_types = getattr(source_class, "RETURN_TYPES", ())
    if output_index < len(return_types) and return_types[output_index] in ("IMAGE", "MASK"):
        return "batch"
    return None


def classify_list_or_batch(values: list[Any], declared_kind: str | None = None) -> tuple[str, int]:
    if not isinstance(values, (list, tuple)):
        raise TypeError("List / Batch Inspector must receive Comfy's whole-list input wrapper.")
    if declared_kind == "list":
        return "list", 0 if not values or all(value is None for value in values) else len(values)
    if declared_kind == "batch":
        if not values or values[0] is None:
            return "batch", 0
        value = values[0]
        if isinstance(value, torch.Tensor) and value.ndim in (3, 4):
            return "batch", value.shape[0]
        return "batch", 0
    if len(values) != 1:
        return "list", len(values)

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
