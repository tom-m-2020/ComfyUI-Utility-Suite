from __future__ import annotations

from typing import Any

import torch
from comfy_api.latest import io


def classify_list_or_batch(values: list[Any]) -> tuple[str, int]:
    if not isinstance(values, (list, tuple)):
        raise TypeError("List / Batch Inspector must receive Comfy's whole-list input wrapper.")
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
            inputs=[io.AnyType.Input("value")],
            outputs=[
                io.String.Output("kind", display_name="kind"),
                io.Int.Output("count", display_name="count"),
            ],
        )

    @classmethod
    def execute(cls, value: list[Any]) -> io.NodeOutput:
        kind, count = classify_list_or_batch(value)
        return io.NodeOutput(kind, count)
