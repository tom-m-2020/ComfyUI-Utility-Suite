from __future__ import annotations

import json
import math

import folder_paths
from comfy_api.latest import io
from nodes import LoraLoaderModelOnly


def parse_power_lora_rows(rows: str) -> list[tuple[str, float]]:
    try:
        parsed_rows = json.loads(rows)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Power LoRA rows must be a JSON array.") from error
    if not isinstance(parsed_rows, list):
        raise TypeError("Power LoRA rows must be a JSON array.")

    enabled_rows = []
    for index, row in enumerate(parsed_rows, start=1):
        if not isinstance(row, dict):
            raise TypeError(f"Power LoRA row {index} must be an object.")

        enabled = row.get("enabled", True)
        if not isinstance(enabled, bool):
            raise TypeError(f"Power LoRA row {index} enabled must be boolean.")
        if not enabled:
            continue

        lora_name = row.get("lora_name")
        strength = row.get("strength", 1.0)
        if not isinstance(lora_name, str) or not lora_name:
            raise ValueError(f"Power LoRA row {index} requires a LoRA name.")
        if isinstance(strength, bool) or not isinstance(strength, (int, float)):
            raise TypeError(f"Power LoRA row {index} strength must be a number.")
        strength = float(strength)
        if not math.isfinite(strength):
            raise ValueError(f"Power LoRA row {index} strength must be finite.")
        if strength != 0.0:
            enabled_rows.append((lora_name, strength))

    return enabled_rows


def load_power_loras(model, rows: str):
    for lora_name, strength in parse_power_lora_rows(rows):
        model = LoraLoaderModelOnly().load_lora_model_only(model, lora_name, strength)[0]
    return model


class PowerLoraLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuitePowerLoraLoader",
            display_name="Power LoRA Loader",
            category="Utility Suite/Loaders",
            description="Applies an ordered set of LoRAs through ComfyUI's native model-only LoRA loader.",
            inputs=[
                io.Model.Input("model"),
                io.String.Input(
                    "rows",
                    default="[]",
                    extra_dict={
                        "lora_names": folder_paths.get_filename_list("loras"),
                        "power_lora_rows": True,
                    },
                ),
            ],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(cls, model, rows: str) -> io.NodeOutput:
        return io.NodeOutput(load_power_loras(model, rows))


__all__ = ["PowerLoraLoader", "load_power_loras", "parse_power_lora_rows"]
