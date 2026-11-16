from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_power_lora_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.power_lora_loader"]


class PowerLoraLoaderTests(unittest.TestCase):
    def test_schema_and_registration(self):
        with patch.object(MODULE.folder_paths, "get_filename_list", return_value=["a.safetensors"]):
            schema = MODULE.PowerLoraLoader.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuitePowerLoraLoader")
        self.assertEqual(schema.display_name, "Power LoRA Loader")
        self.assertEqual([item.id for item in schema.inputs], ["model", "rows"])
        self.assertEqual(schema.inputs[1].extra_dict["lora_names"], ["a.safetensors"])
        self.assertTrue(schema.inputs[1].extra_dict["power_lora_rows"])
        self.assertIn(MODULE.PowerLoraLoader, asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list()))

    def test_applies_enabled_nonzero_rows_in_order(self):
        calls = []

        class Loader:
            def load_lora_model_only(self, model, name, strength):
                calls.append((model, name, strength))
                return (f"{model}+{name}",)

        rows = json.dumps([
            {"lora_name": "first.safetensors", "strength": -0.5, "enabled": True},
            {"lora_name": "disabled.safetensors", "strength": 2, "enabled": False},
            {"lora_name": "zero.safetensors", "strength": 0, "enabled": True},
            {"lora_name": "last.safetensors", "strength": 1.25, "enabled": True},
        ])
        with patch.object(MODULE, "LoraLoaderModelOnly", Loader):
            result = MODULE.load_power_loras("base", rows)

        self.assertEqual(result, "base+first.safetensors+last.safetensors")
        self.assertEqual(calls, [
            ("base", "first.safetensors", -0.5),
            ("base+first.safetensors", "last.safetensors", 1.25),
        ])

    def test_empty_rows_preserve_model_identity(self):
        model = object()
        self.assertIs(MODULE.load_power_loras(model, "[]"), model)

    def test_rejects_invalid_rows(self):
        invalid = [
            ("not json", "JSON array"),
            ("{}", "JSON array"),
            ("[1]", "must be an object"),
            ('[{"enabled": 1}]', "enabled must be boolean"),
            ('[{"enabled": true}]', "requires a LoRA name"),
            ('[{"enabled": true, "lora_name": "a", "strength": true}]', "strength must be a number"),
            ('[{"enabled": true, "lora_name": "a", "strength": 1e999}]', "strength must be finite"),
        ]
        for rows, message in invalid:
            with self.subTest(rows=rows), self.assertRaisesRegex((TypeError, ValueError), message):
                MODULE.parse_power_lora_rows(rows)


if __name__ == "__main__":
    unittest.main()
