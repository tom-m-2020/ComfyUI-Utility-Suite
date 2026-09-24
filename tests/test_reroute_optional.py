from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_reroute_optional_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.reroute_optional"]


class OutputNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("BOOLEAN",)}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True


class ImageSourceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("IMAGE",)


class RerouteOptionalTests(unittest.TestCase):
    def test_schema_uses_optional_matched_type_and_fixed_boolean(self):
        schema = MODULE.RerouteOptional.define_schema()
        value_input = schema.inputs[0]
        routed_output, present_output = schema.outputs

        self.assertEqual(schema.node_id, "UtilitySuiteRerouteOptional")
        self.assertEqual(schema.display_name, "Reroute (Optional)")
        self.assertTrue(value_input.optional)
        self.assertEqual(value_input.io_type, "COMFY_MATCHTYPE_V3")
        self.assertEqual(routed_output.io_type, "COMFY_MATCHTYPE_V3")
        self.assertEqual(value_input.template.template_id, routed_output.template.template_id)
        self.assertEqual(present_output.io_type, "BOOLEAN")

    def test_disconnected_input_returns_none_and_false(self):
        self.assertEqual(MODULE.RerouteOptional.execute().result, (None, False))

    def test_connected_value_preserves_identity(self):
        value = {"samples": torch.zeros((1, 4, 2, 2))}
        routed, present = MODULE.RerouteOptional.execute(value).result
        self.assertIs(routed, value)
        self.assertTrue(present)

    def test_falsey_values_are_present(self):
        values = [False, 0, 0.0, "", [], None, torch.zeros((1, 2, 2))]
        for value in values:
            with self.subTest(value=repr(value)):
                routed, present = MODULE.RerouteOptional.execute(value).result
                self.assertIs(routed, value)
                self.assertTrue(present)

    def test_backend_executor_accepts_omitted_optional_input(self):
        from execution import _async_map_node_over_list, get_input_data

        input_data, missing, v3_data = get_input_data(
            {}, MODULE.RerouteOptional, "reroute", execution_list=None
        )
        self.assertEqual(input_data, {})
        self.assertEqual(missing, {})

        results = asyncio.run(
            _async_map_node_over_list(
                "prompt",
                "reroute",
                MODULE.RerouteOptional,
                input_data,
                "execute",
                v3_data=v3_data,
            )
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].result, (None, False))

    def test_backend_executor_routes_connected_list_without_stacking(self):
        from execution import _async_map_node_over_list

        first = object()
        second = object()
        results = asyncio.run(
            _async_map_node_over_list(
                "prompt",
                "reroute",
                MODULE.RerouteOptional,
                {"value": [first, second]},
                "execute",
                v3_data={},
            )
        )
        self.assertEqual(len(results), 2)
        self.assertIs(results[0][0], first)
        self.assertIs(results[1][0], second)
        self.assertTrue(results[0][1])
        self.assertTrue(results[1][1])

    def test_disconnected_prompt_validates_through_boolean_output(self):
        import nodes
        from execution import validate_prompt

        prompt = {
            "reroute": {
                "class_type": "UtilitySuiteRerouteOptional",
                "inputs": {},
            },
            "output": {
                "class_type": "RerouteOptionalTestOutput",
                "inputs": {"value": ["reroute", 1]},
            },
        }
        mappings = {
            "UtilitySuiteRerouteOptional": MODULE.RerouteOptional,
            "RerouteOptionalTestOutput": OutputNode,
        }
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, mappings):
            valid, error, good_outputs, node_errors = asyncio.run(
                validate_prompt("prompt", prompt, None)
            )

        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertEqual(good_outputs, ["output"])
        self.assertEqual(node_errors, {})

    def test_connected_prompt_validates_with_dynamic_image_input(self):
        import nodes
        from execution import validate_prompt

        prompt = {
            "source": {"class_type": "RerouteOptionalTestImageSource", "inputs": {}},
            "reroute": {
                "class_type": "UtilitySuiteRerouteOptional",
                "inputs": {"value": ["source", 0]},
            },
            "output": {
                "class_type": "RerouteOptionalTestOutput",
                "inputs": {"value": ["reroute", 1]},
            },
        }
        mappings = {
            "UtilitySuiteRerouteOptional": MODULE.RerouteOptional,
            "RerouteOptionalTestImageSource": ImageSourceNode,
            "RerouteOptionalTestOutput": OutputNode,
        }
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, mappings):
            valid, error, good_outputs, node_errors = asyncio.run(
                validate_prompt("prompt", prompt, None)
            )

        self.assertTrue(valid, error)
        self.assertIsNone(error)
        self.assertEqual(good_outputs, ["output"])
        self.assertEqual(node_errors, {})

    def test_package_entrypoint_registers_node(self):
        extension = asyncio.run(PACKAGE.comfy_entrypoint())
        node_ids = {node.define_schema().node_id for node in asyncio.run(extension.get_node_list())}
        self.assertIn("UtilitySuiteRerouteOptional", node_ids)


if __name__ == "__main__":
    unittest.main()
