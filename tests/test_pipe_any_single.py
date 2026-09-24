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
    "utility_suite_pipe_any_single_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
ANY_PIPE = sys.modules[f"{SPEC.name}.any_pipe"]


class BooleanOutputNode:
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


class PipeAnySingleTests(unittest.TestCase):
    def test_schema_is_optional_matched_type_with_boolean(self):
        schema = ANY_PIPE.PipeAnySingle.define_schema()
        value_input = schema.inputs[0]
        value_output, present_output = schema.outputs

        self.assertEqual(schema.node_id, "UtilitySuitePipeAnySingle")
        self.assertEqual(schema.category, "Utility Suite/Routing")
        self.assertTrue(value_input.optional)
        self.assertFalse(schema.is_input_list)
        self.assertEqual(value_input.io_type, "COMFY_MATCHTYPE_V3")
        self.assertEqual(value_output.io_type, "COMFY_MATCHTYPE_V3")
        self.assertEqual(value_input.template.template_id, value_output.template.template_id)
        self.assertEqual(present_output.io_type, "BOOLEAN")

    def test_disconnected_returns_none_and_false(self):
        self.assertEqual(ANY_PIPE.PipeAnySingle.execute().result, (None, False))

    def test_connected_round_trip_preserves_identity(self):
        value = {"samples": torch.zeros((1, 4, 2, 2))}
        routed, present = ANY_PIPE.PipeAnySingle.execute(value).result
        self.assertIs(routed, value)
        self.assertTrue(present)

    def test_falsey_values_are_present(self):
        values = [False, 0, 0.0, "", [], None, torch.zeros((1, 2, 2))]
        for value in values:
            with self.subTest(value=repr(value)):
                routed, present = ANY_PIPE.PipeAnySingle.execute(value).result
                self.assertIs(routed, value)
                self.assertTrue(present)

    def test_equivalent_to_existing_scalar_pipe_round_trip(self):
        values = [
            torch.zeros((1, 2, 2, 3)),
            torch.zeros((1, 2, 2)),
            {"samples": torch.zeros((1, 4, 2, 2))},
            object(),
            False,
            0,
            "",
            None,
        ]
        for value in values:
            with self.subTest(value=repr(value)):
                pipe = ANY_PIPE.PipeToEditAny.execute(any_1=[value]).result[0]
                reference = ANY_PIPE.PipeFromAny.execute(pipe).result[1]
                actual, present = ANY_PIPE.PipeAnySingle.execute(value).result
                self.assertIs(reference, value)
                self.assertIs(actual, value)
                self.assertTrue(present)

    def test_actual_comfy_mapping_preserves_items_order_and_presence(self):
        from execution import get_output_data

        values = [object(), object(), object()]
        output, ui, has_subgraph, pending = asyncio.run(
            get_output_data(
                "prompt",
                "pipe",
                ANY_PIPE.PipeAnySingle,
                {"any": values},
                v3_data={},
            )
        )

        self.assertEqual(len(output[0]), 3)
        self.assertTrue(all(actual is expected for actual, expected in zip(output[0], values, strict=True)))
        self.assertEqual(output[1], [True, True, True])
        self.assertEqual(ui, {})
        self.assertFalse(has_subgraph)
        self.assertFalse(pending)

    def test_backend_executor_accepts_omitted_optional_input(self):
        from execution import get_input_data, get_output_data

        input_data, missing, v3_data = get_input_data(
            {}, ANY_PIPE.PipeAnySingle, "pipe", execution_list=None
        )
        self.assertEqual(input_data, {})
        self.assertEqual(missing, {})
        output, _, _, _ = asyncio.run(
            get_output_data(
                "prompt",
                "pipe",
                ANY_PIPE.PipeAnySingle,
                input_data,
                v3_data=v3_data,
            )
        )
        self.assertEqual(output, [[None], [False]])

    def test_connected_and_disconnected_expanded_prompts_validate(self):
        import nodes
        from execution import validate_prompt

        disconnected = {
            "pipe": {"class_type": "UtilitySuitePipeAnySingle", "inputs": {}},
            "output": {
                "class_type": "PipeAnySingleTestOutput",
                "inputs": {"value": ["pipe", 1]},
            },
        }
        connected = {
            "source": {"class_type": "PipeAnySingleTestImageSource", "inputs": {}},
            "pipe": {
                "class_type": "UtilitySuitePipeAnySingle",
                "inputs": {"any": ["source", 0]},
            },
            "output": {
                "class_type": "PipeAnySingleTestOutput",
                "inputs": {"value": ["pipe", 1]},
            },
        }
        mappings = {
            "UtilitySuitePipeAnySingle": ANY_PIPE.PipeAnySingle,
            "PipeAnySingleTestImageSource": ImageSourceNode,
            "PipeAnySingleTestOutput": BooleanOutputNode,
        }
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, mappings):
            for prompt in (disconnected, connected):
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
        self.assertIn("UtilitySuitePipeAnySingle", node_ids)


if __name__ == "__main__":
    unittest.main()
