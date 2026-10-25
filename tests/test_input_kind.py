import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_input_kind_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
INPUT_KIND = sys.modules[f"{SPEC.name}.input_kind"]


class FakeDynamicPrompt:
    def __init__(self, nodes):
        self.nodes = nodes

    def get_node(self, node_id):
        return self.nodes[node_id]


class ListSource:
    RETURN_TYPES = ("*",)
    OUTPUT_IS_LIST = (True,)


class BatchSource:
    RETURN_TYPES = ("MASK",)
    OUTPUT_IS_LIST = (False,)


class ListBatchInspectorTests(unittest.TestCase):
    def test_multi_item_and_empty_comfy_lists(self):
        self.assertEqual(INPUT_KIND.classify_list_or_batch([]), ("list", 0))
        self.assertEqual(INPUT_KIND.classify_list_or_batch(["a", "b", "c"]), ("list", 3))

    def test_single_python_list_payload(self):
        self.assertEqual(INPUT_KIND.classify_list_or_batch([[1, 2, 3, 4]]), ("list", 4))

    def test_image_and_mask_batches(self):
        self.assertEqual(INPUT_KIND.classify_list_or_batch([torch.zeros(0, 8, 9)]), ("batch", 0))
        self.assertEqual(INPUT_KIND.classify_list_or_batch([torch.zeros(3, 8, 9, 3)]), ("batch", 3))
        self.assertEqual(INPUT_KIND.classify_list_or_batch([torch.zeros(5, 8, 9)]), ("batch", 5))
        self.assertEqual(INPUT_KIND.classify_list_or_batch([torch.zeros(1, 8, 9, 3)]), ("batch", 1))

    def test_declared_list_handles_empty_none_and_singleton(self):
        self.assertEqual(INPUT_KIND.classify_list_or_batch([], "list"), ("list", 0))
        self.assertEqual(INPUT_KIND.classify_list_or_batch([None], "list"), ("list", 0))
        self.assertEqual(INPUT_KIND.classify_list_or_batch([42], "list"), ("list", 1))

    def test_declared_batch_handles_empty_none_and_singleton(self):
        self.assertEqual(INPUT_KIND.classify_list_or_batch([], "batch"), ("batch", 0))
        self.assertEqual(INPUT_KIND.classify_list_or_batch([None], "batch"), ("batch", 0))
        self.assertEqual(
            INPUT_KIND.classify_list_or_batch([torch.zeros(1, 8, 9)], "batch"),
            ("batch", 1),
        )

    def test_connected_source_output_declaration_resolves_singleton_ambiguity(self):
        import nodes

        dynprompt = FakeDynamicPrompt(
            {
                "inspector": {"inputs": {"value": ["source", 0]}},
                "source": {"class_type": "TestSource", "inputs": {}},
            }
        )
        hidden = SimpleNamespace(dynprompt=dynprompt, unique_id="inspector")
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, {"TestSource": ListSource}):
            self.assertEqual(INPUT_KIND._connected_output_kind(hidden), "list")
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, {"TestSource": BatchSource}):
            self.assertEqual(INPUT_KIND._connected_output_kind(hidden), "batch")

    def test_executor_reports_zero_and_one_for_declared_lists_and_batches(self):
        import nodes
        from execution import get_output_data

        dynprompt = FakeDynamicPrompt(
            {
                "inspector": {"inputs": {"value": ["source", 0]}},
                "source": {"class_type": "TestSource", "inputs": {}},
            }
        )
        hidden_inputs = {
            INPUT_KIND.io.Hidden.dynprompt: dynprompt,
            INPUT_KIND.io.Hidden.unique_id: "inspector",
        }

        def execute(values, source_class):
            with patch.dict(nodes.NODE_CLASS_MAPPINGS, {"TestSource": source_class}):
                output, _, _, _ = asyncio.run(
                    get_output_data(
                        "prompt",
                        "inspector",
                        INPUT_KIND.ListBatchInspector,
                        {"value": values},
                        v3_data={"hidden_inputs": hidden_inputs},
                    )
                )
            return output[0][0], output[1][0]

        self.assertEqual(execute([], ListSource), ("list", 0))
        self.assertEqual(execute([None], ListSource), ("list", 0))
        self.assertEqual(execute([42], ListSource), ("list", 1))
        self.assertEqual(execute([], BatchSource), ("batch", 0))
        self.assertEqual(execute([None], BatchSource), ("batch", 0))
        self.assertEqual(execute([torch.zeros(0, 8, 9)], BatchSource), ("batch", 0))
        self.assertEqual(execute([torch.zeros(1, 8, 9)], BatchSource), ("batch", 1))

    def test_unknown_values(self):
        for value in (42, "text", {"samples": torch.zeros(2, 4, 8, 8)}, torch.zeros(8, 9)):
            with self.subTest(value_type=type(value).__name__):
                self.assertEqual(INPUT_KIND.classify_list_or_batch([value]), ("unknown", 0))

    def test_schema_and_execution(self):
        schema = INPUT_KIND.ListBatchInspector.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteListBatchInspector")
        self.assertTrue(schema.is_input_list)
        self.assertEqual(schema.hidden, [INPUT_KIND.io.Hidden.dynprompt, INPUT_KIND.io.Hidden.unique_id])
        self.assertEqual(schema.inputs[0].io_type, "*")
        self.assertEqual([output.io_type for output in schema.outputs], ["STRING", "INT"])
        self.assertEqual(INPUT_KIND.ListBatchInspector.execute([torch.zeros(2, 4, 5, 3)]).result, ("batch", 2))


if __name__ == "__main__":
    unittest.main()
