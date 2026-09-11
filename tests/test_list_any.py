import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_list_any_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
LIST_ANY = sys.modules[f"{SPEC.name}.list_any"]


class ListAnyAppendTests(unittest.TestCase):
    def test_empty_accumulator_appends_one_item(self):
        self.assertEqual(LIST_ANY.append_any(None, "a"), ["a"])
        self.assertEqual(LIST_ANY.ListAnyAppend.execute([None], ["a"]).result, (["a"],))

    def test_existing_accumulator_and_two_iterations(self):
        first = LIST_ANY.append_any(None, "a")
        second = LIST_ANY.append_any(first, "a")
        self.assertEqual(second, ["a", "a"])
        self.assertEqual(LIST_ANY.ListAnyAppend.execute(first, ["a"]).result, (["a", "a"],))

    def test_existing_order_is_preserved(self):
        self.assertEqual(LIST_ANY.append_any(["a", "b"], "c"), ["a", "b", "c"])

    def test_new_list_tensor_and_object_remain_single_items(self):
        tensor = torch.arange(6).reshape(2, 3)
        marker = object()
        nested = LIST_ANY.append_any([], [1, 2])
        with_tensor = LIST_ANY.append_any(nested, tensor)
        with_object = LIST_ANY.append_any(with_tensor, marker)
        self.assertEqual(nested, [[1, 2]])
        self.assertIs(with_tensor[1], tensor)
        self.assertIs(with_object[2], marker)
        self.assertEqual(LIST_ANY.ListAnyAppend.execute([], [[1, 2]]).result, ([[1, 2]],))

    def test_input_accumulator_is_not_mutated(self):
        accumulator = ["a", "b"]
        result = LIST_ANY.append_any(accumulator, "c")
        self.assertEqual(accumulator, ["a", "b"])
        self.assertIsNot(result, accumulator)

    def test_schema_declares_generic_comfy_list_output(self):
        schema = LIST_ANY.ListAnyAppend.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteListAnyAppend")
        self.assertTrue(schema.is_input_list)
        self.assertEqual([input_.io_type for input_ in schema.inputs], ["*", "*"])
        self.assertEqual(schema.outputs[0].io_type, "*")
        self.assertTrue(schema.outputs[0].is_output_list)


if __name__ == "__main__":
    unittest.main()
