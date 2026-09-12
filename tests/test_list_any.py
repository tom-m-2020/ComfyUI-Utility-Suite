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
    def test_empty_accumulator_plus_scalar_transport(self):
        self.assertEqual(LIST_ANY.extend_comfy_items([None], ["a"]), ["a"])
        self.assertEqual(LIST_ANY.ListAnyAppend.execute([None], ["a"]).result, (["a"],))

    def test_existing_list_plus_scalar_transport(self):
        self.assertEqual(LIST_ANY.extend_comfy_items(["a"], ["b"]), ["a", "b"])

    def test_one_item_comfy_list_extends_flat(self):
        self.assertEqual(LIST_ANY.extend_comfy_items(["a"], ["b"]), ["a", "b"])

    def test_multi_item_comfy_list_extends_flat(self):
        self.assertEqual(LIST_ANY.extend_comfy_items(["a"], ["b", "c"]), ["a", "b", "c"])

    def test_list_valued_item_is_preserved_from_transport_container(self):
        item = [1, 2, 3, 4]
        result = LIST_ANY.extend_comfy_items([], [item])
        self.assertEqual(result, [[1, 2, 3, 4]])
        self.assertIs(result[0], item)

    def test_bbox_items_are_not_flattened(self):
        bbox0 = (10, 20, 100, 200)
        bbox1 = (30, 40, 120, 220)
        result = LIST_ANY.extend_comfy_items([], [bbox0, bbox1])
        self.assertEqual(result, [bbox0, bbox1])
        self.assertIs(result[0], bbox0)
        self.assertIs(result[1], bbox1)

    def test_tensor_objects_and_batch_dimensions_are_untouched(self):
        tensor_a = torch.zeros(1, 8, 9)
        tensor_b = torch.ones(4, 8, 9)
        result = LIST_ANY.extend_comfy_items([], [tensor_a, tensor_b])
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], tensor_a)
        self.assertIs(result[1], tensor_b)
        self.assertEqual(result[0].shape, (1, 8, 9))
        self.assertEqual(result[1].shape, (4, 8, 9))

    def test_input_accumulator_is_not_mutated(self):
        accumulator = ["a", "b"]
        result = LIST_ANY.extend_comfy_items(accumulator, ["c", "d"])
        self.assertEqual(accumulator, ["a", "b"])
        self.assertEqual(result, ["a", "b", "c", "d"])
        self.assertIsNot(result, accumulator)

    def test_repeated_loop_style_accumulation_stays_flat_and_ordered(self):
        accumulator = [None]
        for incoming in (["a"], ["b", "c"], ["d"]):
            accumulator = LIST_ANY.extend_comfy_items(accumulator, incoming)
        self.assertEqual(accumulator, ["a", "b", "c", "d"])

    def test_schema_declares_generic_comfy_list_transport(self):
        schema = LIST_ANY.ListAnyAppend.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteListAnyAppend")
        self.assertTrue(schema.is_input_list)
        self.assertEqual([input_.io_type for input_ in schema.inputs], ["*", "*"])
        self.assertEqual(schema.outputs[0].io_type, "*")
        self.assertTrue(schema.outputs[0].is_output_list)


if __name__ == "__main__":
    unittest.main()
