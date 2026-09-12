import dataclasses
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_list_accumulator_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
ACCUMULATOR = sys.modules[f"{SPEC.name}.list_accumulator"]


class ListAccumulatorTests(unittest.TestCase):
    def test_absent_accumulator_appends_one_item(self):
        item = object()
        result = ACCUMULATOR.ListAccumulatorAppend.execute([item]).result[0]
        self.assertEqual(len(result.items), 1)
        self.assertIs(result.items[0], item)

    def test_empty_items_produce_empty_accumulator(self):
        result = ACCUMULATOR.ListAccumulatorAppend.execute([]).result[0]
        self.assertEqual(result.items, ())

    def test_multi_item_append_preserves_identity_and_order(self):
        first = object()
        items = [object(), object(), object()]
        previous = ACCUMULATOR.UtilityAnyList((first,))
        result = ACCUMULATOR.ListAccumulatorAppend.execute(items, [previous]).result[0]
        self.assertEqual(len(result.items), 4)
        self.assertTrue(all(a is e for a, e in zip(result.items, [first, *items], strict=True)))

    def test_previous_accumulator_is_not_mutated(self):
        previous = ACCUMULATOR.UtilityAnyList(("a",))
        result = ACCUMULATOR.append_list_items(previous, ["b"])
        self.assertEqual(previous.items, ("a",))
        self.assertEqual(result.items, ("a", "b"))
        self.assertIsNot(result, previous)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            previous.items = ()

    def test_list_valued_item_remains_one_item(self):
        bbox = [10, 20, 100, 200]
        result = ACCUMULATOR.ListAccumulatorAppend.execute([bbox]).result[0]
        self.assertEqual(len(result.items), 1)
        self.assertIs(result.items[0], bbox)

    def test_tuple_dict_and_custom_items_remain_opaque(self):
        values = [(1, 2, 3, 4), {"x": 1}, object()]
        result = ACCUMULATOR.ListAccumulatorAppend.execute(values).result[0]
        self.assertTrue(all(a is e for a, e in zip(result.items, values, strict=True)))

    def test_tensor_batch_dimensions_are_untouched(self):
        masks = [torch.zeros(1, 8, 9), torch.ones(4, 8, 9)]
        result = ACCUMULATOR.ListAccumulatorAppend.execute(masks).result[0]
        self.assertIs(result.items[0], masks[0])
        self.assertIs(result.items[1], masks[1])
        self.assertEqual([item.shape[0] for item in result.items], [1, 4])

    def test_two_iteration_accumulation_and_conversion(self):
        masks = [torch.zeros(8, 9), torch.ones(8, 9)]
        first = ACCUMULATOR.ListAccumulatorAppend.execute([masks[0]]).result[0]
        second = ACCUMULATOR.ListAccumulatorAppend.execute([masks[1]], [first]).result[0]
        output = ACCUMULATOR.ListAccumulatorToList.execute(second).result[0]
        self.assertEqual(len(output), 2)
        self.assertIs(output[0], masks[0])
        self.assertIs(output[1], masks[1])

    def test_invalid_accumulator_transport_fails(self):
        with self.assertRaisesRegex(ValueError, "exactly one.*received 2"):
            ACCUMULATOR.ListAccumulatorAppend.execute(["item"], [object(), object()])
        with self.assertRaisesRegex(TypeError, "UTILITY_ANY_LIST"):
            ACCUMULATOR.ListAccumulatorAppend.execute(["item"], [object()])

    def test_nodes_2_schema_is_whole_list_in_and_ordinary_accumulator_out(self):
        append_schema = ACCUMULATOR.ListAccumulatorAppend.define_schema()
        convert_schema = ACCUMULATOR.ListAccumulatorToList.define_schema()
        self.assertTrue(append_schema.is_input_list)
        self.assertFalse(append_schema.outputs[0].is_output_list)
        self.assertEqual(append_schema.outputs[0].io_type, "UTILITY_ANY_LIST")
        self.assertEqual(convert_schema.inputs[0].io_type, "UTILITY_ANY_LIST")
        self.assertTrue(convert_schema.outputs[0].is_output_list)


if __name__ == "__main__":
    unittest.main()
