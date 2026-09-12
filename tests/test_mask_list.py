import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_mask_list_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MASK_LIST = sys.modules[f"{SPEC.name}.mask_list"]


class MaskFromListTests(unittest.TestCase):
    def setUp(self):
        self.masks = [torch.full((1, 3, 4), float(index)) for index in range(4)]

    def test_first_item(self):
        self.assertEqual(MASK_LIST.masks_from_list(self.masks, 0, 1), [self.masks[0]])

    def test_middle_range(self):
        self.assertEqual(MASK_LIST.masks_from_list(self.masks, 1, 2), self.masks[1:3])

    def test_final_item_and_through_end(self):
        self.assertEqual(MASK_LIST.masks_from_list(self.masks, 3, 1), [self.masks[3]])
        self.assertEqual(MASK_LIST.masks_from_list(self.masks, 2, 2), self.masks[2:])

    def test_overlong_range_truncates_at_end(self):
        self.assertEqual(MASK_LIST.masks_from_list(self.masks, 2, 99), self.masks[2:])

    def test_start_past_end_clamps_to_final_item(self):
        self.assertEqual(MASK_LIST.masks_from_list(self.masks, 99, 1), [self.masks[3]])

    def test_length_boundary_matches_audited_runtime(self):
        self.assertEqual(MASK_LIST.masks_from_list(self.masks, 0, 0), [])
        self.assertEqual(MASK_LIST.masks_from_list(self.masks, 0, -1), self.masks[:-1])
        self.assertEqual(MASK_LIST.masks_from_list(self.masks, -1, 1), [])

    def test_selected_items_preserve_identity_and_input_is_not_mutated(self):
        original = list(self.masks)
        result = MASK_LIST.masks_from_list(self.masks, 1, 2)
        self.assertEqual(self.masks, original)
        self.assertIsNot(result, self.masks)
        self.assertIs(result[0], self.masks[1])
        self.assertIs(result[1], self.masks[2])

    def test_batched_mask_tensor_remains_one_list_item(self):
        batched = torch.rand((5, 3, 4))
        result = MASK_LIST.masks_from_list([batched, self.masks[0]], 0, 1)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], batched)
        self.assertEqual(result[0].shape, (5, 3, 4))

    def test_empty_list_returns_empty(self):
        self.assertEqual(MASK_LIST.masks_from_list([], 0, 1), [])

    def test_node_execution_and_schema_list_contract(self):
        result = MASK_LIST.MaskFromList.execute(self.masks, [1], [2]).result
        self.assertEqual(result, (self.masks[1:3],))
        schema = MASK_LIST.MaskFromList.define_schema()
        self.assertTrue(schema.is_input_list)
        self.assertEqual([input_.io_type for input_ in schema.inputs], ["MASK", "INT", "INT"])
        self.assertEqual((schema.inputs[1].default, schema.inputs[1].min), (0, 0))
        self.assertEqual((schema.inputs[2].default, schema.inputs[2].min), (1, 1))
        self.assertEqual(schema.outputs[0].io_type, "MASK")
        self.assertTrue(schema.outputs[0].is_output_list)


if __name__ == "__main__":
    unittest.main()
