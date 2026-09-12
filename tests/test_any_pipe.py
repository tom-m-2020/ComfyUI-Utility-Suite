import dataclasses
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_any_pipe_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
ANY_PIPE = sys.modules[f"{SPEC.name}.any_pipe"]


class UtilityAnyPipeTests(unittest.TestCase):
    def test_scalar_only_pipe(self):
        value = object()
        pipe = ANY_PIPE.set_scalar(ANY_PIPE.UtilityAnyPipe(), "value", value)
        self.assertIs(ANY_PIPE.get_scalar(pipe, "value"), value)
        self.assertEqual(pipe.get("value").kind, "scalar")

    def test_list_field_does_not_affect_unrelated_scalars(self):
        image = object()
        pipe = ANY_PIPE.set_list(ANY_PIPE.UtilityAnyPipe(), "masks", ["m0", "m1", "m2"])
        pipe = ANY_PIPE.set_scalar(pipe, "image", image)
        self.assertIs(ANY_PIPE.get_scalar(pipe, "image"), image)
        self.assertEqual(ANY_PIPE.get_list(pipe, "masks"), ["m0", "m1", "m2"])

    def test_two_list_fields_keep_different_lengths(self):
        pipe = ANY_PIPE.set_list(ANY_PIPE.UtilityAnyPipe(), "short", [1, 2])
        pipe = ANY_PIPE.set_list(pipe, "long", [3, 4, 5])
        self.assertEqual(ANY_PIPE.get_list(pipe, "short"), [1, 2])
        self.assertEqual(ANY_PIPE.get_list(pipe, "long"), [3, 4, 5])

    def test_list_getter_reproduces_identity_and_order(self):
        items = [object(), object(), object()]
        pipe = ANY_PIPE.set_list(ANY_PIPE.UtilityAnyPipe(), "items", items)
        result = ANY_PIPE.get_list(pipe, "items")
        self.assertEqual(len(result), 3)
        self.assertTrue(all(actual is expected for actual, expected in zip(result, items, strict=True)))

    def test_bbox_tuple_and_list_items_stay_intact(self):
        tuple_bbox = (10, 20, 100, 200)
        list_bbox = [30, 40, 120, 220]
        pipe = ANY_PIPE.set_list(ANY_PIPE.UtilityAnyPipe(), "bbox", [tuple_bbox, list_bbox])
        result = ANY_PIPE.get_list(pipe, "bbox")
        self.assertIs(result[0], tuple_bbox)
        self.assertIs(result[1], list_bbox)

    def test_tensors_stay_intact_regardless_of_batch_dimension(self):
        first = torch.zeros(1, 8, 9)
        second = torch.ones(4, 8, 9)
        pipe = ANY_PIPE.set_list(ANY_PIPE.UtilityAnyPipe(), "masks", [first, second])
        result = ANY_PIPE.get_list(pipe, "masks")
        self.assertIs(result[0], first)
        self.assertIs(result[1], second)
        self.assertEqual([item.shape[0] for item in result], [1, 4])

    def test_wrong_getter_kind_fails(self):
        pipe = ANY_PIPE.set_scalar(ANY_PIPE.UtilityAnyPipe(), "scalar", 1)
        pipe = ANY_PIPE.set_list(pipe, "items", [1])
        with self.assertRaisesRegex(TypeError, "Pipe Get List"):
            ANY_PIPE.get_scalar(pipe, "items")
        with self.assertRaisesRegex(TypeError, "Pipe Get Any"):
            ANY_PIPE.get_list(pipe, "scalar")

    def test_missing_key_fails(self):
        with self.assertRaisesRegex(KeyError, "missing"):
            ANY_PIPE.get_scalar(ANY_PIPE.UtilityAnyPipe(), "missing")

    def test_overwrite_does_not_mutate_prior_pipe(self):
        original = ANY_PIPE.set_scalar(ANY_PIPE.UtilityAnyPipe(), "value", "old")
        edited = ANY_PIPE.set_scalar(original, "value", "new")
        self.assertEqual(ANY_PIPE.get_scalar(original, "value"), "old")
        self.assertEqual(ANY_PIPE.get_scalar(edited, "value"), "new")
        self.assertIsNot(edited, original)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            original.entries = ()

    def test_sequential_sets_preserve_earlier_fields(self):
        pipe = ANY_PIPE.set_scalar(ANY_PIPE.UtilityAnyPipe(), "a", 1)
        pipe = ANY_PIPE.set_list(pipe, "b", [2, 3])
        pipe = ANY_PIPE.set_scalar(pipe, "c", 4)
        self.assertEqual([entry.key for entry in pipe.entries], ["a", "b", "c"])
        self.assertEqual(ANY_PIPE.get_scalar(pipe, "a"), 1)
        self.assertEqual(ANY_PIPE.get_list(pipe, "b"), [2, 3])
        self.assertEqual(ANY_PIPE.get_scalar(pipe, "c"), 4)

    def test_scalar_setter_requires_exactly_one_transport_item(self):
        with self.assertRaisesRegex(ValueError, "received 0"):
            ANY_PIPE.PipeSetAny.execute(["key"], [])
        with self.assertRaisesRegex(ValueError, "received 2"):
            ANY_PIPE.PipeSetAny.execute(["key"], ["a", "b"])

    def test_explicit_list_setter_preserves_singleton_list_kind(self):
        pipe = ANY_PIPE.PipeSetList.execute(["items"], ["only"]).result[0]
        self.assertEqual(pipe.get("items").kind, "list")
        self.assertEqual(ANY_PIPE.PipeGetList.execute(pipe, "items").result, (["only"],))

    def test_nodes_2_schema_transport_contracts(self):
        set_any = ANY_PIPE.PipeSetAny.define_schema()
        set_list = ANY_PIPE.PipeSetList.define_schema()
        get_any = ANY_PIPE.PipeGetAny.define_schema()
        get_list = ANY_PIPE.PipeGetList.define_schema()
        self.assertTrue(set_any.is_input_list)
        self.assertTrue(set_list.is_input_list)
        self.assertFalse(get_any.outputs[0].is_output_list)
        self.assertTrue(get_list.outputs[0].is_output_list)
        self.assertEqual(set_any.outputs[0].io_type, "UTILITY_ANY_PIPE")
        self.assertEqual(get_list.inputs[0].io_type, "UTILITY_ANY_PIPE")


if __name__ == "__main__":
    unittest.main()
