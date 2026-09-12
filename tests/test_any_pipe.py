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

    def test_compact_builder_scalar_only_and_missing_outputs(self):
        value = object()
        pipe = ANY_PIPE.PipeToEditAny.execute(any_1=[value]).result[0]
        output = ANY_PIPE.PipeFromAny.execute(pipe).result
        self.assertIs(output[0], pipe)
        self.assertIs(output[1], value)
        self.assertEqual(output[2:7], (None, None, None, None, None))
        self.assertEqual(output[7:], ([], [], []))

    def test_compact_builder_one_list_and_several_scalars(self):
        scalars = [object(), object(), object()]
        masks = [object(), object(), object()]
        pipe = ANY_PIPE.PipeToEditAny.execute(
            any_1=[scalars[0]], any_2=[scalars[1]], any_3=[scalars[2]], list_1=masks
        ).result[0]
        output = ANY_PIPE.PipeFromAny.execute(pipe).result
        self.assertTrue(all(output[index + 1] is value for index, value in enumerate(scalars)))
        self.assertTrue(all(actual is expected for actual, expected in zip(output[7], masks, strict=True)))

    def test_compact_builder_all_scalar_slots(self):
        values = [object() for _ in range(6)]
        inputs = {f"any_{index + 1}": [value] for index, value in enumerate(values)}
        pipe = ANY_PIPE.PipeToEditAny.execute(**inputs).result[0]
        output = ANY_PIPE.PipeFromAny.execute(pipe).result
        self.assertTrue(all(output[index + 1] is value for index, value in enumerate(values)))

    def test_compact_builder_independent_list_lengths(self):
        lists = [[object() for _ in range(length)] for length in (2, 3, 5)]
        pipe = ANY_PIPE.PipeToEditAny.execute(
            list_1=lists[0], list_2=lists[1], list_3=lists[2]
        ).result[0]
        output = ANY_PIPE.PipeFromAny.execute(pipe).result
        self.assertEqual([len(output[index]) for index in range(7, 10)], [2, 3, 5])
        for actual, expected in zip(output[7:], lists, strict=True):
            self.assertTrue(all(a is e for a, e in zip(actual, expected, strict=True)))

    def test_compact_builder_edit_preserves_disconnected_fields(self):
        original = ANY_PIPE.PipeToEditAny.execute(any_1=["old"], any_2=["keep"], list_1=[1, 2]).result[0]
        edited = ANY_PIPE.PipeToEditAny.execute(pipe=[original], any_1=["new"]).result[0]
        self.assertEqual(ANY_PIPE.get_scalar(original, "any_1"), "old")
        self.assertEqual(ANY_PIPE.get_scalar(edited, "any_1"), "new")
        self.assertEqual(ANY_PIPE.get_scalar(edited, "any_2"), "keep")
        self.assertEqual(ANY_PIPE.get_list(edited, "list_1"), [1, 2])

    def test_compact_builder_preserves_opaque_bbox_items(self):
        tuple_bbox = (1, 2, 3, 4)
        list_bbox = [5, 6, 7, 8]
        pipe = ANY_PIPE.PipeToEditAny.execute(list_1=[tuple_bbox, list_bbox]).result[0]
        result = ANY_PIPE.PipeFromAny.execute(pipe).result[7]
        self.assertIs(result[0], tuple_bbox)
        self.assertIs(result[1], list_bbox)

    def test_compact_builder_rejects_multi_item_scalar_slot(self):
        with self.assertRaisesRegex(ValueError, "any_4.*received 2"):
            ANY_PIPE.PipeToEditAny.execute(any_4=["a", "b"])

    def test_compact_nodes_2_schema_transport_contracts(self):
        builder = ANY_PIPE.PipeToEditAny.define_schema()
        extractor = ANY_PIPE.PipeFromAny.define_schema()
        self.assertTrue(builder.is_input_list)
        self.assertFalse(extractor.is_input_list)
        self.assertEqual([item.id for item in builder.inputs], [
            "pipe", "any_1", "any_2", "any_3", "any_4", "any_5", "any_6", "list_1", "list_2", "list_3"
        ])
        self.assertEqual([output.is_output_list for output in extractor.outputs], [
            False, False, False, False, False, False, False, True, True, True
        ])


if __name__ == "__main__":
    unittest.main()
