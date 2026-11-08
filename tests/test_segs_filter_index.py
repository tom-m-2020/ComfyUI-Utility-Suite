from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import namedtuple
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_segs_filter_index_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.segs_filter_index"]
INSERT = sys.modules[f"{SPEC.name}.segs_insert_by_index"]

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)


def segment(label: str) -> SEG:
    return SEG(object(), object(), 0.5, (1, 2, 7, 9), (2, 3, 6, 8), label, object())


class SEGSFilterIndexTests(unittest.TestCase):
    def setUp(self):
        self.header = (100, 200)
        self.entries = [segment(label) for label in "ABCDE"]
        self.segs = (self.header, self.entries)

    def execute(self, index=None, index_text=""):
        return MODULE.SEGSFilterIndex.execute(self.segs, index=index, index_text=index_text).result

    def test_connected_indices_partition_segs(self):
        filtered, remained, indices = self.execute(index=[1, 3])
        self.assertEqual(indices, [1, 3])
        self.assertEqual(filtered[1], [self.entries[1], self.entries[3]])
        self.assertEqual(remained[1], [self.entries[0], self.entries[2], self.entries[4]])
        self.assertIs(filtered[0], self.header)
        self.assertIs(remained[0], self.header)

    def test_unsorted_indices_preserve_pairing_and_round_trip(self):
        filtered, remained, indices = self.execute(index=[3, 1])
        self.assertEqual(filtered[1], [self.entries[3], self.entries[1]])
        self.assertEqual(indices, [3, 1])
        reconstructed = INSERT.insert_segs_by_index(remained, filtered, indices)
        self.assertTrue(
            all(actual is expected for actual, expected in zip(reconstructed[1], self.entries, strict=True))
        )

    def test_nonempty_manual_indices_override_connected_input(self):
        filtered, remained, indices = self.execute(index=[0, 2], index_text=" 3, 1 ")
        self.assertEqual(filtered[1], [self.entries[3], self.entries[1]])
        self.assertEqual(remained[1], [self.entries[0], self.entries[2], self.entries[4]])
        self.assertEqual(indices, [3, 1])

    def test_empty_or_whitespace_manual_value_uses_connected_input(self):
        for text in ("", "   "):
            with self.subTest(text=text):
                filtered, _, indices = self.execute(index=[2], index_text=text)
                self.assertIs(filtered[1][0], self.entries[2])
                self.assertEqual(indices, [2])

    def test_missing_both_index_sources_fails(self):
        with self.assertRaisesRegex(ValueError, "SEGS_INDICES input or manually entered indices"):
            self.execute()

    def test_invalid_manual_syntax_never_falls_back(self):
        for text, message in (
            ("1,x", "integers"),
            ("1,,3", "empty element"),
            (",1", "empty element"),
            ("1,", "empty element"),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                self.execute(index=[0], index_text=text)

    def test_invalid_manual_values_never_fall_back(self):
        for text, message in (("-1", "outside"), ("5", "outside"), ("1,1", "duplicate")):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                self.execute(index=[0], index_text=text)

    def test_connected_indices_are_validated(self):
        cases = (
            ([-1], ValueError, "outside"),
            ([5], ValueError, "outside"),
            ([1, 1], ValueError, "duplicate"),
            ([True], TypeError, "integer"),
            ([1.0], TypeError, "integer"),
            ((1,), TypeError, "list"),
        )
        for index, error_type, message in cases:
            with self.subTest(index=index), self.assertRaisesRegex(error_type, message):
                self.execute(index=index)

    def test_empty_index_selects_none_without_mutating_inputs(self):
        entries_before = list(self.entries)
        index = []
        filtered, remained, output_index = self.execute(index=index)
        self.assertEqual(filtered, (self.header, []))
        self.assertEqual(remained[1], self.entries)
        self.assertEqual(output_index, [])
        self.assertIsNot(output_index, index)
        self.assertTrue(all(actual is expected for actual, expected in zip(self.entries, entries_before, strict=True)))

    def test_empty_segs_accepts_only_empty_index(self):
        empty = (self.header, [])
        filtered, remained, indices = MODULE.SEGSFilterIndex.execute(empty, index=[], index_text="").result
        self.assertEqual(filtered, (self.header, []))
        self.assertEqual(remained, (self.header, []))
        self.assertEqual(indices, [])
        with self.assertRaisesRegex(ValueError, "outside"):
            MODULE.SEGSFilterIndex.execute(empty, index=None, index_text="0")

    def test_schema_uses_single_value_index_transport(self):
        schema = MODULE.SEGSFilterIndex.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteSEGSFilterIndex")
        self.assertEqual(schema.display_name, "SEGS Filter (index)")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual([entry.io_type for entry in schema.inputs], ["SEGS", "SEGS_INDICES", "STRING"])
        self.assertEqual([entry.id for entry in schema.inputs], ["segs", "index", "index_text"])
        self.assertTrue(schema.inputs[1].optional)
        self.assertEqual(schema.inputs[1].display_name, "index")
        self.assertEqual(schema.inputs[2].display_name, "index")
        self.assertEqual(schema.inputs[2].default, "")
        self.assertFalse(schema.inputs[2].multiline)
        self.assertEqual(
            [output.io_type for output in schema.outputs],
            ["SEGS", "SEGS", "SEGS_INDICES"],
        )
        self.assertFalse(schema.outputs[2].is_output_list)
        self.assertFalse(schema.is_input_list)


if __name__ == "__main__":
    unittest.main()
