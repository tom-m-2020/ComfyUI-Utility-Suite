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
    "utility_suite_segs_insert_by_index_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.segs_insert_by_index"]
FILTER = sys.modules[f"{SPEC.name}.segs_range_filter_plus"]

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)


def segment(label: str, confidence: float = 0.5) -> SEG:
    return SEG(
        object(),
        object(),
        confidence,
        (1, 2, 7, 9),
        (2, 3, 6, 8),
        label,
        object(),
    )


class SEGSInsertByIndexTests(unittest.TestCase):
    def setUp(self):
        self.header = (100, 200)
        self.a, self.b, self.c, self.d, self.e = [segment(label) for label in "ABCDE"]

    def insert(self, destination, source, indices, destination_header=None, source_header=None):
        return MODULE.insert_segs_by_index(
            (destination_header or self.header, destination),
            (source_header or self.header, source),
            indices,
        )

    def test_standard_noncontiguous_reconstruction(self):
        result = self.insert([self.a, self.c, self.e], [self.b, self.d], [1, 3])
        self.assertEqual(result[1], [self.a, self.b, self.c, self.d, self.e])

    def test_unsorted_indices_keep_source_correspondence(self):
        result = self.insert([self.a, self.c, self.e], [self.d, self.b], [3, 1])
        self.assertEqual(result[1], [self.a, self.b, self.c, self.d, self.e])

    def test_insertion_at_beginning_and_end(self):
        result = self.insert([self.b, self.c, self.d], [self.a, self.e], [0, 4])
        self.assertEqual(result[1], [self.a, self.b, self.c, self.d, self.e])

    def test_empty_combinations(self):
        self.assertEqual(self.insert([], [self.a, self.b], [0, 1])[1], [self.a, self.b])
        destination = [self.a, self.b]
        destination_only = self.insert(destination, [], [])
        self.assertEqual(destination_only[1], [self.a, self.b])
        self.assertIsNot(destination_only[1], destination)
        self.assertEqual(self.insert([], [], []), (self.header, []))

    def test_invalid_indices_fail_clearly(self):
        invalid_cases = [
            ([self.b, self.d], [1, 1], ValueError, "duplicate"),
            ([self.b], [-1], ValueError, "outside"),
            ([self.b], [4], ValueError, "outside"),
            ([self.b], [1.0], TypeError, "integer"),
            ([self.b], [True], TypeError, "integer"),
        ]
        for source, indices, error_type, message in invalid_cases:
            with self.subTest(indices=indices), self.assertRaisesRegex(error_type, message):
                self.insert([self.a, self.c, self.e], source, indices)
        with self.assertRaisesRegex(TypeError, "list"):
            self.insert([self.a], [self.b], (1,))

    def test_source_and_index_counts_must_match(self):
        with self.assertRaisesRegex(ValueError, "2 source entries but 1 indices"):
            self.insert([self.a], [self.b, self.c], [1])

    def test_source_size_headers_must_match(self):
        with self.assertRaisesRegex(ValueError, "matching source-size headers"):
            self.insert([self.a], [self.b], [1], source_header=(101, 200))

    def test_identity_metadata_and_inputs_are_preserved(self):
        destination = [self.a, self.c, self.e]
        source = [self.d, self.b]
        destination_before = list(destination)
        source_before = list(source)
        indices = [3, 1]
        indices_before = list(indices)

        header, result = self.insert(destination, source, indices)

        self.assertIs(header, self.header)
        self.assertEqual(destination, destination_before)
        self.assertEqual(source, source_before)
        self.assertEqual(indices, indices_before)
        self.assertIsNot(result, destination)
        for actual, expected in zip(result, [self.a, self.b, self.c, self.d, self.e]):
            self.assertIs(actual, expected)

    def test_filter_modify_insert_round_trip_restores_original_order(self):
        original = [
            segment("A", 0.1),
            segment("B", 0.5),
            segment("C", 0.9),
            segment("D", 0.3),
        ]
        filtered, remained, indices = FILTER.filter_segs_range_plus(
            (self.header, original), "confidence(0-100)", True, 20, 80
        )
        modified = [filtered[1][0]._replace(label="B modified"), filtered[1][1]._replace(label="D modified")]

        result = MODULE.insert_segs_by_index(remained, (self.header, modified), indices)

        self.assertEqual([entry.label for entry in result[1]], ["A", "B modified", "C", "D modified"])
        self.assertIs(result[1][0], original[0])
        self.assertIs(result[1][1], modified[0])
        self.assertIs(result[1][2], original[2])
        self.assertIs(result[1][3], modified[1])

    def test_manual_indices_without_connected_input(self):
        indices = MODULE.resolve_indices(" 3, 1 ", None)
        result = self.insert([self.a, self.c, self.e], [self.d, self.b], indices)
        self.assertEqual(indices, [3, 1])
        self.assertEqual(result[1], [self.a, self.b, self.c, self.d, self.e])

    def test_empty_or_whitespace_manual_value_uses_connected_input(self):
        connected = [1, 3]
        self.assertIs(MODULE.resolve_indices("", connected), connected)
        self.assertIs(MODULE.resolve_indices("   ", connected), connected)

    def test_nonempty_manual_value_overrides_connected_input(self):
        self.assertEqual(MODULE.resolve_indices("1, 3", [0, 2]), [1, 3])

    def test_invalid_manual_value_never_falls_back_to_connected_input(self):
        invalid = {
            "1,x,5": "integers",
            "1,,5": "empty element",
            ",1,5": "empty element",
            "1,5,": "empty element",
        }
        for text, message in invalid.items():
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                MODULE.resolve_indices(text, [0, 2, 4])

    def test_manual_indices_use_existing_range_count_and_duplicate_validation(self):
        cases = [
            ("-1", [self.a, self.c], [self.b], ValueError, "outside"),
            ("1,1", [self.a, self.c], [self.b, self.d], ValueError, "duplicate"),
            ("3", [self.a, self.c], [self.b], ValueError, "outside"),
            ("1", [self.a, self.c], [self.b, self.d], ValueError, "2 source entries but 1 indices"),
        ]
        for text, destination, source, error_type, message in cases:
            with self.subTest(text=text), self.assertRaisesRegex(error_type, message):
                indices = MODULE.resolve_indices(text, [0, 2])
                self.insert(destination, source, indices)

    def test_missing_index_sources_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "SEGS_INDICES input or manually entered indices"):
            MODULE.resolve_indices("   ", None)

    def test_execute_supports_manual_override_and_legacy_socket(self):
        destination = (self.header, [self.a, self.c, self.e])
        source = (self.header, [self.b, self.d])
        legacy = MODULE.SEGSInsertByIndex.execute(destination, source, index=[1, 3], index_text="").result[0]
        manual = MODULE.SEGSInsertByIndex.execute(destination, source, index=None, index_text="1,3").result[0]
        override = MODULE.SEGSInsertByIndex.execute(destination, source, index=[0, 2], index_text="1,3").result[0]
        expected = [self.a, self.b, self.c, self.d, self.e]
        self.assertEqual(legacy[1], expected)
        self.assertEqual(manual[1], expected)
        self.assertEqual(override[1], expected)

    def test_schema_uses_single_value_segs_indices_transport(self):
        schema = MODULE.SEGSInsertByIndex.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteSEGSInsertByIndex")
        self.assertEqual(schema.display_name, "SEGS Insert by Index")
        self.assertEqual([entry.io_type for entry in schema.inputs], ["SEGS", "SEGS", "SEGS_INDICES", "STRING"])
        self.assertEqual([entry.id for entry in schema.inputs], ["segs_destination", "segs_source", "index", "index_text"])
        self.assertEqual(schema.inputs[2].display_name, "index")
        self.assertTrue(schema.inputs[2].optional)
        self.assertEqual(schema.inputs[3].display_name, "index")
        self.assertEqual(schema.inputs[3].default, "")
        self.assertFalse(schema.inputs[3].multiline)
        self.assertFalse(schema.is_input_list)
        self.assertEqual([output.io_type for output in schema.outputs], ["SEGS"])
        self.assertFalse(schema.outputs[0].is_output_list)


if __name__ == "__main__":
    unittest.main()
