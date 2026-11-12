from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_combine_feather_segs_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.combine_feather_segs"]
COMBINE = sys.modules[f"{SPEC.name}.combine_tiles_segs"]
INVERT = sys.modules[f"{SPEC.name}.invert_mask_segs"]

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)


def segment(crop, mask, label):
    return SEG(object(), mask, 0.75, crop, tuple(crop), label, object())


class CombineFeatherSEGSTests(unittest.TestCase):
    def test_matches_explicit_invert_combine_and_remove_source_chain(self):
        destination = (
            (4, 8),
            [segment((0, 0, 6, 4), np.full((4, 6), 0.8, dtype=np.float32), "destination")],
        )
        source = (
            (4, 8),
            [segment((2, 0, 8, 4), np.full((4, 6), 0.3, dtype=np.float32), "source")],
        )
        inverted = INVERT.invert_mask_segs(source)
        explicit = COMBINE.combine_tiles_segs({"segs0": destination, "segs1": inverted})
        expected = (explicit[0], explicit[1][: len(destination[1])])

        actual = MODULE.combine_feather_segs(destination, source)

        self.assertEqual(len(actual[1]), 1)
        np.testing.assert_array_equal(actual[1][0].cropped_mask, expected[1][0].cropped_mask)

    def test_inverted_source_owns_overlap_and_source_entry_is_removed(self):
        crop = (0, 0, 4, 3)
        destination = ((3, 4), [segment(crop, np.ones((3, 4), dtype=np.float32), "destination")])
        source = ((3, 4), [segment(crop, np.full((3, 4), 0.25, dtype=np.float32), "source")])

        result = MODULE.combine_feather_segs(destination, source)

        self.assertEqual([entry.label for entry in result[1]], ["destination"])
        np.testing.assert_allclose(result[1][0].cropped_mask, 0.25, rtol=0, atol=1e-7)

    def test_source_outside_destination_does_not_change_destination(self):
        destination_mask = np.linspace(0.1, 0.9, 12, dtype=np.float32).reshape(3, 4)
        destination = ((3, 8), [segment((0, 0, 4, 3), destination_mask, "destination")])
        source = ((3, 8), [segment((4, 0, 8, 3), np.zeros((3, 4), dtype=np.float32), "source")])
        result = MODULE.combine_feather_segs(destination, source)
        np.testing.assert_array_equal(result[1][0].cropped_mask, destination_mask)

    def test_multiple_destination_entries_preserve_order_and_non_mask_fields(self):
        destinations = [
            segment((0, 0, 4, 3), np.ones((3, 4), dtype=np.float32), "a"),
            segment((2, 0, 6, 3), np.full((3, 4), 0.5, dtype=np.float32), "b"),
        ]
        source = [segment((1, 0, 5, 3), np.full((3, 4), 0.4, dtype=np.float32), "source")]
        result = MODULE.combine_feather_segs(((3, 6), destinations), ((3, 6), source))
        self.assertEqual([entry.label for entry in result[1]], ["a", "b"])
        self.assertEqual(len(result[1]), len(destinations))
        for original, output in zip(destinations, result[1]):
            self.assertEqual(output[:1] + output[2:], original[:1] + original[2:])

    def test_empty_source_leaves_destination_unchanged(self):
        header = (3, 4)
        mask = np.full((3, 4), 0.6, dtype=np.float32)
        destination = (header, [segment((0, 0, 4, 3), mask, "destination")])
        result = MODULE.combine_feather_segs(destination, (header, []))
        self.assertIs(result[0], header)
        np.testing.assert_array_equal(result[1][0].cropped_mask, mask)

    def test_empty_destination_returns_no_source_entries(self):
        header = (3, 4)
        source = (header, [segment((0, 0, 4, 3), np.ones((3, 4), dtype=np.float32), "source")])
        result = MODULE.combine_feather_segs((header, []), source)
        self.assertIs(result[0], header)
        self.assertEqual(result[1], [])

    def test_inputs_are_not_mutated(self):
        crop = (0, 0, 4, 3)
        destination_mask = np.full((3, 4), 0.8, dtype=np.float32)
        source_mask = np.full((3, 4), 0.3, dtype=np.float32)
        destination = ((3, 4), [segment(crop, destination_mask, "destination")])
        source = ((3, 4), [segment(crop, source_mask, "source")])
        MODULE.combine_feather_segs(destination, source)
        np.testing.assert_array_equal(destination[1][0].cropped_mask, destination_mask)
        np.testing.assert_array_equal(source[1][0].cropped_mask, source_mask)

    def test_matching_source_shapes_are_required(self):
        with self.assertRaisesRegex(ValueError, "matching SEGS source shapes"):
            MODULE.combine_feather_segs(((3, 4), []), ((4, 4), []))

    def test_schema_registration_and_no_feather_widgets(self):
        schema = MODULE.CombineFeatherSEGS.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteCombineFeatherSEGS")
        self.assertEqual(schema.display_name, "Combine Feather (SEGS)")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual([item.id for item in schema.inputs], ["segs_destination", "segs_source"])
        self.assertEqual([item.io_type for item in schema.outputs], ["SEGS"])
        self.assertIn(MODULE.CombineFeatherSEGS, asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list()))


if __name__ == "__main__":
    unittest.main()
