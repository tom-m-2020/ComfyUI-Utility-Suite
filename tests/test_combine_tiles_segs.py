import asyncio
import importlib.util
import sys
import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_combine_tiles_segs_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.combine_tiles_segs"]

SEG = namedtuple(
    "SEG",
    [
        "cropped_image",
        "cropped_mask",
        "confidence",
        "crop_region",
        "bbox",
        "label",
        "control_net_wrapper",
    ],
)


def segment(crop, mask, label):
    return SEG(object(), mask, 0.75, list(crop), tuple(crop), label, object())


def envelope(segs):
    (height, width), entries = segs
    result = np.zeros((height, width), dtype=np.float32)
    for entry in entries:
        x1, y1, x2, y2 = entry.crop_region
        np.maximum(result[y1:y2, x1:x2], entry.cropped_mask, out=result[y1:y2, x1:x2])
    return result


class CombineTilesSEGSTests(unittest.TestCase):
    def test_schema_registration_and_paired_autogrow_contract(self):
        schema = MODULE.CombineTilesSEGS.define_schema()

        self.assertEqual(schema.node_id, "UtilitySuiteCombineTilesSEGS")
        self.assertEqual(schema.display_name, "Combine Tiles (SEGS)")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual([item.id for item in schema.inputs], ["segs_inputs", "feather_inputs"])
        self.assertEqual(schema.inputs[0].template.prefix, "segs")
        self.assertEqual(schema.inputs[0].template.min, 1)
        self.assertEqual(schema.inputs[0].template.get_all()[0].id, "segs0")
        self.assertEqual(schema.inputs[1].template.prefix, "feather")
        self.assertTrue(schema.inputs[1].optional)
        self.assertEqual(schema.inputs[1].template.get_all()[0].id, "feather0")
        nodes = asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list())
        self.assertIn(MODULE.CombineTilesSEGS, nodes)

    def test_single_group_receives_inner_feather_even_without_overlap(self):
        mask = np.ones((7, 7), dtype=np.float32)
        source = ((9, 9), [segment((1, 1, 8, 8), mask, "only")])

        result = MODULE.combine_tiles_segs({"segs0": source}, {"feather0": 3})
        output = result[1][0].cropped_mask

        self.assertAlmostEqual(float(output[0, 3]), 1 / 3)
        self.assertAlmostEqual(float(output[1, 3]), 2 / 3)
        self.assertEqual(float(output[2, 3]), 1.0)
        self.assertEqual(result[1][0].crop_region, source[1][0].crop_region)
        self.assertEqual(result[1][0].label, "only")

    def test_two_solid_groups_form_complementary_crossfade_without_deficit(self):
        first = ((7, 9), [segment((0, 0, 9, 7), np.ones((7, 9), dtype=np.float32), "first")])
        second = ((7, 9), [segment((2, 0, 7, 7), np.ones((7, 5), dtype=np.float32), "second")])

        result = MODULE.combine_tiles_segs(
            {"segs0": first, "segs1": second},
            {"feather0": 0, "feather1": 3},
        )
        first_out, second_out = result[1]
        combined = np.zeros((7, 9), dtype=np.float32)
        combined += first_out.cropped_mask
        combined[:, 2:7] += second_out.cropped_mask

        self.assertTrue(np.allclose(combined, 1.0))
        self.assertTrue(np.allclose(first_out.cropped_mask[:, 2:7] + second_out.cropped_mask, 1.0))

    def test_internal_group_relationship_is_scaled_uniformly(self):
        left = np.ones((5, 6), dtype=np.float32)
        right = np.full((5, 6), 0.5, dtype=np.float32)
        group0 = ((5, 10), [segment((0, 0, 6, 5), left, "left"), segment((4, 0, 10, 5), right, "right")])
        group1 = ((5, 10), [segment((3, 0, 7, 5), np.ones((5, 4), dtype=np.float32), "later")])

        result = MODULE.combine_tiles_segs(
            {"segs0": group0, "segs1": group1},
            {"feather1": 2},
        )
        left_out, right_out, _ = result[1]

        self.assertTrue(np.allclose(right_out.cropped_mask[:, :2], left_out.cropped_mask[:, 4:6] * 0.5))

    def test_three_groups_preserve_the_strongest_feathered_coverage(self):
        groups = {}
        feathers = {}
        for index, crop in enumerate(((0, 0, 9, 9), (1, 1, 8, 8), (2, 2, 7, 7))):
            x1, y1, x2, y2 = crop
            groups[f"segs{index}"] = (
                (9, 9),
                [segment(crop, np.ones((y2 - y1, x2 - x1), dtype=np.float32), str(index))],
            )
            feathers[f"feather{index}"] = 3

        result = MODULE.combine_tiles_segs(groups, feathers)
        actual = np.zeros((9, 9), dtype=np.float32)
        for entry in result[1]:
            x1, y1, x2, y2 = entry.crop_region
            actual[y1:y2, x1:x2] += entry.cropped_mask
        expected_canvas = np.zeros((9, 9), dtype=np.float32)
        for index in range(3):
            np.maximum(expected_canvas, envelope(groups[f"segs{index}"]) * MODULE._inner_feather(envelope(groups[f"segs{index}"]), 3), out=expected_canvas)

        self.assertTrue(np.allclose(actual, expected_canvas))

    def test_requires_segs0_matching_headers_and_matching_feather_slots(self):
        empty = ((5, 5), [])
        with self.assertRaisesRegex(ValueError, "requires segs0"):
            MODULE.combine_tiles_segs({"segs1": empty})
        with self.assertRaisesRegex(ValueError, "matching SEGS source shapes"):
            MODULE.combine_tiles_segs({"segs0": empty, "segs1": ((6, 5), [])})
        with self.assertRaisesRegex(ValueError, "without matching SEGS"):
            MODULE.combine_tiles_segs({"segs0": empty}, {"feather1": 2})


if __name__ == "__main__":
    unittest.main()
