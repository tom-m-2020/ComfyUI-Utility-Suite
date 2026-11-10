import asyncio
import importlib.util
import sys
import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_crossfade_mask_overlap_segs_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.crossfade_mask_overlap_segs"]

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


def segment(crop, label, value=1.0):
    x1, y1, x2, y2 = crop
    mask = np.full((y2 - y1, x2 - x1), value, dtype=np.float32)
    return SEG(object(), mask, 0.75, list(crop), tuple(crop), label, object())


def combined(segs):
    (height, width), entries = segs
    result = np.zeros((height, width), dtype=np.float32)
    for entry in entries:
        x1, y1, x2, y2 = entry.crop_region
        result[y1:y2, x1:x2] += entry.cropped_mask
    return result


class CrossfadeMaskOverlapSEGSTests(unittest.TestCase):
    def test_schema_and_registration(self):
        schema = MODULE.CrossfadeMaskOverlapSEGS.define_schema()

        self.assertEqual(schema.node_id, "UtilitySuiteCrossfadeMaskOverlapSEGS")
        self.assertEqual(schema.display_name, "Crossfade Mask Overlap (SEGS)")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual(
            [item.id for item in schema.inputs],
            ["segs", "feather_mode", "max_feather"],
        )
        self.assertEqual(schema.inputs[1].default, "pixel")
        self.assertEqual(schema.inputs[2].default, 0)
        nodes = asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list())
        self.assertIn(MODULE.CrossfadeMaskOverlapSEGS, nodes)

    def test_horizontal_overlap_gets_complementary_feather(self):
        source = (
            (4, 10),
            [segment((0, 0, 6, 4), "left"), segment((4, 0, 10, 4), "right")],
        )

        result = MODULE.crossfade_mask_overlap_segs(source)

        self.assertTrue(np.allclose(combined(result), 1.0))
        self.assertTrue(
            np.array_equal(
                result[1][0].cropped_mask[0],
                np.array([1, 1, 1, 1, 1, 0], dtype=np.float32),
            )
        )
        self.assertTrue(
            np.array_equal(
                result[1][1].cropped_mask[0],
                np.array([0, 1, 1, 1, 1, 1], dtype=np.float32),
            )
        )

    def test_vertical_overlap_gets_complementary_feather(self):
        source = (
            (10, 4),
            [segment((0, 0, 4, 6), "top"), segment((0, 4, 4, 10), "bottom")],
        )

        result = MODULE.crossfade_mask_overlap_segs(source)

        self.assertTrue(np.allclose(combined(result), 1.0))
        self.assertTrue(np.array_equal(result[1][0].cropped_mask[:, 0], [1, 1, 1, 1, 1, 0]))
        self.assertTrue(np.array_equal(result[1][1].cropped_mask[:, 0], [0, 1, 1, 1, 1, 1]))

    def test_two_by_two_grid_multiplies_orthogonal_corner_ramps(self):
        source = (
            (10, 10),
            [
                segment((0, 0, 6, 6), "top-left"),
                segment((4, 0, 10, 6), "top-right"),
                segment((0, 4, 6, 10), "bottom-left"),
                segment((4, 4, 10, 10), "bottom-right"),
            ],
        )

        result = MODULE.crossfade_mask_overlap_segs(source)

        self.assertTrue(np.allclose(combined(result), 1.0))
        self.assertEqual(float(result[1][0].cropped_mask[-1, -1]), 0.0)
        self.assertEqual(float(result[1][3].cropped_mask[0, 0]), 0.0)

    def test_nonoverlapping_masks_and_metadata_are_unchanged(self):
        first = segment((0, 0, 3, 4), "first", 0.5)
        second = segment((5, 0, 8, 4), "second", 0.75)

        result = MODULE.crossfade_mask_overlap_segs(((4, 8), [first, second]))

        for source, output in zip((first, second), result[1]):
            self.assertTrue(np.array_equal(output.cropped_mask, source.cropped_mask))
            self.assertEqual(output.crop_region, source.crop_region)
            self.assertEqual(output.bbox, source.bbox)
            self.assertEqual(output.label, source.label)
            self.assertIs(output.cropped_image, source.cropped_image)

    def test_multiple_neighbors_on_one_side_use_largest_overlap_once(self):
        crops = [(0, 0, 10, 8), (7, 0, 14, 4), (8, 4, 14, 8)]

        overlaps = MODULE._neighbor_overlaps(crops)

        self.assertEqual(overlaps[0].right, 3)

    def test_pixel_cap_limits_both_counterparts_to_same_amount(self):
        crops = [(0, 0, 8, 4), (4, 0, 12, 4)]

        uncapped = MODULE._neighbor_overlaps(crops)
        capped = MODULE._neighbor_overlaps(crops, 2, "pixel")

        self.assertEqual((uncapped[0].right, uncapped[1].left), (4, 4))
        self.assertEqual((capped[0].right, capped[1].left), (2, 2))

    def test_percent_cap_uses_each_crop_axis_and_smallest_counterpart_cap(self):
        crops = [(0, 0, 8, 4), (4, 0, 16, 4)]

        capped = MODULE._neighbor_overlaps(crops, 25, "percent")

        self.assertEqual((capped[0].right, capped[1].left), (2, 2))

    def test_zero_cap_is_undefined_and_matches_uncapped_behavior(self):
        crops = [(0, 0, 8, 4), (4, 0, 12, 4)]

        self.assertEqual(
            MODULE._neighbor_overlaps(crops, 0, "percent"),
            MODULE._neighbor_overlaps(crops),
        )

    def test_percent_cap_rejects_above_one_hundred_only_in_percent_mode(self):
        empty = ((4, 4), [])

        with self.assertRaisesRegex(ValueError, "cannot exceed 100"):
            MODULE.crossfade_mask_overlap_segs(empty, "percent", 101)
        self.assertEqual(
            MODULE.crossfade_mask_overlap_segs(empty, "pixel", 101),
            empty,
        )

    def test_identical_centered_crops_have_no_directional_relationship(self):
        source = (
            (4, 4),
            [segment((0, 0, 4, 4), "first"), segment((0, 0, 4, 4), "second")],
        )

        result = MODULE.crossfade_mask_overlap_segs(source)

        self.assertTrue(np.array_equal(result[1][0].cropped_mask, np.ones((4, 4))))
        self.assertTrue(np.array_equal(result[1][1].cropped_mask, np.ones((4, 4))))


if __name__ == "__main__":
    unittest.main()
