import asyncio
import importlib.util
import sys
import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_feather_group_masks_segs_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.feather_group_masks_segs"]

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


class FeatherGroupOfMasksSEGSTests(unittest.TestCase):
    def test_schema_and_registration(self):
        schema = MODULE.FeatherGroupOfMasksSEGS.define_schema()

        self.assertEqual(schema.node_id, "UtilitySuiteFeatherGroupOfMasksSEGS")
        self.assertEqual(schema.display_name, "Feather Group of Masks (SEGS)")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual(
            [item.id for item in schema.inputs],
            ["segs", "left", "top", "right", "bottom"],
        )
        self.assertTrue(all(item.default == 0 for item in schema.inputs[1:]))
        nodes = asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list())
        self.assertIn(MODULE.FeatherGroupOfMasksSEGS, nodes)

    def test_left_feather_affects_left_column_of_two_by_two_grid(self):
        entries = [
            segment((0, 0, 4, 4), np.ones((4, 4), dtype=np.float32), "top-left"),
            segment((4, 0, 8, 4), np.ones((4, 4), dtype=np.float32), "top-right"),
            segment((0, 4, 4, 8), np.ones((4, 4), dtype=np.float32), "bottom-left"),
            segment((4, 4, 8, 8), np.ones((4, 4), dtype=np.float32), "bottom-right"),
        ]

        result = MODULE.feather_group_masks_segs(((8, 8), entries), 2, 0, 0, 0)

        expected_left = np.ones((4, 4), dtype=np.float32)
        expected_left[:, 0] = 0.5
        self.assertTrue(np.array_equal(result[1][0].cropped_mask, expected_left))
        self.assertTrue(np.array_equal(result[1][2].cropped_mask, expected_left))
        self.assertTrue(np.array_equal(result[1][1].cropped_mask, np.ones((4, 4))))
        self.assertTrue(np.array_equal(result[1][3].cropped_mask, np.ones((4, 4))))

    def test_overlapping_crops_use_outer_union_boundary_only(self):
        first = segment((0, 0, 6, 4), np.ones((4, 6), dtype=np.float32), "first")
        second = segment((4, 0, 10, 4), np.ones((4, 6), dtype=np.float32), "second")

        result = MODULE.feather_group_masks_segs(
            ((4, 10), [first, second]), 4, 0, 0, 0
        )

        self.assertTrue(
            np.array_equal(
                result[1][0].cropped_mask[0],
                np.array([0.25, 0.5, 0.75, 1.0, 1.0, 1.0], dtype=np.float32),
            )
        )
        self.assertTrue(np.array_equal(result[1][1].cropped_mask, np.ones((4, 6))))

    def test_crop_union_not_nonzero_mask_defines_boundary(self):
        mask = np.zeros((4, 8), dtype=np.float32)
        mask[:, 2:6] = 1.0
        source = ((4, 8), [segment((0, 0, 8, 4), mask, "soft")])

        result = MODULE.feather_group_masks_segs(source, 4, 0, 0, 0)

        self.assertEqual(float(result[1][0].cropped_mask[0, 2]), 0.75)
        self.assertEqual(float(result[1][0].cropped_mask[0, 3]), 1.0)

    def test_selected_sides_multiply_at_group_corner(self):
        source = (
            (4, 4),
            [segment((0, 0, 4, 4), np.ones((4, 4), dtype=np.float32), "square")],
        )

        result = MODULE.feather_group_masks_segs(source, 2, 2, 0, 0)

        self.assertEqual(float(result[1][0].cropped_mask[0, 0]), 0.25)
        self.assertEqual(float(result[1][0].cropped_mask[1, 1]), 1.0)

    def test_zero_feathers_preserve_masks_and_metadata(self):
        mask = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
        source_entry = segment((1, 1, 3, 3), mask, "original")

        result = MODULE.feather_group_masks_segs(
            ((4, 4), [source_entry]), 0, 0, 0, 0
        )

        output = result[1][0]
        self.assertTrue(np.array_equal(output.cropped_mask, mask))
        self.assertEqual(output.crop_region, source_entry.crop_region)
        self.assertEqual(output.bbox, source_entry.bbox)
        self.assertEqual(output.label, source_entry.label)
        self.assertIs(output.cropped_image, source_entry.cropped_image)

    def test_empty_segs_and_invalid_amounts(self):
        self.assertEqual(
            MODULE.feather_group_masks_segs(((4, 5), []), 2, 2, 2, 2),
            ((4, 5), []),
        )
        with self.assertRaisesRegex(ValueError, "must be nonnegative"):
            MODULE.feather_group_masks_segs(((4, 5), []), -1, 0, 0, 0)
        with self.assertRaisesRegex(TypeError, "must be integers"):
            MODULE.feather_group_masks_segs(((4, 5), []), True, 0, 0, 0)


if __name__ == "__main__":
    unittest.main()
