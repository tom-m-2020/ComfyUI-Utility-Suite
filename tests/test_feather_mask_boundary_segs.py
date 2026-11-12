from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_feather_mask_boundary_segs_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.feather_mask_boundary_segs"]
BOUNDARY = sys.modules[f"{SPEC.name}.feather_mask_boundary"]

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)


def segment(crop, mask, label):
    return SEG(object(), mask, 0.75, crop, tuple(crop), label, object())


class FeatherMaskFromBoundarySEGSTests(unittest.TestCase):
    def test_each_cropped_mask_gets_its_own_left_feather(self):
        first = segment((0, 0, 5, 4), np.ones((4, 5), dtype=np.float32), "first")
        second = segment((5, 0, 10, 4), np.ones((4, 5), dtype=np.float32), "second")

        result = MODULE.feather_mask_from_boundary_segs(
            ((4, 10), [first, second]), 2, 0, 0, 0, False, False
        )

        expected = np.ones((4, 5), dtype=np.float32)
        expected[:, 0] = 0.5
        np.testing.assert_array_equal(result[1][0].cropped_mask, expected)
        np.testing.assert_array_equal(result[1][1].cropped_mask, expected)

    def test_left_and_right_apply_to_every_mask(self):
        entries = [
            segment((0, 0, 6, 3), np.ones((3, 6), dtype=np.float32), "a"),
            segment((4, 3, 10, 6), np.ones((3, 6), dtype=np.float32), "b"),
        ]
        result = MODULE.feather_mask_from_boundary_segs(
            ((6, 10), entries), 3, 0, 2, 0, False, False
        )
        expected = np.array([1 / 3, 2 / 3, 1, 1, 1, 1 / 2], dtype=np.float32)
        for entry in result[1]:
            np.testing.assert_allclose(entry.cropped_mask[1], expected, rtol=0, atol=0)

    def test_matches_mask_node_independently_for_irregular_soft_masks(self):
        first_mask = np.zeros((7, 8), dtype=np.float32)
        first_mask[1:6, 2:7] = 0.7
        first_mask[3, 4] = 0
        second_mask = np.zeros((6, 9), dtype=np.float32)
        second_mask[2:5, 1:8] = np.linspace(0.2, 0.9, 21, dtype=np.float32).reshape(3, 7)
        entries = [
            segment((0, 0, 8, 7), first_mask, "first"),
            segment((8, 1, 17, 7), second_mask, "second"),
        ]

        result = MODULE.feather_mask_from_boundary_segs(
            ((7, 17), entries), 2, 3, 4, 1, False, True
        )

        for original, output in zip(entries, result[1]):
            expected = BOUNDARY.feather_mask_from_boundary(
                torch.from_numpy(original.cropped_mask), 2, 3, 4, 1, False, True
            )
            np.testing.assert_array_equal(output.cropped_mask, expected.numpy())

    def test_outward_and_tapered_corner_modes_are_forwarded(self):
        mask = np.zeros((12, 12), dtype=np.float32)
        mask[4:8, 4:8] = 1
        entry = segment((0, 0, 12, 12), mask, "square")
        square = MODULE.feather_mask_from_boundary_segs(
            ((12, 12), [entry]), 0, 0, 4, 4, True, False
        )[1][0].cropped_mask
        tapered = MODULE.feather_mask_from_boundary_segs(
            ((12, 12), [entry]), 0, 0, 4, 4, True, True
        )[1][0].cropped_mask
        expected_square = BOUNDARY.feather_mask_from_boundary(
            torch.from_numpy(mask), 0, 0, 4, 4, True, False
        ).numpy()
        expected_tapered = BOUNDARY.feather_mask_from_boundary(
            torch.from_numpy(mask), 0, 0, 4, 4, True, True
        ).numpy()
        np.testing.assert_array_equal(square, expected_square)
        np.testing.assert_array_equal(tapered, expected_tapered)
        self.assertFalse(np.array_equal(square, tapered))
        np.testing.assert_array_equal(square[4:8, 4:8], np.ones((4, 4), dtype=np.float32))
        np.testing.assert_array_equal(tapered[4:8, 4:8], np.ones((4, 4), dtype=np.float32))

    def test_header_order_and_non_mask_fields_are_preserved(self):
        header = (8, 12)
        entries = [
            segment((0, 0, 6, 8), np.ones((8, 6), dtype=np.float32), "a"),
            segment((6, 0, 12, 8), np.ones((8, 6), dtype=np.float32), "b"),
        ]
        snapshots = [tuple(entry) for entry in entries]
        result = MODULE.feather_mask_from_boundary_segs(
            (header, entries), 2, 0, 0, 0, False, False
        )
        self.assertIs(result[0], header)
        self.assertEqual([entry.label for entry in result[1]], ["a", "b"])
        for original, output in zip(entries, result[1]):
            self.assertIsNot(output, original)
            self.assertEqual(output[:1] + output[2:], original[:1] + original[2:])
        self.assertEqual([tuple(entry) for entry in entries], snapshots)

    def test_empty_segs_and_all_zero_masks(self):
        header = (5, 7)
        empty = MODULE.feather_mask_from_boundary_segs(
            (header, []), 2, 2, 2, 2, False, False
        )
        self.assertIs(empty[0], header)
        self.assertEqual(empty[1], [])
        zero = segment((0, 0, 7, 5), np.zeros((5, 7), dtype=np.float32), "zero")
        result = MODULE.feather_mask_from_boundary_segs(
            (header, [zero]), 2, 2, 2, 2, False, False
        )
        np.testing.assert_array_equal(result[1][0].cropped_mask, zero.cropped_mask)

    def test_missing_and_malformed_masks_fail_clearly(self):
        entry = segment((0, 0, 5, 4), None, "missing")
        with self.assertRaisesRegex(TypeError, "no cropped_mask"):
            MODULE.feather_mask_from_boundary_segs(
                ((4, 5), [entry]), 0, 0, 0, 0, False, False
            )
        malformed = entry._replace(cropped_mask=np.ones((3, 5), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "does not match"):
            MODULE.feather_mask_from_boundary_segs(
                ((4, 5), [malformed]), 0, 0, 0, 0, False, False
            )

    def test_schema_and_registration(self):
        schema = MODULE.FeatherMaskFromBoundarySEGS.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteFeatherMaskFromBoundarySEGS")
        self.assertEqual(schema.display_name, "Feather Mask from Boundary (SEGS)")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual(
            [item.id for item in schema.inputs],
            ["segs", "left", "top", "right", "bottom", "outward", "tapered_corners"],
        )
        self.assertEqual([output.io_type for output in schema.outputs], ["SEGS"])
        self.assertIn(MODULE.FeatherMaskFromBoundarySEGS, asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list()))


if __name__ == "__main__":
    unittest.main()
