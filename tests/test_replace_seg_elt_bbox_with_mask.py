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
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_replace_seg_elt_bbox_with_mask_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.replace_seg_elt_bbox_with_mask"]

SEG = namedtuple(
    "SEG",
    "cropped_image cropped_mask confidence crop_region bbox label control_net_wrapper",
)


def segment(crop, label):
    return SEG(object(), np.ones((crop[3] - crop[1], crop[2] - crop[0])), 0.75, crop, crop, label, object())


class ReplaceSEGELTBBoxWithMASKTests(unittest.TestCase):
    def setUp(self):
        self.header = (10, 12)
        self.first = segment((1, 2, 5, 7), "first")
        self.second = segment((6, 1, 11, 8), "second")
        self.segs = (self.header, [self.first, self.second])

    def test_schema_and_registration(self):
        schema = MODULE.ReplaceSEGELTBBoxWithMASK.define_schema()
        self.assertEqual(schema.display_name, "Replace SEG_ELT bbox with MASK")
        self.assertEqual([item.id for item in schema.inputs], ["segs", "mask", "index"])
        self.assertEqual(schema.inputs[2].default, 0)
        self.assertEqual([item.id for item in schema.outputs], ["segs"])
        nodes = asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list())
        self.assertIn(MODULE.ReplaceSEGELTBBoxWithMASK, nodes)

    def test_uses_mask_boundary_without_crop_offset(self):
        mask = torch.zeros((1, 10, 12))
        mask[0, 3:8, 2:9] = 1.0

        output = MODULE.replace_seg_elt_bbox_with_mask(self.segs, mask, 1)

        self.assertEqual(output[1][1].bbox, (2, 3, 9, 8))
        self.assertNotEqual(output[1][1].bbox, (8, 4, 15, 9))

    def test_truncates_boundary_to_source_canvas(self):
        mask = torch.zeros((14, 18))
        mask[8:13, 9:17] = 1.0

        output = MODULE.replace_seg_elt_bbox_with_mask(self.segs, mask, 0)

        self.assertEqual(output[1][0].bbox, (9, 8, 12, 10))

    def test_empty_or_fully_outside_boundary_falls_back_to_crop_region(self):
        empty = MODULE.replace_seg_elt_bbox_with_mask(self.segs, torch.zeros((10, 12)), 1)
        outside_mask = torch.zeros((20, 20))
        outside_mask[12:18, 14:19] = 1.0
        outside = MODULE.replace_seg_elt_bbox_with_mask(self.segs, outside_mask, 0)

        self.assertEqual(empty[1][1].bbox, tuple(self.second.crop_region))
        self.assertEqual(outside[1][0].bbox, tuple(self.first.crop_region))

    def test_clamps_mask_before_quantized_boundary(self):
        mask = torch.zeros((10, 12))
        mask[1:3, 2:5] = 2.0
        mask[8, 10] = -1.0

        output = MODULE.replace_seg_elt_bbox_with_mask(self.segs, mask, 0)

        self.assertEqual(output[1][0].bbox, (2, 1, 5, 3))

    def test_updates_only_selected_entry_and_preserves_input(self):
        mask = torch.zeros((10, 12))
        mask[4:6, 3:7] = 1.0
        original = tuple(self.second)

        output = MODULE.replace_seg_elt_bbox_with_mask(self.segs, mask, 1)

        self.assertIs(output[0], self.header)
        self.assertIs(output[1][0], self.first)
        self.assertIsNot(output[1][1], self.second)
        self.assertIs(output[1][1].cropped_image, self.second.cropped_image)
        self.assertIs(output[1][1].cropped_mask, self.second.cropped_mask)
        self.assertEqual(tuple(self.second), original)

    def test_validates_index_and_single_mask(self):
        mask = torch.zeros((10, 12))
        for index in (-1, 2):
            with self.subTest(index=index), self.assertRaisesRegex(IndexError, "outside SEGS entry range"):
                MODULE.replace_seg_elt_bbox_with_mask(self.segs, mask, index)
        with self.assertRaisesRegex(TypeError, "index"):
            MODULE.replace_seg_elt_bbox_with_mask(self.segs, mask, True)
        with self.assertRaisesRegex(ValueError, r"MASK \[H,W\] or \[1,H,W\]"):
            MODULE.replace_seg_elt_bbox_with_mask(self.segs, torch.zeros((2, 10, 12)), 0)


if __name__ == "__main__":
    unittest.main()
