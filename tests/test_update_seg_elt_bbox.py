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
    "utility_suite_update_seg_elt_bbox_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.update_seg_elt_bbox"]

SEG = namedtuple(
    "SEG",
    "cropped_image cropped_mask confidence crop_region bbox label control_net_wrapper",
)


def segment(crop, mask, bbox=(0, 0, 1, 1)):
    return SEG(object(), mask, 0.75, crop, bbox, "label", object())


class UpdateSEGELTBBoxTests(unittest.TestCase):
    def test_schema_and_registration(self):
        schema = MODULE.UpdateSEGELTBBox.define_schema()
        self.assertEqual(schema.display_name, "Update SEG_ELT bbox")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual([item.id for item in schema.inputs], ["segs"])
        self.assertEqual([item.id for item in schema.outputs], ["segs"])
        nodes = asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list())
        self.assertIn(MODULE.UpdateSEGELTBBox, nodes)

    def test_updates_each_bbox_in_source_canvas_coordinates(self):
        first_mask = np.zeros((6, 8), dtype=np.float32)
        first_mask[1:5, 2:7] = 1.0
        second_mask = torch.zeros((1, 5, 7))
        second_mask[0, 0:2, 4:7] = 1.0
        first = segment((10, 20, 18, 26), first_mask)
        second = segment((30, 40, 37, 45), second_mask)

        output = MODULE.update_seg_elt_bbox(((100, 120), [first, second]))

        self.assertEqual(output[1][0].bbox, (12, 21, 17, 25))
        self.assertEqual(output[1][1].bbox, (34, 40, 37, 42))

    def test_uses_shared_eight_bit_quantized_boundary(self):
        below = np.nextafter(np.float32(1.0 / 255.0), np.float32(0.0))
        mask = np.zeros((5, 6), dtype=np.float32)
        mask[0, 0] = below
        mask[3, 4] = np.float32(1.0 / 255.0)

        output = MODULE.update_seg_elt_bbox(((20, 20), [segment((7, 8, 13, 13), mask)]))

        self.assertEqual(output[1][0].bbox, (11, 11, 12, 12))

    def test_empty_mask_uses_crop_region_fallback(self):
        item = segment((7, 8, 13, 13), np.zeros((5, 6), dtype=np.float32))
        output = MODULE.update_seg_elt_bbox(((20, 20), [item]))
        self.assertEqual(output[1][0].bbox, (7, 8, 13, 13))

    def test_preserves_header_metadata_order_and_input(self):
        header = (20, 20)
        wrapper = object()
        image = object()
        item = segment((2, 3, 6, 8), np.ones((5, 4), dtype=np.float32))._replace(
            cropped_image=image,
            confidence=0.25,
            label="kept",
            control_net_wrapper=wrapper,
        )
        original = tuple(item)

        output = MODULE.update_seg_elt_bbox((header, [item]))

        self.assertIs(output[0], header)
        self.assertIsNot(output[1][0], item)
        self.assertIs(output[1][0].cropped_image, image)
        self.assertIs(output[1][0].cropped_mask, item.cropped_mask)
        self.assertEqual(output[1][0].confidence, 0.25)
        self.assertEqual(output[1][0].label, "kept")
        self.assertIs(output[1][0].control_net_wrapper, wrapper)
        self.assertEqual(tuple(item), original)

    def test_empty_segs_preserves_header(self):
        header = (20, 30)
        output = MODULE.update_seg_elt_bbox((header, []))
        self.assertIs(output[0], header)
        self.assertEqual(output[1], [])

    def test_rejects_missing_or_misaligned_masks(self):
        missing = segment((0, 0, 4, 3), None)
        with self.assertRaisesRegex(TypeError, "requires a cropped_mask"):
            MODULE.update_seg_elt_bbox(((10, 10), [missing]))
        misaligned = segment((0, 0, 4, 3), np.zeros((4, 4), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "does not match crop_region extent"):
            MODULE.update_seg_elt_bbox(((10, 10), [misaligned]))


if __name__ == "__main__":
    unittest.main()
