import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_mask_batch_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MASK_BATCH = sys.modules[f"{SPEC.name}.mask_batch"]


def convert(mask, bbox_fill=False, drop_size=1, contour_fill=False):
    return MASK_BATCH.mask_to_mask_batch(mask, bbox_fill, drop_size, contour_fill)


class MaskToMaskBatchTests(unittest.TestCase):
    def test_one_region_full_canvas_and_schema(self):
        mask = torch.zeros(12, 18)
        mask[2:8, 4:12] = 1
        result = convert(mask)
        self.assertEqual(result.shape, (1, 12, 18))
        self.assertTrue(torch.equal(result[0], mask))

        schema = MASK_BATCH.MaskToMaskBatch.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteMaskToMaskBatch")
        self.assertEqual(schema.display_name, "Mask to Mask Batch")
        self.assertEqual(schema.category, "Utility Suite/Mask")
        self.assertEqual([item.id for item in schema.inputs], ["mask", "bbox_fill", "drop_size", "contour_fill"])
        self.assertEqual(schema.inputs[1].default, False)
        self.assertEqual(schema.inputs[2].default, 10)
        self.assertEqual(schema.inputs[2].min, 1)
        self.assertEqual(schema.inputs[3].default, False)
        self.assertEqual(schema.outputs[0].io_type, "MASK")
        self.assertFalse(schema.outputs[0].is_output_list)

    def test_three_regions_preserve_opencv_order_and_placement(self):
        mask = torch.zeros(30, 40)
        mask[2:8, 3:10] = 0.25
        mask[12:20, 15:24] = 0.5
        mask[21:29, 30:39] = 0.75
        result = convert(mask)

        expected = []
        source = (mask.numpy() * 255).astype(np.uint8)
        contours, hierarchy = MASK_BATCH.cv2.findContours(source, MASK_BATCH.cv2.RETR_TREE, MASK_BATCH.cv2.CHAIN_APPROX_SIMPLE)
        for index, contour in enumerate(contours):
            if hierarchy[0][index][3] == -1:
                separated = np.zeros_like(source)
                MASK_BATCH.cv2.drawContours(separated, [contour], 0, 255, -1)
                expected.append(torch.from_numpy((mask.numpy() * (separated / 255.0) * 255).astype(np.uint8) / 255.0))

        self.assertEqual(result.shape, (3, 30, 40))
        for actual, wanted in zip(result, expected):
            self.assertTrue(torch.equal(actual, wanted.to(torch.float32)))

    def test_bbox_fill_uses_component_bounding_rectangle(self):
        mask = torch.zeros(14, 17)
        mask[2:10, 3:5] = 1
        mask[8:10, 3:12] = 1
        result = convert(mask, bbox_fill=True)
        expected = torch.zeros_like(mask)
        expected[2:10, 3:12] = 1
        self.assertTrue(torch.equal(result[0], expected))

    def test_contour_fill_controls_holes(self):
        mask = torch.zeros(20, 20)
        mask[2:18, 2:18] = 0.8
        mask[7:13, 7:13] = 0
        preserved = convert(mask, contour_fill=False)[0]
        filled = convert(mask, contour_fill=True)[0]
        self.assertEqual(preserved[9, 9].item(), 0)
        self.assertEqual(filled[9, 9].item(), 1)
        self.assertAlmostEqual(preserved[3, 3].item(), 204 / 255)

    def test_drop_size_is_strict_for_both_bbox_dimensions(self):
        mask = torch.zeros(30, 40)
        mask[1:6, 1:6] = 1
        mask[10:16, 10:16] = 1
        mask[20:27, 20:27] = 1
        self.assertEqual(convert(mask, drop_size=4).shape[0], 3)
        at_cutoff = convert(mask, drop_size=5)
        self.assertEqual(at_cutoff.shape[0], 2)
        self.assertEqual(int(at_cutoff[:, 1:6, 1:6].count_nonzero()), 0)
        self.assertEqual(convert(mask, drop_size=6).shape[0], 1)

    def test_border_region_and_non_square_canvas(self):
        mask = torch.zeros(11, 23)
        mask[0:6, 0:8] = 1
        result = convert(mask)
        self.assertEqual(result.shape, (1, 11, 23))
        self.assertTrue(torch.equal(result[0], mask))

    def test_empty_or_dropped_result_is_one_cpu_float32_zero_mask(self):
        result = convert(torch.zeros(2, 9, 13), drop_size=10)
        self.assertEqual(result.shape, (1, 9, 13))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(result.count_nonzero().item(), 0)

    def test_batch_planes_are_processed_in_source_then_contour_order(self):
        mask = torch.zeros(2, 20, 25)
        mask[0, 2:8, 2:8] = 0.4
        mask[1, 10:18, 14:23] = 0.8
        result = convert(mask)
        self.assertEqual(result.shape, (2, 20, 25))
        self.assertGreater(result[0, 3, 3].item(), 0)
        self.assertEqual(result[0, 12, 16].item(), 0)
        self.assertGreater(result[1, 12, 16].item(), 0)

    def test_rank_and_spatial_validation(self):
        with self.assertRaisesRegex(ValueError, "rank-2 or rank-3"):
            convert(torch.zeros(1, 1, 1, 2, 2))
        with self.assertRaisesRegex(ValueError, "nonempty spatial"):
            convert(torch.zeros(1, 0, 3))

    def test_production_module_has_no_impact_or_kj_dependency(self):
        source = (PACKAGE_ROOT / "mask_batch.py").read_text(encoding="utf-8")
        self.assertNotIn("impact", source.lower())
        self.assertNotIn("kjnodes", source.lower())


if __name__ == "__main__":
    unittest.main()
