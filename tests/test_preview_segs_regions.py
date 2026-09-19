from __future__ import annotations

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
    "utility_suite_preview_segs_regions_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.preview_segs_regions"]
MASK_TILE = sys.modules[f"{SPEC.name}.mask_tile_segs"]

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)


def segment(crop, image=None, mask=None, label=""):
    x1, y1, x2, y2 = crop
    if mask is None:
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
    return SEG(image, mask, 1.0, crop, crop, label, None)


def preview(shape, entries, background="image", line_width=1, fallback=None):
    return MODULE.preview_segs_regions((shape, entries), background, line_width, fallback)[0].numpy()


class PreviewSEGSRegionsTests(unittest.TestCase):
    def test_no_overlap_draws_only_blue_boundaries(self):
        entries = [segment((0, 0, 6, 6)), segment((6, 0, 12, 6))]
        result = preview((6, 12), entries)
        np.testing.assert_array_equal(result[0, 2], (0, 0, 1))
        np.testing.assert_array_equal(result[3, 3], (0, 0, 0))
        self.assertFalse(np.any(np.all(result == (1, 1, 0), axis=2)))
        self.assertFalse(np.any(np.all(result == (1, 0, 0), axis=2)))

    def test_two_tile_overlap_uses_yellow_hatch(self):
        entries = [segment((0, 0, 16, 16)), segment((8, 0, 24, 16))]
        result = preview((16, 24), entries)
        np.testing.assert_array_equal(result[7, 9], (1, 1, 0))
        np.testing.assert_allclose(result[7, 11], (0.22, 0.22, 0.0))

    def test_three_tile_intersection_has_yellow_and_red(self):
        entries = [segment((0, 0, 10, 18)), segment((2, 0, 12, 18)), segment((4, 0, 14, 18))]
        result = preview((18, 14), entries)
        np.testing.assert_array_equal(result[13, 3], (1, 1, 0))
        np.testing.assert_array_equal(result[8, 8], (1, 0, 0))

    def test_four_way_grid_intersection_has_yellow_and_red(self):
        crops = [(0, 0, 20, 20), (8, 0, 28, 20), (0, 8, 20, 28), (8, 8, 28, 28)]
        result = preview((28, 28), [segment(crop) for crop in crops])
        np.testing.assert_array_equal(result[7, 9], (1, 1, 0))
        np.testing.assert_array_equal(result[17, 15], (1, 0, 0))

    def test_line_width_is_drawn_inside_exclusive_crop(self):
        result = preview((8, 9), [segment((2, 1, 7, 7))], line_width=2)
        np.testing.assert_array_equal(result[1, 3], (0, 0, 1))
        np.testing.assert_array_equal(result[2, 3], (0, 0, 1))
        np.testing.assert_array_equal(result[3, 4], (0, 0, 0))
        np.testing.assert_array_equal(result[7, 3], (0, 0, 0))

    def test_image_present_has_priority_over_fallback(self):
        image = np.full((1, 6, 6, 3), 0.25, dtype=np.float32)
        fallback = torch.full((1, 6, 6, 3), 0.75)
        result = preview((6, 6), [segment((0, 0, 6, 6), image=image)], fallback=fallback)
        np.testing.assert_array_equal(result[3, 3], (0.25, 0.25, 0.25))

    def test_image_absent_uses_fallback(self):
        fallback = torch.full((1, 6, 6, 3), 0.75)
        result = preview((6, 6), [segment((0, 0, 6, 6))], fallback=fallback)
        np.testing.assert_array_equal(result[3, 3], (0.75, 0.75, 0.75))

    def test_image_absent_without_fallback_is_black_even_with_mask(self):
        item = segment((0, 0, 6, 6), mask=np.ones((6, 6), dtype=np.float32))
        np.testing.assert_array_equal(preview((6, 6), [item])[3, 3], (0, 0, 0))

    def test_mask_present_renders_grayscale_gradient(self):
        mask = np.zeros((6, 6), dtype=np.float32)
        mask[3, 3] = 0.375
        result = preview((6, 6), [segment((0, 0, 6, 6), mask=mask)], background="mask")
        np.testing.assert_array_equal(result[3, 3], (0.375, 0.375, 0.375))

    def test_mask_absent_is_black(self):
        item = segment((0, 0, 6, 6))._replace(cropped_mask=None)
        np.testing.assert_array_equal(preview((6, 6), [item], background="mask")[3, 3], (0, 0, 0))

    def test_mask_absent_ignores_image_and_fallback(self):
        image = np.ones((1, 6, 6, 3), dtype=np.float32)
        fallback = torch.ones((1, 6, 6, 3))
        item = segment((0, 0, 6, 6), image=image)._replace(cropped_mask=None)
        result = preview((6, 6), [item], background="mask", fallback=fallback)
        np.testing.assert_array_equal(result[3, 3], (0, 0, 0))

    def test_overlapping_image_crops_use_later_entry_overwrite(self):
        first = segment((0, 0, 8, 8), image=np.full((1, 8, 8, 3), 0.2, dtype=np.float32))
        second = segment((4, 0, 12, 8), image=np.full((1, 8, 8, 3), 0.8, dtype=np.float32))
        result = preview((8, 12), [first, second])
        np.testing.assert_allclose(result[4, 6], (0.844, 0.844, 0.624))

    def test_overlapping_masks_use_gradient_preserving_maximum(self):
        first = segment((0, 0, 8, 8), mask=np.full((8, 8), 0.7, dtype=np.float32))
        second = segment((4, 0, 12, 8), mask=np.full((8, 8), 0.4, dtype=np.float32))
        result = preview((8, 12), [first, second], background="mask")
        np.testing.assert_allclose(result[4, 6], (0.766, 0.766, 0.546), atol=1e-7)

    def test_mixed_images_leave_missing_crop_black_and_do_not_patch_with_fallback(self):
        first = segment((0, 0, 6, 6), image=np.full((1, 6, 6, 3), 0.4, dtype=np.float32))
        second = segment((6, 0, 12, 6))
        fallback = torch.ones((1, 6, 12, 3))
        result = preview((6, 12), [first, second], fallback=fallback)
        np.testing.assert_allclose(result[3, 3], (0.4, 0.4, 0.4))
        np.testing.assert_array_equal(result[3, 9], (0, 0, 0))

    def test_empty_segs_background_rules(self):
        fallback = torch.full((1, 6, 7, 3), 0.6)
        np.testing.assert_allclose(preview((6, 7), [], fallback=fallback), 0.6)
        np.testing.assert_array_equal(preview((6, 7), [], background="mask", fallback=fallback), 0)

    def test_invalid_fallback_and_crop_payload_fail(self):
        with self.assertRaisesRegex(ValueError, "do not match"):
            preview((6, 7), [], fallback=torch.zeros((1, 5, 7, 3)))
        bad_image = np.zeros((1, 5, 6, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "does not match"):
            preview((6, 6), [segment((0, 0, 6, 6), image=bad_image)])
        bad_mask = np.zeros((5, 6), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "does not match"):
            preview((6, 6), [segment((0, 0, 6, 6), mask=bad_mask)], background="mask")

    def test_invalid_structure_background_and_line_width_fail(self):
        with self.assertRaisesRegex(TypeError, "shaped as"):
            MODULE.preview_segs_regions([], "image", 1)
        with self.assertRaisesRegex(ValueError, "expected 'image' or 'mask'"):
            preview((6, 6), [], background="other")
        with self.assertRaisesRegex(ValueError, "positive"):
            preview((6, 6), [], line_width=0)

    def test_mask_to_tile_segs_fixed_and_uniform_interoperability(self):
        mask = torch.linspace(0, 1, 14 * 14).reshape(1, 14, 14)
        fixed, _ = MASK_TILE.mask_to_tile_segs(mask, "fixed_tile", 8, 8, 2, 2, 3, 3, 0, 0, 1, 1)
        fixed_result = MODULE.preview_segs_regions(fixed, "mask", 1)
        self.assertEqual(tuple(fixed_result.shape), (1, 14, 14, 3))

        uniform, _ = MASK_TILE.mask_to_tile_segs(mask, "uniform_grid", 10, 10, 0, 0, 3, 3, 2, 2, 2, 2)
        uniform_result = MODULE.preview_segs_regions(uniform, "mask", 1)[0].numpy()
        self.assertTrue(np.any(np.all(uniform_result == (1, 1, 0), axis=2)))
        self.assertTrue(np.any(np.all(uniform_result == (1, 0, 0), axis=2)))

    def test_schema_is_one_ordinary_image_output(self):
        schema = MODULE.PreviewSEGSRegions.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuitePreviewSEGSRegions")
        self.assertEqual(schema.display_name, "Preview SEGS Regions")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual([entry.io_type for entry in schema.inputs], ["SEGS", "IMAGE", "COMBO", "INT"])
        self.assertEqual([output.io_type for output in schema.outputs], ["IMAGE"])
        self.assertFalse(schema.outputs[0].is_output_list)


if __name__ == "__main__":
    unittest.main()
