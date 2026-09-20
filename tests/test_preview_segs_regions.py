from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

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


def segment(crop, image=None, mask=None, label="", bbox=None):
    x1, y1, x2, y2 = crop
    if mask is None:
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
    return SEG(image, mask, 1.0, crop, crop if bbox is None else bbox, label, None)


def preview(shape, entries, background="image", line_width=1, fallback=None, labels=False):
    if labels:
        return MODULE.preview_segs_regions((shape, entries), background, line_width, fallback)[0].numpy()
    with mock.patch.object(MODULE, "_draw_index_labels"):
        return MODULE.preview_segs_regions((shape, entries), background, line_width, fallback)[0].numpy()


class PreviewSEGSRegionsTests(unittest.TestCase):
    def test_index_specs_are_zero_based_and_centered_on_crop_region(self):
        crops = [(10, 20, 30, 60), (50, 5, 90, 25), (0, 0, 12, 12)]
        specs = MODULE._index_label_specs(crops)
        self.assertEqual([spec[0] for spec in specs], ["0", "1", "2"])
        self.assertEqual([spec[1] for spec in specs], [(20.0, 40.0), (70.0, 15.0), (6.0, 6.0)])

    def test_bbox_does_not_affect_label_position(self):
        item = segment((10, 20, 50, 80), bbox=(0, 0, 2, 2))
        specs = MODULE._index_label_specs([item.crop_region])
        self.assertEqual(specs[0][1], (30.0, 50.0))

    def test_sparse_and_reordered_entries_use_current_sequence(self):
        spatial_first = segment((0, 0, 20, 20), label="original-0")
        spatial_last = segment((80, 0, 100, 20), label="original-4")
        reordered = [spatial_last, spatial_first]
        specs = MODULE._index_label_specs([entry.crop_region for entry in reordered])
        self.assertEqual(specs, [("0", (90.0, 10.0), 12), ("1", (10.0, 10.0), 12)])

    def test_overlapping_crops_each_receive_one_label(self):
        specs = MODULE._index_label_specs([(0, 0, 40, 40), (10, 10, 50, 50), (20, 20, 60, 60)])
        self.assertEqual(len(specs), 3)
        self.assertEqual([spec[0] for spec in specs], ["0", "1", "2"])

    def test_font_size_scales_with_conservative_limits(self):
        specs = MODULE._index_label_specs([(0, 0, 4, 4), (0, 0, 200, 100), (0, 0, 1000, 1000)])
        self.assertEqual([spec[2] for spec in specs], [12, 18, 72])

    def test_labels_are_visible_blue_and_translucent(self):
        black = np.zeros((100, 100, 3), dtype=np.float32)
        white = np.ones((100, 100, 3), dtype=np.float32)
        MODULE._draw_index_labels(black, [(0, 0, 100, 100)])
        MODULE._draw_index_labels(white, [(0, 0, 100, 100)])
        for canvas in (black, white):
            changed = canvas[35:65, 35:65]
            self.assertTrue(np.any(changed[:, :, 2] > changed[:, :, 0]))
        self.assertTrue(np.any((black > 0) & (black < 1)))
        self.assertTrue(np.any((white > 0) & (white < 1)))

    def test_labels_render_after_boundaries(self):
        order = []
        with (
            mock.patch.object(MODULE, "_draw_boundaries", side_effect=lambda *_: order.append("boundaries")),
            mock.patch.object(MODULE, "_draw_index_labels", side_effect=lambda *_: order.append("labels")),
        ):
            MODULE.preview_segs_regions(((32, 32), [segment((0, 0, 32, 32))]), "image", 1)
        self.assertEqual(order, ["boundaries", "labels"])

    def test_label_integration_preserves_shape_and_input_objects(self):
        mask = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
        item = segment((0, 0, 64, 64), mask=mask, label="unchanged")
        original_mask = item.cropped_mask.copy()
        result = preview((64, 64), [item], background="mask", labels=True)
        self.assertEqual(result.shape, (64, 64, 3))
        self.assertEqual(item.label, "unchanged")
        np.testing.assert_array_equal(item.cropped_mask, original_mask)
        self.assertTrue(np.any(result[:, :, 2] > result[:, :, 0]))

    def test_labels_render_in_image_and_mask_background_modes(self):
        image = np.full((1, 80, 80, 3), 0.8, dtype=np.float32)
        mask = np.full((80, 80), 0.2, dtype=np.float32)
        item = segment((0, 0, 80, 80), image=image, mask=mask)
        image_result = preview((80, 80), [item], background="image", labels=True)
        mask_result = preview((80, 80), [item], background="mask", labels=True)
        self.assertTrue(np.any(image_result[:, :, 2] > image_result[:, :, 0]))
        self.assertTrue(np.any(mask_result[:, :, 2] > mask_result[:, :, 0]))

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
        np.testing.assert_allclose(preview((6, 7), [], fallback=fallback, labels=True), 0.6)
        np.testing.assert_array_equal(preview((6, 7), [], background="mask", fallback=fallback, labels=True), 0)

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
        with mock.patch.object(MODULE, "_draw_index_labels"):
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
