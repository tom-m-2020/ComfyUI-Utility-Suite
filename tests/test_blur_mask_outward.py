from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_blur_mask_outward_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.blur_mask_outward"]


def reference_blur(mask: torch.Tensor, radius: float) -> torch.Tensor:
    batch = mask.unsqueeze(0) if mask.ndim == 2 else mask
    results = []
    for item in batch:
        array = np.clip(item.numpy() * 255.0, 0, 255).astype(np.uint8)
        image = Image.fromarray(array).filter(ImageFilter.GaussianBlur(radius))
        results.append(torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255.0))
    output = torch.stack(results)
    return output[0] if mask.ndim == 2 else output


class BlurMaskOutwardTests(unittest.TestCase):
    def test_schema_defaults_and_registration_contract(self):
        schema = MODULE.BlurMaskOutward.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteBlurMaskOutward")
        self.assertEqual(schema.display_name, "Blur Mask Outward")
        self.assertEqual(schema.category, "Utility Suite/Mask")
        self.assertEqual(
            [item.id for item in schema.inputs],
            ["mask", "blur_radius", "tapered_corners", "outward"],
        )
        self.assertEqual(schema.inputs[1].default, 0.0)
        self.assertEqual(schema.inputs[1].min, 0.0)
        self.assertEqual(schema.inputs[1].max, 100.0)
        self.assertEqual(schema.inputs[1].step, 0.1)
        self.assertFalse(schema.inputs[2].default)
        self.assertTrue(schema.inputs[3].default)

    def test_outward_false_exactly_matches_kj_pillow_path(self):
        mask = torch.zeros((2, 31, 45), dtype=torch.float32)
        mask[0, :, :22] = 1.0
        mask[1, 4:27, 8:35] = torch.linspace(0.0, 0.9, 27)[None, :]
        expected = reference_blur(mask, 4.5)
        actual = MODULE.blur_mask_outward(mask, 4.5, False, False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_straight_compensation_matches_audited_table(self):
        expected = {1: 3, 2: 5, 4: 10, 8: 20, 16: 41, 32: 81, 64: 163, 100: 254}
        actual = {radius: MODULE._minimum_straight_compensation(radius) for radius in expected}
        self.assertEqual(actual, expected)

    def test_square_outward_preserves_hard_plateau_for_audited_radii(self):
        for radius in (1, 2, 4, 8, 16, 32, 64, 100):
            with self.subTest(radius=radius):
                support = MODULE._pillow_support_bound(radius)
                mask = torch.zeros((2 * support + 9, 2 * support + 17), dtype=torch.float32)
                mask[4 : support + 4, 8 : support + 8] = 1.0
                actual = MODULE.blur_mask_outward(mask, radius, False, True)
                self.assertTrue(torch.equal(actual[mask == 1], torch.ones_like(actual[mask == 1])))
                self.assertGreater(torch.count_nonzero(actual[:, support + 8 :]), 0)
                self.assertEqual(tuple(actual.shape), tuple(mask.shape))

    def test_square_outward_preserves_even_an_isolated_hard_pixel(self):
        mask = torch.zeros((9, 9))
        mask[4, 4] = 1.0
        actual = MODULE.blur_mask_outward(mask, 64, False, True)
        self.assertEqual(float(actual[4, 4]), 1.0)
        self.assertGreater(torch.count_nonzero(actual), 1)

    def test_radius_64_requires_163_not_64_for_straight_edge(self):
        radius = 64
        support = MODULE._pillow_support_bound(radius)
        boundary = support + 4
        step = np.zeros((1, 2 * boundary), dtype=np.float32)
        step[:, :boundary] = 1.0

        insufficient = MODULE._pillow_blur_uint8(MODULE._grow(step, 64, False), radius)
        sufficient = MODULE._pillow_blur_uint8(MODULE._grow(step, 163, False), radius)
        self.assertLess(float(insufficient[0, boundary - 1]), 1.0)
        self.assertEqual(float(sufficient[0, boundary - 1]), 1.0)

    def test_tapered_rectangle_reference_minima_and_conservative_calibration(self):
        rectangle_minima = {8: 31, 16: 62, 64: 246}
        for radius, amount in rectangle_minima.items():
            with self.subTest(radius=radius):
                support = MODULE._pillow_support_bound(radius)
                margin = amount + support + 3
                extent = 2 * support
                mask = np.zeros((2 * margin + extent, 2 * margin + extent), dtype=np.float32)
                mask[margin : margin + extent, margin : margin + extent] = 1.0
                corner = (margin, margin)
                below = MODULE._pillow_blur_uint8(
                    MODULE._grow(mask, amount - 1, True), radius
                )[corner]
                exact = MODULE._pillow_blur_uint8(MODULE._grow(mask, amount, True), radius)[corner]
                self.assertLess(float(below), 1.0)
                self.assertEqual(float(exact), 1.0)
                self.assertGreaterEqual(MODULE._minimum_tapered_compensation(radius), amount)

    def test_tapered_outward_preserves_isolated_hard_pixel_and_rectangle(self):
        for radius in (8, 16, 64):
            with self.subTest(radius=radius):
                point = torch.zeros((9, 9))
                point[4, 4] = 1.0
                point_result = MODULE.blur_mask_outward(point, radius, True, True)
                self.assertEqual(float(point_result[4, 4]), 1.0)

                rectangle = torch.zeros((11, 13))
                rectangle[3:8, 4:10] = 1.0
                rectangle_result = MODULE.blur_mask_outward(rectangle, radius, True, True)
                self.assertTrue(
                    torch.equal(
                        rectangle_result[rectangle == 1],
                        torch.ones_like(rectangle_result[rectangle == 1]),
                    )
                )

    def test_occupied_canvas_edges_and_corners_do_not_fade(self):
        masks = []
        for side in ("left", "right", "top", "bottom"):
            mask = torch.zeros((25, 29))
            if side == "left":
                mask[5:20, :12] = 1
                coordinate = (12, 0)
            elif side == "right":
                mask[5:20, 17:] = 1
                coordinate = (12, 28)
            elif side == "top":
                mask[:12, 6:23] = 1
                coordinate = (0, 14)
            else:
                mask[13:, 6:23] = 1
                coordinate = (24, 14)
            masks.append((mask, coordinate))

        corner = torch.zeros((25, 29))
        corner[:12, :14] = 1
        masks.append((corner, (0, 0)))
        for tapered in (False, True):
            for mask, coordinate in masks:
                actual = MODULE.blur_mask_outward(mask, 8, tapered, True)
                self.assertEqual(float(actual[coordinate]), 1.0)

    def test_internal_boundaries_still_have_outward_gaussian_transition(self):
        mask = torch.zeros((17, 31))
        mask[4:13, 7:16] = 1.0
        actual = MODULE.blur_mask_outward(mask, 4, False, True)
        self.assertEqual(float(actual[8, 15]), 1.0)
        self.assertGreater(float(actual[8, 16]), 0.0)
        self.assertLess(float(actual[8, 20]), float(actual[8, 16]))

    def test_fractional_radius_is_deterministic_and_cached(self):
        first = MODULE._minimum_straight_compensation(2.5)
        second = MODULE._minimum_straight_compensation(2.5)
        self.assertEqual(first, second)
        self.assertEqual(MODULE._minimum_straight_compensation.cache_info().hits >= 1, True)

        mask = torch.zeros((13, 23))
        mask[3:10, 5:13] = 1
        actual = MODULE.blur_mask_outward(mask, 2.5, False, True)
        self.assertTrue(torch.all(actual[mask == 1] == 1))

    def test_zero_radius_is_identity_without_mutating_input(self):
        mask = torch.rand((2, 9, 11))
        original = mask.clone()
        actual = MODULE.blur_mask_outward(mask, 0, False, True)
        self.assertTrue(torch.equal(actual, mask))
        self.assertIsNot(actual, mask)
        self.assertTrue(torch.equal(mask, original))

    def test_batch_and_rank_two_preserve_shape_and_order(self):
        mask = torch.zeros((2, 15, 19))
        mask[0, 3:10, 2:8] = 1
        mask[1, 5:13, 11:18] = 1
        actual = MODULE.blur_mask_outward(mask, 2, False, True)
        self.assertEqual(tuple(actual.shape), (2, 15, 19))
        self.assertGreater(float(actual[0, 6, 6]), 0)
        self.assertGreater(float(actual[0, 6, 6]), float(actual[0, 8, 15]))
        self.assertGreater(float(actual[1, 8, 15]), 0)
        self.assertGreater(float(actual[1, 8, 15]), float(actual[1, 6, 6]))

        rank_two = MODULE.blur_mask_outward(mask[0], 2, False, True)
        self.assertEqual(tuple(rank_two.shape), (15, 19))

    def test_grayscale_is_not_thresholded_or_normalized(self):
        mask = torch.zeros((17, 25))
        mask[4:13, 6:16] = 0.7
        mask[7:10, 9:13] = 0.35
        original = mask.clone()
        actual = MODULE.blur_mask_outward(mask, 3, False, True)
        self.assertLessEqual(float(actual.max()), float(np.float32(178 / 255)))
        self.assertGreater(torch.unique(actual).numel(), 2)
        self.assertTrue(torch.equal(mask, original))

    def test_validation_rejects_bad_radius_shape_and_boolean_values(self):
        mask = torch.zeros((4, 5))
        for radius in (-1, 101, float("nan"), True):
            with self.subTest(radius=radius), self.assertRaises((TypeError, ValueError)):
                MODULE.blur_mask_outward(mask, radius)
        with self.assertRaises(ValueError):
            MODULE.blur_mask_outward(torch.zeros((1, 2, 3, 4)), 2)
        with self.assertRaises(TypeError):
            MODULE.blur_mask_outward(mask, 2, tapered_corners=1)
        with self.assertRaises(TypeError):
            MODULE.blur_mask_outward(mask, 2, outward=1)


if __name__ == "__main__":
    unittest.main()
