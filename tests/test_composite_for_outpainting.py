from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_composite_outpainting_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.composite_for_outpainting"]


def run_node(
    canvas: torch.Tensor,
    subject: torch.Tensor,
    **overrides,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    options = {
        "x_percent": 50,
        "y_percent": 50,
        "resize": False,
        "resize_type": "fit in canvas",
        "pixel_or_percent": "pixel",
        "target_size": 0,
        "scale_method": "nearest-exact",
        "blur_radius": 0.0,
        "tapered_corners": False,
        "combine_mask": "pixelwise min",
        "canvas_mask": None,
        "subject_mask": None,
    }
    options.update(overrides)
    return MODULE.composite_for_outpainting(canvas, subject, **options)


class CompositeForOutpaintingTests(unittest.TestCase):
    def test_missing_subject_mask_creates_centered_rectangular_hole(self):
        canvas = torch.ones((1, 6, 8, 3))
        subject = torch.zeros((1, 2, 4, 3))
        composited, masked, mask = run_node(canvas, subject)

        expected_mask = torch.ones((1, 6, 8))
        expected_mask[:, 2:4, 2:6] = 0
        torch.testing.assert_close(mask, expected_mask, rtol=0, atol=0)
        torch.testing.assert_close(composited[:, 2:4, 2:6], subject, rtol=0, atol=0)
        torch.testing.assert_close(masked, composited, rtol=0, atol=0)
        torch.testing.assert_close(canvas, torch.ones_like(canvas), rtol=0, atol=0)

    def test_subject_mask_controls_hole_but_not_hard_image_paste(self):
        canvas = torch.ones((1, 1, 4, 3))
        subject = torch.zeros((1, 1, 2, 3))
        subject_mask = torch.tensor([[[1.0, 0.0]]])
        composited, masked, mask = run_node(
            canvas,
            subject,
            x_percent=0,
            y_percent=0,
            subject_mask=subject_mask,
        )

        torch.testing.assert_close(composited[0, 0, :2], torch.zeros((2, 3)), rtol=0, atol=0)
        torch.testing.assert_close(mask[0, 0], torch.tensor([0.0, 1.0, 1.0, 1.0]), rtol=0, atol=0)
        torch.testing.assert_close(masked[0, 0, 0], torch.zeros(3), rtol=0, atol=0)
        torch.testing.assert_close(masked[0, 0, 1], torch.ones(3), rtol=0, atol=0)

    def test_fit_in_canvas_preserves_aspect_ratio_and_ignores_target_size(self):
        canvas = torch.ones((1, 8, 10, 3))
        subject = torch.zeros((1, 4, 2, 3))
        composited, _, mask = run_node(
            canvas,
            subject,
            resize=True,
            resize_type="fit in canvas",
            pixel_or_percent="percent",
            target_size=999,
        )
        self.assertEqual(tuple(composited.shape), (1, 8, 10, 3))
        self.assertEqual(int(torch.count_nonzero(mask == 0)), 32)
        self.assertTrue(torch.all(composited[:, :, 3:7] == 0))

    def test_manual_width_pixel_and_height_percent_preserve_proportion(self):
        canvas = torch.ones((1, 10, 20, 3))
        subject = torch.zeros((1, 2, 4, 3))

        _, _, width_mask = run_node(
            canvas,
            subject,
            resize=True,
            resize_type="width",
            pixel_or_percent="pixel",
            target_size=10,
        )
        self.assertEqual(int(torch.count_nonzero(width_mask == 0)), 50)

        _, _, height_mask = run_node(
            canvas,
            subject,
            resize=True,
            resize_type="height",
            pixel_or_percent="percent",
            target_size=50,
        )
        self.assertEqual(int(torch.count_nonzero(height_mask == 0)), 50)

    def test_zero_manual_target_leaves_subject_unchanged(self):
        canvas = torch.ones((1, 7, 9, 3))
        subject = torch.zeros((1, 3, 5, 3))
        _, _, mask = run_node(
            canvas,
            subject,
            resize=True,
            resize_type="width",
            target_size=0,
        )
        self.assertEqual(int(torch.count_nonzero(mask == 0)), 15)

    def test_percent_target_above_safety_limit_fails_but_fit_ignores_it(self):
        canvas = torch.ones((1, 8, 8, 3))
        subject = torch.zeros((1, 2, 2, 3))
        with self.assertRaisesRegex(ValueError, "cannot exceed 200"):
            run_node(
                canvas,
                subject,
                resize=True,
                resize_type="width",
                pixel_or_percent="percent",
                target_size=201,
            )
        run_node(
            canvas,
            subject,
            resize=True,
            resize_type="fit in canvas",
            pixel_or_percent="percent",
            target_size=201,
        )

    def test_subject_mask_is_stretched_to_resized_subject(self):
        canvas = torch.ones((1, 6, 6, 3))
        subject = torch.zeros((1, 2, 2, 3))
        subject_mask = torch.ones((1, 1, 1))
        _, _, mask = run_node(
            canvas,
            subject,
            resize=True,
            resize_type="width",
            target_size=4,
            subject_mask=subject_mask,
        )
        self.assertEqual(int(torch.count_nonzero(mask == 0)), 16)

    def test_canvas_mask_combination_supports_min_and_multiply(self):
        canvas = torch.ones((1, 61, 61, 3))
        subject = torch.zeros((1, 31, 31, 3))
        previous = torch.full((1, 61, 61), 0.5)
        _, _, minimum = run_node(
            canvas,
            subject,
            blur_radius=2.0,
            canvas_mask=previous,
            combine_mask="pixelwise min",
        )
        _, _, multiplied = run_node(
            canvas,
            subject,
            blur_radius=2.0,
            canvas_mask=previous,
            combine_mask="multiply",
        )
        self.assertTrue(torch.all(multiplied <= minimum))
        self.assertTrue(torch.any(multiplied < minimum))

    def test_negative_coordinates_clip_oversized_subject_symmetrically(self):
        canvas = torch.zeros((1, 4, 4, 3))
        subject = torch.arange(36, dtype=torch.float32).reshape(1, 6, 6, 1).expand(-1, -1, -1, 3)
        composited, _, mask = run_node(canvas, subject)
        torch.testing.assert_close(composited, subject[:, 1:5, 1:5], rtol=0, atol=0)
        self.assertTrue(torch.all(mask == 0))

    def test_outputs_chain_without_losing_previous_mask_holes(self):
        canvas = torch.ones((1, 2, 8, 3))
        subject = torch.zeros((1, 2, 2, 3))
        first_image, _, first_mask = run_node(
            canvas,
            subject,
            x_percent=0,
            y_percent=0,
        )
        second_image, _, second_mask = run_node(
            first_image,
            subject,
            x_percent=100,
            y_percent=0,
            canvas_mask=first_mask,
        )
        self.assertTrue(torch.all(second_image[:, :, :2] == 0))
        self.assertTrue(torch.all(second_image[:, :, -2:] == 0))
        self.assertTrue(torch.all(second_mask[:, :, :2] == 0))
        self.assertTrue(torch.all(second_mask[:, :, -2:] == 0))
        self.assertTrue(torch.all(second_mask[:, :, 2:6] == 1))

    def test_schema_and_registration_contract(self):
        schema = MODULE.CompositeForOutpainting.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteCompositeForOutpainting")
        self.assertEqual(schema.display_name, "Composite for Outpainting")
        self.assertEqual(schema.category, "Utility Suite/Image")
        self.assertEqual(
            [item.id for item in schema.inputs],
            [
                "canvas_image",
                "subject_image",
                "canvas_mask",
                "subject_mask",
                "x_percent",
                "y_percent",
                "resize",
                "resize_type",
                "pixel_or_percent",
                "target_size",
                "scale_method",
                "blur_radius",
                "tapered_corners",
                "combine_mask",
            ],
        )
        self.assertEqual(
            [item.id for item in schema.outputs],
            ["canvas_image", "canvas_image_masked", "canvas_mask"],
        )
        self.assertEqual((schema.inputs[4].min, schema.inputs[4].max), (0, 100))
        self.assertEqual((schema.inputs[5].min, schema.inputs[5].max), (0, 100))
        self.assertEqual(schema.inputs[9].min, 0)
        self.assertEqual(schema.inputs[11].step, 0.1)
        self.assertIn(
            MODULE.CompositeForOutpainting,
            asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list()),
        )


if __name__ == "__main__":
    unittest.main()
