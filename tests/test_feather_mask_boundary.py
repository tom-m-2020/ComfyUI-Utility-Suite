from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_feather_boundary_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.feather_mask_boundary"]

from comfy_extras.nodes_mask import FeatherMask


def reference(mask: torch.Tensor, left: int, top: int, right: int, bottom: int) -> torch.Tensor:
    batch = mask.unsqueeze(0) if mask.ndim == 2 else mask
    output = torch.zeros_like(batch)
    for index, item in enumerate(batch):
        coordinates = torch.nonzero(item, as_tuple=False)
        if coordinates.numel() == 0:
            continue
        y0 = int(coordinates[:, 0].min())
        y1 = int(coordinates[:, 0].max()) + 1
        x0 = int(coordinates[:, 1].min())
        x1 = int(coordinates[:, 1].max()) + 1
        crop = item[y0:y1, x0:x1]
        feathered = FeatherMask.execute(crop, left, top, right, bottom).result[0][0]
        output[index, y0:y1, x0:x1] = feathered
    return output[0] if mask.ndim == 2 else output


class FeatherMaskFromBoundaryTests(unittest.TestCase):
    def assert_reference(self, mask, left, top, right, bottom):
        expected = reference(mask, left, top, right, bottom)
        actual = MODULE.feather_mask_from_boundary(mask, left, top, right, bottom)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return actual

    def test_internal_hard_rectangle_matches_crop_core_feather_and_placement(self):
        mask = torch.zeros((1, 15, 19))
        mask[:, 3:12, 5:15] = 1
        actual = self.assert_reference(mask, 2, 3, 4, 5)
        self.assertEqual(tuple(actual.shape), tuple(mask.shape))
        self.assertEqual(torch.count_nonzero(actual[:, :3]), 0)
        self.assertEqual(torch.count_nonzero(actual[:, :, :5]), 0)

    def test_each_isolated_side_and_asymmetric_all_sides(self):
        mask = torch.zeros((1, 11, 14))
        mask[:, 2:9, 3:12] = 1
        settings = ((3, 0, 0, 0), (0, 3, 0, 0), (0, 0, 3, 0), (0, 0, 0, 3), (1, 2, 4, 5))
        for widths in settings:
            with self.subTest(widths=widths):
                self.assert_reference(mask, *widths)

    def test_content_touching_one_or_multiple_canvas_edges(self):
        masks = []
        one_edge = torch.zeros((1, 9, 12))
        one_edge[:, 2:8, :7] = 1
        masks.append(one_edge)
        two_edges = torch.zeros((1, 9, 12))
        two_edges[:, :6, 5:] = 1
        masks.append(two_edges)
        all_edges = torch.ones((1, 9, 12))
        masks.append(all_edges)
        for mask in masks:
            with self.subTest(bounds=tuple(torch.nonzero(mask).amin(0).tolist())):
                self.assert_reference(mask, 2, 3, 4, 5)

    def test_non_square_bbox_and_corner_factors_multiply(self):
        mask = torch.zeros((1, 10, 16))
        mask[:, 2:8, 4:14] = 1
        actual = self.assert_reference(mask, 0, 3, 4, 0)
        self.assertAlmostEqual(float(actual[0, 2, 13]), (1 / 3) * (1 / 4))
        self.assertAlmostEqual(float(actual[0, 4, 10]), 1.0)

    def test_width_smaller_equal_and_greater_than_bbox_dimension(self):
        mask = torch.zeros((1, 8, 11))
        mask[:, 2:6, 3:8] = 1
        for width in (3, 5, 9):
            with self.subTest(width=width):
                self.assert_reference(mask, width, width, width, width)

    def test_soft_values_are_multiplied_not_replaced(self):
        mask = torch.zeros((1, 7, 10), dtype=torch.float64)
        values = torch.linspace(0.1, 0.9, 24, dtype=torch.float64).reshape(4, 6)
        mask[:, 2:6, 3:9] = values
        original = mask.clone()
        actual = self.assert_reference(mask, 2, 2, 3, 1)
        self.assertEqual(actual.dtype, torch.float64)
        self.assertAlmostEqual(float(actual[0, 2, 3]), float(values[0, 0] * 0.5 * 0.5))
        torch.testing.assert_close(mask, original, rtol=0, atol=0)

    def test_internal_holes_and_irregular_content_remain_zero(self):
        mask = torch.zeros((1, 12, 14))
        mask[:, 2:10, 3:12] = 1
        mask[:, 4:8, 6:10] = 0
        mask[:, 2:5, 10:12] = 0
        actual = self.assert_reference(mask, 2, 3, 4, 2)
        self.assertEqual(torch.count_nonzero(actual[:, 4:8, 6:10]), 0)
        self.assertEqual(torch.count_nonzero(actual[:, 2:5, 10:12]), 0)

    def test_all_zero_mask_returns_deterministic_zero(self):
        mask = torch.zeros((2, 8, 9))
        actual = MODULE.feather_mask_from_boundary(mask, 4, 4, 4, 4)
        self.assertTrue(torch.equal(actual, mask))
        self.assertIsNot(actual, mask)

    def test_batch_items_use_independent_bounding_boxes(self):
        mask = torch.zeros((3, 12, 15))
        mask[0, 1:6, 2:8] = 1
        mask[1, 5:11, 8:15] = 0.7
        mask[2, 3:9, 4:12] = 1
        mask[2, 5:7, 6:10] = 0
        self.assert_reference(mask, 2, 3, 4, 1)

    def test_internal_content_discriminator_differs_from_core_canvas_edge_feather(self):
        mask = torch.zeros((1, 12, 20))
        mask[:, 3:9, 4:14] = 1
        boundary = MODULE.feather_mask_from_boundary(mask, 0, 0, 4, 0)
        core = FeatherMask.execute(mask, 0, 0, 4, 0).result[0]
        self.assertEqual(float(core[0, 5, 13]), 1.0)
        self.assertEqual(float(boundary[0, 5, 13]), 0.25)
        self.assertEqual(float(boundary[0, 5, 10]), 1.0)

    def test_exact_nonzero_bbox_includes_tiny_positive_values(self):
        mask = torch.zeros((1, 6, 9))
        mask[0, 2, 2] = torch.finfo(mask.dtype).tiny
        mask[0, 2:5, 5:8] = 1
        actual = self.assert_reference(mask, 2, 0, 0, 0)
        self.assertGreater(float(actual[0, 2, 2]), 0.0)

    def test_rank_two_input_preserves_rank_and_canvas(self):
        mask = torch.zeros((8, 10))
        mask[2:7, 3:9] = 1
        actual = self.assert_reference(mask, 2, 1, 3, 4)
        self.assertEqual(tuple(actual.shape), (8, 10))

    def test_invalid_input_and_widths_fail_clearly(self):
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            MODULE.feather_mask_from_boundary([[1.0]], 0, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, r"\[H,W\]"):
            MODULE.feather_mask_from_boundary(torch.ones((1, 1, 2, 3)), 0, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            MODULE.feather_mask_from_boundary(torch.empty((1, 0, 3)), 0, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            MODULE.feather_mask_from_boundary(torch.ones((3, 3)), -1, 0, 0, 0)

    def test_schema(self):
        schema = MODULE.FeatherMaskFromBoundary.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteFeatherMaskFromBoundary")
        self.assertEqual(schema.display_name, "Feather Mask from Boundary")
        self.assertEqual(schema.category, "Utility Suite/Mask")
        self.assertEqual([item.id for item in schema.inputs], ["mask", "left", "top", "right", "bottom"])
        self.assertEqual(schema.outputs[0].io_type, "MASK")


if __name__ == "__main__":
    unittest.main()
