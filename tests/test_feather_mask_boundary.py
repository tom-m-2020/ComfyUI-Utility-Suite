from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional

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

from comfy_extras.nodes_mask import FeatherMask, GrowMask


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


def outward_reference(mask: torch.Tensor, left: int, top: int, right: int, bottom: int) -> torch.Tensor:
    batch = mask.unsqueeze(0) if mask.ndim == 2 else mask
    grow_amount = max(max(left, top, right, bottom) - 1, 0)
    if grow_amount:
        padded = functional.pad(batch, (grow_amount, grow_amount, grow_amount, grow_amount), value=0)
        grown = GrowMask.execute(padded, grow_amount, True).result[0]
    else:
        grown = batch
    feathered = reference(grown, left, top, right, bottom)
    height, width = batch.shape[-2:]
    result = feathered[:, grow_amount : grow_amount + height, grow_amount : grow_amount + width]
    return result[0] if mask.ndim == 2 else result


class FeatherMaskFromBoundaryTests(unittest.TestCase):
    def assert_reference(self, mask, left, top, right, bottom):
        expected = reference(mask, left, top, right, bottom)
        actual = MODULE.feather_mask_from_boundary(mask, left, top, right, bottom)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return actual

    def assert_outward_reference(self, mask, left, top, right, bottom):
        expected = outward_reference(mask, left, top, right, bottom)
        actual = MODULE.feather_mask_from_boundary(mask, left, top, right, bottom, True)
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

    def test_outward_right_moves_ramp_outside_original_boundary(self):
        mask = torch.zeros((1, 15, 24))
        mask[:, 4:11, 5:15] = 1
        inward = MODULE.feather_mask_from_boundary(mask, 0, 0, 4, 0, False)
        outward = self.assert_outward_reference(mask, 0, 0, 4, 0)
        self.assertEqual(inward[0, 7, 11:15].tolist(), [1.0, 0.75, 0.5, 0.25])
        self.assertTrue(torch.equal(outward[0, 7, 5:15], torch.ones(10)))
        torch.testing.assert_close(outward[0, 7, 14:18], torch.tensor([1.0, 0.75, 0.5, 0.25]))
        self.assertEqual(float(outward[0, 7, 18]), 0.0)

    def test_automatic_grow_amount_accounts_for_factor_one_position(self):
        self.assertEqual(MODULE._automatic_grow_amount(0, 0, 0, 0), 0)
        self.assertEqual(MODULE._automatic_grow_amount(1, 0, 0, 0), 0)
        self.assertEqual(MODULE._automatic_grow_amount(0, 0, 4, 0), 3)
        self.assertEqual(MODULE._automatic_grow_amount(10, 20, 100, 40), 99)

    def test_outward_each_side_preserves_original_hard_interior(self):
        widths = ((4, 0, 0, 0), (0, 4, 0, 0), (0, 0, 4, 0), (0, 0, 0, 4))
        for setting in widths:
            with self.subTest(widths=setting):
                mask = torch.zeros((1, 18, 22))
                mask[:, 5:13, 6:16] = 1
                actual = self.assert_outward_reference(mask, *setting)
                self.assertTrue(torch.equal(actual[:, 5:13, 6:16], torch.ones((1, 8, 10))))

    def test_outward_corner_and_all_side_combinations_preserve_core_ramps(self):
        settings = (
            (4, 4, 0, 0),
            (0, 4, 4, 0),
            (0, 0, 4, 4),
            (4, 0, 0, 4),
            (3, 4, 5, 6),
        )
        for widths in settings:
            with self.subTest(widths=widths):
                mask = torch.zeros((1, 24, 28))
                mask[:, 8:16, 9:19] = 1
                actual = self.assert_outward_reference(mask, *widths)
                self.assertTrue(torch.equal(actual[:, 8:16, 9:19], torch.ones((1, 8, 10))))

    def test_outward_asymmetric_sides_use_maximum_and_keep_individual_feathers(self):
        mask = torch.zeros((1, 260, 320))
        mask[:, 110:150, 130:190] = 1
        actual = self.assert_outward_reference(mask, 0, 20, 100, 40)
        self.assertTrue(torch.equal(actual[:, 110:150, 130:190], torch.ones((1, 40, 60))))
        self.assertEqual(float(actual[0, 130, 90]), 1.0)
        self.assertAlmostEqual(float(actual[0, 130, 288]), 0.01)
        self.assertAlmostEqual(float(actual[0, 11, 160]), 0.05)
        self.assertAlmostEqual(float(actual[0, 248, 160]), 0.025)

    def test_outward_large_asymmetric_widths(self):
        mask = torch.zeros((1, 40, 500))
        mask[:, 15:25, 220:280] = 1
        actual = self.assert_outward_reference(mask, 10, 0, 200, 0)
        self.assertTrue(torch.equal(actual[:, 15:25, 220:280], torch.ones((1, 10, 60))))
        self.assertAlmostEqual(float(actual[0, 20, 478]), 0.005)
        self.assertAlmostEqual(float(actual[0, 20, 21]), 0.1)

    def test_outward_zero_feather_side_retains_omnidirectional_grown_support(self):
        mask = torch.zeros((1, 20, 24))
        mask[:, 7:13, 8:16] = 1
        actual = self.assert_outward_reference(mask, 0, 0, 4, 0)
        self.assertEqual(float(actual[0, 10, 5]), 1.0)
        self.assertEqual(float(actual[0, 4, 12]), 1.0)
        self.assertEqual(float(actual[0, 15, 12]), 1.0)

    def test_outward_padding_preserves_boundary_when_content_touches_canvas_edge(self):
        mask = torch.zeros((1, 12, 20))
        mask[:, 3:9, 11:20] = 1
        actual = self.assert_outward_reference(mask, 0, 0, 4, 0)
        self.assertTrue(torch.equal(actual[:, 3:9, 11:20], torch.ones((1, 6, 9))))
        self.assertEqual(float(actual[0, 6, 19]), 1.0)

    def test_outward_soft_mask_preserves_grayscale_grow_without_normalizing(self):
        mask = torch.zeros((1, 16, 24))
        mask[:, 5:11, 7:15] = 0.7
        original = mask.clone()
        actual = self.assert_outward_reference(mask, 0, 0, 4, 0)
        self.assertAlmostEqual(float(actual.max()), 0.7)
        self.assertTrue(torch.allclose(actual[:, 5:11, 7:15], torch.full((1, 6, 8), 0.7)))
        self.assertAlmostEqual(float(actual[0, 8, 15]), 0.525)
        torch.testing.assert_close(mask, original, rtol=0, atol=0)

    def test_outward_batch_uses_independent_boundaries(self):
        mask = torch.zeros((2, 20, 26))
        mask[0, 3:9, 4:11] = 1
        mask[1, 11:17, 15:22] = 0.6
        self.assert_outward_reference(mask, 3, 2, 5, 4)

    def test_invalid_input_and_widths_fail_clearly(self):
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            MODULE.feather_mask_from_boundary([[1.0]], 0, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, r"\[H,W\]"):
            MODULE.feather_mask_from_boundary(torch.ones((1, 1, 2, 3)), 0, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            MODULE.feather_mask_from_boundary(torch.empty((1, 0, 3)), 0, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            MODULE.feather_mask_from_boundary(torch.ones((3, 3)), -1, 0, 0, 0)
        with self.assertRaisesRegex(TypeError, "outward"):
            MODULE.feather_mask_from_boundary(torch.ones((3, 3)), 0, 0, 0, 0, 1)

    def test_schema(self):
        schema = MODULE.FeatherMaskFromBoundary.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteFeatherMaskFromBoundary")
        self.assertEqual(schema.display_name, "Feather Mask from Boundary")
        self.assertEqual(schema.category, "Utility Suite/Mask")
        self.assertEqual(
            [item.id for item in schema.inputs], ["mask", "left", "top", "right", "bottom", "outward"]
        )
        self.assertFalse(schema.inputs[-1].default)
        self.assertEqual(schema.outputs[0].io_type, "MASK")


if __name__ == "__main__":
    unittest.main()
