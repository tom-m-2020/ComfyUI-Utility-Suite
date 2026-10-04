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
    "utility_suite_bridge_masks_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.bridge_masks"]


def rectangle(bbox, shape=(24, 30), value=1.0):
    x1, y1, x2, y2 = bbox
    mask = torch.zeros(shape, dtype=torch.float32)
    mask[y1:y2, x1:x2] = value
    return mask


class BridgeMasksTests(unittest.TestCase):
    def test_schema(self):
        schema = MODULE.BridgeMasks.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteBridgeMasks")
        self.assertEqual(schema.display_name, "Bridge Masks")
        self.assertEqual(schema.category, "Utility Suite/Mask")
        self.assertEqual([item.id for item in schema.inputs], ["mask_a", "mask_b", "mode"])
        self.assertEqual(schema.inputs[2].options, ["convex_hull", "gap"])
        self.assertEqual([output.io_type for output in schema.outputs], ["MASK"])

    def test_horizontal_and_vertical_separation(self):
        cases = (
            ((2, 5, 6, 10), (13, 5, 18, 10), (8, 7)),
            ((7, 2, 12, 6), (7, 13, 12, 18), (9, 9)),
        )
        for bbox_a, bbox_b, bridge_pixel in cases:
            with self.subTest(bbox_a=bbox_a, bbox_b=bbox_b):
                output = MODULE.bridge_masks(rectangle(bbox_a), rectangle(bbox_b), "gap")
                self.assertEqual(float(output[bridge_pixel[1], bridge_pixel[0]]), 1.0)

    def test_diagonal_different_sizes_produce_polygonal_hull(self):
        bbox_a = (2, 2, 6, 7)
        bbox_b = (15, 12, 24, 18)
        vertices = MODULE._hull_vertices(bbox_a, bbox_b)
        self.assertGreater(len(vertices), 4)
        output = MODULE.bridge_masks(rectangle(bbox_a), rectangle(bbox_b), "convex_hull")
        self.assertEqual(float(output[9, 11]), 1.0)
        self.assertEqual(float(output[2, 20]), 0.0)

    def test_irregular_endpoints_distinguish_convex_hull_from_gap(self):
        mask_a = torch.zeros((24, 30))
        mask_a[2, 2:7] = 0.4
        mask_a[2:8, 2] = 0.7
        mask_a[7, 6] = 0.6
        mask_b = torch.zeros_like(mask_a)
        mask_b[13, 17:25] = 0.3
        mask_b[13:20, 24] = 0.8
        mask_b[19, 17] = 0.5

        hull = MODULE.bridge_masks(mask_a, mask_b, "convex_hull")
        gap = MODULE.bridge_masks(mask_a, mask_b, "gap")
        self.assertEqual(float(hull[5, 5]), 1.0)
        self.assertEqual(float(gap[5, 5]), 0.0)
        self.assertAlmostEqual(float(gap[2, 2]), 0.7)
        self.assertEqual(float(gap[16, 20]), 0.0)
        self.assertEqual(float(gap[10, 13]), 1.0)

    def test_geometry_uses_bboxes_not_detailed_contours(self):
        bbox_a = (2, 2, 8, 8)
        bbox_b = (18, 12, 26, 20)
        first_a = rectangle(bbox_a)
        first_b = rectangle(bbox_b)
        second_a = torch.zeros_like(first_a)
        second_b = torch.zeros_like(first_b)
        second_a[2, 2] = second_a[7, 7] = 1
        second_b[12, 18] = second_b[19, 25] = 1
        first = MODULE.bridge_masks(first_a, first_b, "convex_hull")
        second = MODULE.bridge_masks(second_a, second_b, "convex_hull")
        self.assertTrue(torch.equal(first, second))

    def test_touching_boxes_have_no_artificial_gap_bridge(self):
        mask_a = rectangle((2, 3, 6, 9), value=0.4)
        mask_b = rectangle((6, 3, 11, 9), value=0.8)
        expected = torch.maximum(mask_a, mask_b)
        actual = MODULE.bridge_masks(mask_a, mask_b, "gap")
        self.assertTrue(torch.equal(actual, expected))

    def test_overlapping_contained_and_identical_boxes_have_no_gap_bridge(self):
        cases = (
            ((2, 3, 10, 11), (7, 7, 15, 14)),
            ((2, 2, 18, 18), (6, 6, 10, 10)),
            ((4, 5, 13, 16), (4, 5, 13, 16)),
        )
        for bbox_a, bbox_b in cases:
            with self.subTest(bbox_a=bbox_a, bbox_b=bbox_b):
                mask_a = rectangle(bbox_a, value=0.35)
                mask_b = rectangle(bbox_b, value=0.75)
                expected = torch.maximum(mask_a, mask_b)
                self.assertTrue(torch.equal(MODULE.bridge_masks(mask_a, mask_b, "gap"), expected))

    def test_canvas_edges_and_exclusive_end_coordinates(self):
        mask_a = rectangle((0, 0, 3, 4), shape=(12, 14))
        mask_b = rectangle((11, 8, 14, 12), shape=(12, 14))
        output = MODULE.bridge_masks(mask_a, mask_b, "convex_hull")
        self.assertEqual(tuple(output.shape), (12, 14))
        self.assertEqual(float(output[0, 0]), 1.0)
        self.assertEqual(float(output[11, 13]), 1.0)

    def test_one_pixel_wide_and_high_boxes(self):
        vertical = rectangle((3, 2, 4, 9), shape=(14, 18))
        horizontal = rectangle((10, 11, 16, 12), shape=(14, 18))
        output = MODULE.bridge_masks(vertical, horizontal, "convex_hull")
        self.assertEqual(float(output[2, 3]), 1.0)
        self.assertEqual(float(output[11, 15]), 1.0)
        self.assertEqual(float(output[8, 8]), 1.0)

    def test_gap_generated_pixels_are_hard_and_soft_sources_are_preserved(self):
        mask_a = rectangle((2, 4, 6, 9), value=0.2)
        mask_b = rectangle((14, 5, 19, 10), value=0.85)
        output = MODULE.bridge_masks(mask_a, mask_b, "gap")
        self.assertEqual(float(output[5, 3]), float(torch.tensor(0.2)))
        self.assertEqual(float(output[6, 16]), float(torch.tensor(0.85)))
        generated = output[(mask_a == 0) & (mask_b == 0)]
        self.assertTrue(set(torch.unique(generated).tolist()).issubset({0.0, 1.0}))
        self.assertIn(1.0, torch.unique(generated).tolist())

    def test_empty_mismatch_rank_and_mode_validation(self):
        valid = rectangle((2, 2, 5, 5))
        with self.assertRaisesRegex(ValueError, "no nonzero bounding box"):
            MODULE.bridge_masks(torch.zeros_like(valid), valid, "gap")
        with self.assertRaisesRegex(ValueError, "matching canvas"):
            MODULE.bridge_masks(valid, torch.ones((10, 10)), "gap")
        with self.assertRaisesRegex(ValueError, r"must be \[H,W\] or \[B,H,W\]"):
            MODULE.bridge_masks(valid.unsqueeze(0).unsqueeze(0), valid, "gap")
        with self.assertRaisesRegex(ValueError, "expected one of"):
            MODULE.bridge_masks(valid, valid, "other")

    def test_batch_pairing_equal_and_singleton_broadcast(self):
        a0 = rectangle((1, 2, 4, 6))
        a1 = rectangle((2, 12, 6, 17))
        b0 = rectangle((12, 2, 16, 6))
        b1 = rectangle((14, 12, 20, 18))
        equal = MODULE.bridge_masks(torch.stack((a0, a1)), torch.stack((b0, b1)), "gap")
        self.assertEqual(tuple(equal.shape), (2, 24, 30))
        singleton = MODULE.bridge_masks(a0.unsqueeze(0), torch.stack((b0, b1)), "gap")
        self.assertEqual(tuple(singleton.shape), (2, 24, 30))
        with self.assertRaisesRegex(ValueError, "equal batch sizes or a singleton"):
            MODULE.bridge_masks(torch.stack((a0, a1)), torch.stack((b0, b1, b0)), "gap")

    def test_inputs_are_not_mutated(self):
        mask_a = rectangle((2, 2, 6, 7), value=0.3)
        mask_b = rectangle((15, 12, 24, 18), value=0.9)
        snapshot_a, snapshot_b = mask_a.clone(), mask_b.clone()
        MODULE.bridge_masks(mask_a, mask_b, "gap")
        self.assertTrue(torch.equal(mask_a, snapshot_a))
        self.assertTrue(torch.equal(mask_b, snapshot_b))


if __name__ == "__main__":
    unittest.main()
