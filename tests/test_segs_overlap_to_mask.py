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
    "utility_suite_segs_overlap_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.segs_overlap_to_mask"]
MASK_TILE = sys.modules[f"{SPEC.name}.mask_tile_segs"]
REORDER = sys.modules[f"{SPEC.name}.reorder_segs"]
PREVIEW = sys.modules[f"{SPEC.name}.preview_segs_regions"]

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)


def segment(crop, mask=None, image=None, label=""):
    x1, y1, x2, y2 = crop
    if mask is None:
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
    return SEG(image, mask, 0.7, crop, (x1 + 1, y1 + 1, x2 - 1, y2 - 1), label, object())


def derive(entries, mode="all_segs", cap=0, keep=False, invert=False, shape=(40, 40)):
    return MODULE.segs_overlap_to_mask((shape, entries), keep, mode, cap, invert)


def rows(lines):
    return np.array([[float(character) for character in line] for line in lines], dtype=np.float32)


class SEGSOverlapToMASKTests(unittest.TestCase):
    def test_horizontal_overlap_all_is_symmetric(self):
        first = segment((0, 0, 8, 4))
        second = segment((5, 0, 12, 4))
        header, entries = derive([first, second])
        self.assertEqual(header, (40, 40))
        np.testing.assert_array_equal(entries[0].cropped_mask[:, 5:8], 1)
        np.testing.assert_array_equal(entries[1].cropped_mask[:, 0:3], 1)
        self.assertEqual(int(entries[0].cropped_mask.sum()), 12)
        self.assertEqual(int(entries[1].cropped_mask.sum()), 12)

    def test_previous_is_asymmetric_at_same_source_pixels(self):
        first = segment((0, 0, 8, 4))
        second = segment((5, 0, 12, 4))
        _, entries = derive([first, second], mode="previous_segs")
        self.assertFalse(entries[0].cropped_mask.any())
        self.assertTrue(np.all(entries[1].cropped_mask[:, :3] == 1))

    def test_vertical_overlap(self):
        first = segment((0, 0, 4, 8))
        second = segment((0, 5, 4, 12))
        _, entries = derive([first, second])
        self.assertEqual(int(entries[0].cropped_mask.sum()), 12)
        np.testing.assert_array_equal(entries[1].cropped_mask[:3, :], 1)

    def test_exclusive_end_touch_has_no_overlap(self):
        _, entries = derive([segment((0, 0, 4, 4)), segment((4, 0, 8, 4))])
        self.assertFalse(any(entry.cropped_mask.any() for entry in entries))

    def test_four_way_grid_union_is_hard(self):
        crops = [(0, 0, 6, 6), (4, 0, 10, 6), (0, 4, 6, 10), (4, 4, 10, 10)]
        _, entries = derive([segment(crop) for crop in crops])
        self.assertEqual(entries[0].cropped_mask[4, 4], 1)
        self.assertEqual(entries[0].cropped_mask[0, 0], 0)
        for entry in entries:
            self.assertEqual(set(np.unique(entry.cropped_mask)), {0.0, 1.0})

    def test_cap_exact_below_above_and_unlimited(self):
        for overlap in (3, 4, 5, 12):
            first = segment((0, 0, 16, 4))
            second = segment((16 - overlap, 0, 26, 4))
            _, capped = derive([first, second], cap=4)
            _, unlimited = derive([first, second], cap=0)
            self.assertEqual(int(capped[0].cropped_mask.sum()), 4 * min(overlap, 4))
            self.assertEqual(int(unlimited[0].cropped_mask.sum()), 4 * overlap)

    def test_current_edge_cap_coordinates(self):
        self.assertEqual(MODULE._capped_overlap((0, 0, 12, 4), (8, 0, 20, 4), 2), (10, 0, 12, 4))
        self.assertEqual(MODULE._capped_overlap((0, 0, 12, 4), (8, 0, 20, 4), 3), (9, 0, 12, 4))
        self.assertEqual(MODULE._capped_overlap((0, 0, 12, 20), (8, 0, 20, 20), 2), (10, 0, 12, 20))

    def test_exact_horizontal_current_edge_discriminator(self):
        a = (100, 0, 600, 500)
        b = (500, 0, 1000, 500)
        self.assertEqual(MODULE._capped_overlap(a, b, 20), (580, 0, 600, 500))
        self.assertEqual(MODULE._capped_overlap(b, a, 20), (500, 0, 520, 500))
        self.assertNotEqual(MODULE._capped_overlap(a, b, 20), (540, 0, 560, 500))

    def test_exact_vertical_current_edge_discriminator(self):
        a = (0, 100, 500, 600)
        b = (0, 500, 500, 1000)
        self.assertEqual(MODULE._capped_overlap(a, b, 20), (0, 580, 500, 600))
        self.assertEqual(MODULE._capped_overlap(b, a, 20), (0, 500, 500, 520))

    def test_large_left_right_top_bottom_cap(self):
        current = (100, 100, 300, 300)
        self.assertEqual(MODULE._capped_overlap(current, (0, 100, 200, 300), 20), (100, 100, 120, 300))
        self.assertEqual(MODULE._capped_overlap(current, (200, 100, 400, 300), 20), (280, 100, 300, 300))
        self.assertEqual(MODULE._capped_overlap(current, (100, 0, 300, 200), 20), (100, 100, 300, 120))
        self.assertEqual(MODULE._capped_overlap(current, (100, 200, 300, 400), 20), (100, 280, 300, 300))

    def test_vertical_and_corner_capping(self):
        self.assertEqual(MODULE._capped_overlap((0, 0, 10, 10), (0, 6, 10, 16), 2), (0, 8, 10, 10))
        self.assertEqual(MODULE._capped_overlap((0, 0, 10, 10), (6, 6, 16, 16), 2), (8, 8, 10, 10))

    def test_all_four_corner_anchors(self):
        current = (100, 100, 200, 200)
        self.assertEqual(MODULE._capped_overlap(current, (50, 50, 150, 150), 20), (100, 100, 120, 120))
        self.assertEqual(MODULE._capped_overlap(current, (150, 50, 250, 150), 20), (180, 100, 200, 120))
        self.assertEqual(MODULE._capped_overlap(current, (50, 150, 150, 250), 20), (100, 180, 120, 200))
        self.assertEqual(MODULE._capped_overlap(current, (150, 150, 250, 250), 20), (180, 180, 200, 200))

    def test_containment_is_deterministic_and_order_independent(self):
        outer = (0, 0, 12, 12)
        inner = (2, 3, 10, 11)
        self.assertEqual(MODULE._capped_overlap(outer, inner, 2), inner)
        self.assertEqual(MODULE._capped_overlap(inner, outer, 2), inner)

    def test_fully_internal_axis_is_not_midpoint_capped(self):
        current = (0, 0, 20, 20)
        other = (5, 0, 15, 12)
        self.assertEqual(MODULE._capped_overlap(current, other, 2), (5, 0, 15, 2))

    def test_pairwise_caps_precede_union(self):
        current = segment((0, 0, 20, 4))
        left = segment((0, 0, 12, 4))
        right = segment((8, 0, 20, 4))
        _, all_entries = derive([current, left, right], cap=4)
        self.assertEqual(int(all_entries[0].cropped_mask.sum()), 32)
        self.assertTrue(np.all(all_entries[0].cropped_mask[:, 0:4] == 1))
        self.assertTrue(np.all(all_entries[0].cropped_mask[:, 16:20] == 1))
        self.assertFalse(all_entries[0].cropped_mask[:, 4:16].any())
        # Capping the merged 0:20 union has no unique entering edge.

    def test_all_segs_pair_masks_are_current_relative(self):
        a = segment((100, 0, 600, 500))
        b = segment((500, 0, 1000, 500))
        _, entries = derive([a, b], cap=20, shape=(500, 1000))
        self.assertTrue(np.all(entries[0].cropped_mask[:, 480:500] == 1))
        self.assertFalse(entries[0].cropped_mask[:, 400:480].any())
        self.assertTrue(np.all(entries[1].cropped_mask[:, 0:20] == 1))
        self.assertFalse(entries[1].cropped_mask[:, 20:100].any())

    def test_previous_segs_keeps_edge_cap_and_list_asymmetry(self):
        a = segment((100, 0, 600, 500))
        b = segment((500, 0, 1000, 500))
        _, entries = derive([a, b], mode="previous_segs", cap=20, shape=(500, 1000))
        self.assertFalse(entries[0].cropped_mask.any())
        self.assertTrue(np.all(entries[1].cropped_mask[:, :20] == 1))

    def test_current_order_controls_previous(self):
        a = segment((0, 0, 8, 4), label="a")
        b = segment((5, 0, 12, 4), label="b")
        _, forward = derive([a, b], mode="previous_segs")
        _, backward = derive([b, a], mode="previous_segs")
        self.assertFalse(forward[0].cropped_mask.any())
        self.assertFalse(backward[0].cropped_mask.any())
        self.assertEqual(int(forward[1].cropped_mask.sum()), 12)
        self.assertEqual(int(backward[1].cropped_mask.sum()), 12)

    def test_all_four_no_overlap_combinations(self):
        original = np.full((4, 4), 0.7, dtype=np.float32)
        item = segment((0, 0, 4, 4), original)
        for keep, invert, expected in (
            (False, False, 0), (False, True, 1), (True, False, 0), (True, True, 0.7)
        ):
            _, entries = derive([item], keep=keep, invert=invert)
            np.testing.assert_allclose(entries[0].cropped_mask, expected, atol=1e-7)

    def test_keep_original_examples_a_and_b(self):
        original = rows(["11111000"] * 8)
        generated = rows(["00000011"] * 6 + ["11111111"] * 2)
        expected_a = rows(["00000000"] * 6 + ["11111000"] * 2)
        expected_b = rows(["11111000"] * 6 + ["00000000"] * 2)
        np.testing.assert_array_equal(MODULE._keep_original(generated, original), expected_a)
        np.testing.assert_array_equal(MODULE._keep_original(1 - generated, original), expected_b)

    def test_keep_original_example_c(self):
        original = rows(["11111110"] * 7 + ["00000000"])
        generated = rows(["11111000"] * 8)
        expected = rows(["11111000"] * 7 + ["00000000"])
        np.testing.assert_array_equal(MODULE._keep_original(generated, original), expected)

    def test_grayscale_subtraction_not_multiplication(self):
        original = np.array([[0.7, 0.7]], dtype=np.float32)
        generated = np.array([[1.0, 0.5]], dtype=np.float32)
        np.testing.assert_allclose(MODULE._keep_original(generated, original), [[0.7, 0.2]], atol=1e-7)

    def test_keep_original_uses_each_original_mask_and_does_not_mutate(self):
        first_mask = np.full((4, 8), 0.25, dtype=np.float32)
        second_mask = torch.full((4, 7), 0.75)
        first = segment((0, 0, 8, 4), first_mask)
        second = segment((5, 0, 12, 4), second_mask)
        _, entries = derive([first, second], keep=True)
        np.testing.assert_allclose(entries[0].cropped_mask[:, 5:8], 0.25)
        np.testing.assert_allclose(entries[1].cropped_mask[:, :3], 0.75)
        self.assertIs(first.cropped_mask, first_mask)
        self.assertIs(second.cropped_mask, second_mask)
        np.testing.assert_array_equal(first_mask, 0.25)
        self.assertTrue(torch.all(second_mask == 0.75))

    def test_original_masks_are_ignored_when_keep_false(self):
        first = segment((0, 0, 8, 4), mask=None)
        second = segment((5, 0, 12, 4), mask=None)
        first = first._replace(cropped_mask=None)
        _, entries = derive([first, second])
        self.assertEqual(int(entries[0].cropped_mask.sum()), 12)

    def test_fields_and_image_references_are_preserved(self):
        image = torch.rand((1, 4, 8, 3))
        first = segment((0, 0, 8, 4), image=image)
        second = segment((5, 0, 12, 4), image=None)
        _, entries = derive([first, second])
        self.assertIs(entries[0].cropped_image, image)
        self.assertIsNone(entries[1].cropped_image)
        self.assertIsNot(entries[0], first)
        self.assertIsNot(entries[0].cropped_mask, first.cropped_mask)
        for old, new in zip([first, second], entries):
            for field in SEG._fields:
                if field != "cropped_mask":
                    self.assertIs(getattr(new, field), getattr(old, field))

    def test_empty_input_and_header_identity(self):
        header = [20, 30]
        result = MODULE.segs_overlap_to_mask((header, []), False, "all_segs", 0, False)
        self.assertIs(result[0], header)
        self.assertEqual(result[1], [])

    def test_invalid_modes_cap_crops_and_masks(self):
        valid = segment((0, 0, 4, 4))
        with self.assertRaisesRegex(ValueError, "compare_mode"):
            derive([valid], mode="bad")
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            derive([valid], cap=-1)
        with self.assertRaisesRegex(ValueError, "outside source canvas"):
            derive([segment((0, 0, 50, 4))])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            derive([segment((0, 0, 0, 4))])
        with self.assertRaisesRegex(ValueError, "does not match crop_region"):
            derive([valid._replace(cropped_mask=np.zeros((3, 4)))], keep=True)
        with self.assertRaisesRegex(TypeError, "cropped_mask"):
            derive([valid._replace(cropped_mask=None)], keep=True)

    def test_singleton_mask_form(self):
        first = segment((0, 0, 8, 4), np.ones((1, 4, 8), dtype=np.float32))
        second = segment((5, 0, 12, 4))
        _, entries = derive([first, second], keep=True)
        self.assertEqual(entries[0].cropped_mask.shape, (4, 8))

    def test_real_tile_output_reorder_and_preview_interoperability(self):
        mask = torch.ones((1, 8, 12), dtype=torch.float32)
        original, _ = MASK_TILE.mask_to_tile_segs(mask, "fixed_tile", 8, 8, 4, 0, 4, 1, 0, 0, 1, 1)
        reordered = REORDER.reorder_segs(original, "clockwise")
        result = MODULE.segs_overlap_to_mask(reordered, False, "previous_segs", 0, False)
        self.assertIs(result[0], original[0])
        self.assertEqual(len(result[1]), len(original[1]))
        self.assertFalse(result[1][0].cropped_mask.any())
        self.assertTrue(result[1][1].cropped_mask.any())
        preview = PREVIEW.preview_segs_regions(result, "mask", 1)
        self.assertEqual(tuple(preview.shape), (1, 8, 12, 3))

    def test_schema(self):
        schema = MODULE.SEGSOverlapToMASK.define_schema()
        self.assertEqual(schema.display_name, "SEGS Overlap to MASK")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual([entry.io_type for entry in schema.inputs], ["SEGS", "BOOLEAN", "COMBO", "INT", "BOOLEAN"])
        self.assertEqual([entry.io_type for entry in schema.outputs], ["SEGS"])


if __name__ == "__main__":
    unittest.main()
