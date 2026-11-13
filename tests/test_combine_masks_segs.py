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
    "utility_suite_combine_masks_segs_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.combine_masks_segs"]

from comfy_extras.nodes_mask import MaskComposite

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)


def segment(index, mask, *, crop=None, image=None, label=None):
    height, width = tuple(mask.shape[-2:])
    if crop is None:
        x1 = index * 10
        crop = (x1, 0, x1 + width, height)
    return SEG(image, mask, 0.25 + index, crop, (99, 98, 101, 102), label or f"seg-{index}", object())


def destination(count=3, height=4, width=5):
    entries = []
    for index in range(count):
        mask = np.full((height, width), 0.2 + index * 0.2, dtype=np.float32)
        entries.append(segment(index, mask))
    return (height, max(1, count) * 10), entries


def masks(result):
    return [entry.cropped_mask for entry in result[1]]


class CombineMasksSEGSTests(unittest.TestCase):
    def test_local_helper_matches_core_all_operations_with_offset_and_clipping(self):
        destination_mask = torch.tensor(
            [[0.0, 0.2, 0.5, 0.8, 1.0], [1.0, 0.8, 0.5, 0.2, 0.0], [0.1, 0.4, 0.6, 0.9, 0.3]]
        )
        source = torch.tensor([[0.9, 0.6, 0.5, 0.4], [0.2, 0.8, 1.0, 0.0], [1.0, 0.3, 0.7, 0.1]])
        for operation in MODULE.OPERATIONS:
            expected = MaskComposite.execute(destination_mask, source, 3, 1, operation).result[0]
            actual = MODULE.mask_composite(destination_mask, source, 3, 1, operation)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_full_canvas_mask_is_sliced_by_each_crop_region(self):
        segs = destination(3)
        source = torch.zeros((4, 30))
        source[:, 0:5] = 0.1
        source[:, 10:15] = 0.5
        source[:, 20:25] = 0.9

        result = MODULE.combine_segs_and_mask(segs, source, 0, 0, "multiply")

        np.testing.assert_allclose(result[1][0].cropped_mask, 0.02)
        np.testing.assert_allclose(result[1][1].cropped_mask, 0.2)
        np.testing.assert_allclose(result[1][2].cropped_mask, 0.54)

    def test_global_offset_positions_mask_on_source_canvas(self):
        crop = (10, 5, 15, 9)
        entry = segment(0, np.full((4, 5), 0.8, dtype=np.float32), crop=crop)
        source = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

        result = MODULE.combine_segs_and_mask(((20, 30), [entry]), source, 12, 6, "subtract")

        expected = np.full((4, 5), 0.8, dtype=np.float32)
        expected[1:3, 2:4] -= source.numpy()
        np.testing.assert_allclose(result[1][0].cropped_mask, expected)

    def test_mask_larger_than_canvas_is_clipped_by_intersections(self):
        segs = destination(1)
        source = torch.full((20, 20), 0.5)
        result = MODULE.combine_segs_and_mask(segs, source, 0, 0, "multiply")
        np.testing.assert_allclose(result[1][0].cropped_mask, 0.1)

    def test_nonoverlapping_global_mask_leaves_seg_unchanged(self):
        segs = destination(1)
        original = segs[1][0].cropped_mask.copy()
        result = MODULE.combine_segs_and_mask(segs, torch.zeros((2, 2)), 8, 8, "subtract")
        np.testing.assert_array_equal(result[1][0].cropped_mask, original)

    def test_execute_accepts_one_mask_without_list_transport(self):
        segs = destination(1)
        source = torch.full((4, 5), 0.5)
        result = MODULE.CombineMasksSEGSAndMASK.execute(segs, source, 0, 0, "multiply").result[0]
        np.testing.assert_allclose(result[1][0].cropped_mask, 0.1)

    def test_segs_source_pairs_by_order_and_reuses_last(self):
        dest = destination(4)
        source_entries = [
            segment(0, np.zeros((4, 5), dtype=np.float32), crop=(20, 10, 25, 14), label="source-z"),
            segment(1, np.ones((4, 5), dtype=np.float32), crop=(0, 0, 5, 4), label="source-o"),
        ]
        source = ((100, 100), source_entries)
        result = MODULE.combine_segs_and_segs(dest, source, 0, 0, "multiply")
        np.testing.assert_array_equal(result[1][0].cropped_mask, 0)
        for index in range(1, 4):
            np.testing.assert_allclose(result[1][index].cropped_mask, dest[1][index].cropped_mask)
        self.assertEqual([entry.label for entry in result[1]], [entry.label for entry in dest[1]])

    def test_segs_source_surplus_is_ignored_and_header_metadata_irrelevant(self):
        dest = destination(1)
        source_entries = [segment(index, np.full((2, 3), value, dtype=np.float32)) for index, value in enumerate((0.5, 0.7, 0.9))]
        result = MODULE.combine_segs_and_segs(dest, ((50, 50), source_entries), 1, 1, "add")
        expected = MaskComposite.execute(
            torch.from_numpy(dest[1][0].cropped_mask), torch.full((2, 3), 0.5), 1, 1, "add"
        ).result[0][0]
        np.testing.assert_array_equal(result[1][0].cropped_mask, expected.numpy())

    def test_empty_destination_preserves_header_even_with_empty_source(self):
        header = [10, 20]
        self.assertIs(MODULE.combine_segs_and_mask((header, []), torch.zeros((1, 1)), 0, 0, "add")[0], header)
        self.assertIs(MODULE.combine_segs_and_segs((header, []), ((1, 1), []), 0, 0, "add")[0], header)

    def test_invalid_mask_fails_clearly_for_nonempty_destination(self):
        dest = destination(1)
        with self.assertRaisesRegex(TypeError, "source MASK"):
            MODULE.combine_segs_and_mask(dest, [], 0, 0, "add")
        with self.assertRaisesRegex(ValueError, "at least one source mask"):
            MODULE.combine_segs_and_segs(dest, ((1, 1), []), 0, 0, "add")

    def test_destination_structure_fields_and_inputs_are_preserved(self):
        header, entries = destination(2)
        image = torch.rand((1, 4, 5, 3))
        entries[0] = entries[0]._replace(cropped_image=image)
        originals = [entry.cropped_mask.copy() for entry in entries]
        source = torch.ones((4, 5))
        result_header, result = MODULE.combine_segs_and_mask((header, entries), source, 0, 0, "multiply")
        self.assertIs(result_header, header)
        self.assertEqual(len(result), len(entries))
        for old, new, original_mask in zip(entries, result, originals):
            self.assertIsNot(old, new)
            np.testing.assert_array_equal(old.cropped_mask, original_mask)
            for field in SEG._fields:
                if field != "cropped_mask":
                    self.assertIs(getattr(new, field), getattr(old, field))

    def test_invalid_masks_offsets_and_operation_fail_clearly(self):
        dest = destination(1)
        with self.assertRaisesRegex(TypeError, "source MASK"):
            MODULE.combine_segs_and_mask(dest, np.zeros((4, 5)), 0, 0, "add")
        with self.assertRaisesRegex(ValueError, r"\[H,W\] or \[1,H,W\]"):
            MODULE.combine_segs_and_mask(dest, torch.zeros((2, 4, 5)), 0, 0, "add")
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            MODULE.combine_segs_and_mask(dest, torch.ones((4, 5)), -1, 0, "add")
        with self.assertRaisesRegex(ValueError, "operation"):
            MODULE.combine_segs_and_mask(dest, torch.ones((4, 5)), 0, 0, "bad")
        malformed = dest[1][0]._replace(cropped_mask=np.zeros((3, 5)))
        with self.assertRaisesRegex(ValueError, "does not match crop_region"):
            MODULE.combine_segs_and_mask((dest[0], [malformed]), torch.ones((4, 5)), 0, 0, "add")

    def test_schema_names_order_and_list_mode(self):
        mask_schema = MODULE.CombineMasksSEGSAndMASK.define_schema()
        segs_schema = MODULE.CombineMasksSEGSAndSEGS.define_schema()
        self.assertEqual(mask_schema.display_name, "Combine Masks (SEGS & MASK)")
        self.assertEqual(segs_schema.display_name, "Combine Masks (SEGS & SEGS)")
        self.assertEqual(mask_schema.category, "Utility Suite/SEGS")
        self.assertFalse(mask_schema.is_input_list)
        self.assertFalse(segs_schema.is_input_list)
        self.assertEqual(mask_schema.inputs[-1].options, list(MODULE.OPERATIONS))
        self.assertEqual(segs_schema.inputs[-1].options, list(MODULE.OPERATIONS))


if __name__ == "__main__":
    unittest.main()
