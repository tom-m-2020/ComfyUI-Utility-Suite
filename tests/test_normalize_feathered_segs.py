from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_normalize_feathered_segs_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.normalize_feathered_segs"]

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)


def segment(crop, mask, label):
    return SEG(object(), mask, 0.75, crop, tuple(crop), label, object())


def combined(segs):
    (height, width), entries = segs
    result = np.zeros((height, width), dtype=np.float32)
    for entry in entries:
        x1, y1, x2, y2 = entry.crop_region
        result[y1:y2, x1:x2] += entry.cropped_mask
    return result


def reference_envelope(segs):
    (height, width), entries = segs
    result = np.zeros((height, width), dtype=np.float32)
    for entry in entries:
        x1, y1, x2, y2 = entry.crop_region
        np.maximum(result[y1:y2, x1:x2], entry.cropped_mask, out=result[y1:y2, x1:x2])
    return result


class NormalizeFeatheredSEGSTests(unittest.TestCase):
    def test_two_overlapping_feathers_reconstruct_reference_envelope(self):
        crops = [(0, 0, 7, 4), (4, 0, 11, 4)]
        reference = ((4, 11), [segment(crop, np.ones((4, 7), dtype=np.float32), str(i)) for i, crop in enumerate(crops)])
        first = np.ones((4, 7), dtype=np.float32)
        first[:, 4:] = [0.75, 0.5, 0.25]
        second = np.ones((4, 7), dtype=np.float32)
        second[:, :3] = [0.25, 0.5, 0.75]
        feathered = ((4, 11), [segment(crops[0], first, "0"), segment(crops[1], second, "1")])

        result = MODULE.normalize_feathered_segs(feathered, reference)

        np.testing.assert_allclose(combined(result), reference_envelope(reference), rtol=0, atol=1e-7)
        np.testing.assert_allclose(result[1][0].cropped_mask[:, 4:] + result[1][1].cropped_mask[:, :3], 1.0)

    def test_doubled_soft_overlap_is_scaled_back_to_original_amount(self):
        value = np.float32(0.396)
        crop = (0, 0, 4, 3)
        reference_entries = [segment(crop, np.full((3, 4), value, dtype=np.float32), str(i)) for i in range(2)]
        feathered_entries = [segment(crop, np.full((3, 4), value, dtype=np.float32), str(i)) for i in range(2)]

        result = MODULE.normalize_feathered_segs(((3, 4), feathered_entries), ((3, 4), reference_entries))

        self.assertAlmostEqual(float(result[1][0].cropped_mask[1, 1]), float(value / 2))
        self.assertAlmostEqual(float(combined(result)[1, 1]), float(value))

    def test_multirow_bumps_are_removed_while_global_gradient_is_preserved(self):
        width, height = 12, 8
        gradient = np.linspace(1.0, 0.1, width, dtype=np.float32)
        target = np.broadcast_to(gradient, (height, width)).copy()
        crops = [(0, 0, 12, 5), (0, 3, 12, 8)]
        reference = ((height, width), [segment(crop, target[crop[1]:crop[3]].copy(), str(i)) for i, crop in enumerate(crops)])
        feathered = ((height, width), [segment(crop, target[crop[1]:crop[3]].copy(), str(i)) for i, crop in enumerate(crops)])

        result = MODULE.normalize_feathered_segs(feathered, reference)

        np.testing.assert_allclose(combined(result), target, rtol=0, atol=1e-7)
        np.testing.assert_allclose(combined(result)[1], combined(result)[4], rtol=0, atol=1e-7)

    def test_four_way_overlap_is_order_independent_and_never_exceeds_target(self):
        crops = [(0, 0, 4, 4), (2, 0, 6, 4), (0, 2, 4, 6), (2, 2, 6, 6)]
        references = [segment(crop, np.ones((4, 4), dtype=np.float32), str(i)) for i, crop in enumerate(crops)]
        masks = [np.full((4, 4), value, dtype=np.float32) for value in (0.2, 0.4, 0.6, 0.8)]
        feathered = [segment(crop, mask, str(i)) for i, (crop, mask) in enumerate(zip(crops, masks))]
        source = ((6, 6), feathered)
        result = MODULE.normalize_feathered_segs(source, ((6, 6), references))
        reversed_result = MODULE.normalize_feathered_segs(
            ((6, 6), list(reversed(feathered))), ((6, 6), list(reversed(references)))
        )
        np.testing.assert_allclose(combined(result), np.minimum(combined(source), 1.0), rtol=0, atol=1e-7)
        np.testing.assert_allclose(combined(reversed_result), combined(result), rtol=0, atol=1e-7)

    def test_nonoverlapping_masks_below_reference_remain_unchanged(self):
        crops = [(0, 0, 3, 3), (3, 0, 6, 3)]
        reference = ((3, 6), [segment(crop, np.full((3, 3), 0.7, dtype=np.float32), str(i)) for i, crop in enumerate(crops)])
        feathered = ((3, 6), [segment(crop, np.full((3, 3), 0.35, dtype=np.float32), str(i)) for i, crop in enumerate(crops)])
        result = MODULE.normalize_feathered_segs(feathered, reference)
        np.testing.assert_array_equal(combined(result), combined(feathered))
        for original, output in zip(feathered[1], result[1]):
            np.testing.assert_array_equal(output.cropped_mask, original.cropped_mask)

    def test_positive_reference_with_zero_total_feather_remains_zero(self):
        crop = (0, 0, 3, 3)
        feathered = ((3, 3), [segment(crop, np.zeros((3, 3), dtype=np.float32), "f")])
        reference = ((3, 3), [segment(crop, np.ones((3, 3), dtype=np.float32), "r")])
        result = MODULE.normalize_feathered_segs(feathered, reference)
        np.testing.assert_array_equal(result[1][0].cropped_mask, np.zeros((3, 3), dtype=np.float32))

    def test_output_never_exceeds_input_feather_values(self):
        crop = (0, 0, 4, 3)
        feathered = (
            (3, 4),
            [
                segment(crop, np.full((3, 4), 0.2, dtype=np.float32), "a"),
                segment(crop, np.full((3, 4), 0.7, dtype=np.float32), "b"),
            ],
        )
        reference = (
            (3, 4),
            [
                segment(crop, np.full((3, 4), 0.5, dtype=np.float32), "a"),
                segment(crop, np.full((3, 4), 0.5, dtype=np.float32), "b"),
            ],
        )
        result = MODULE.normalize_feathered_segs(feathered, reference)
        for original, output in zip(feathered[1], result[1]):
            self.assertTrue(np.all(output.cropped_mask <= original.cropped_mask))
        np.testing.assert_allclose(combined(result), 0.5, rtol=0, atol=1e-7)

    def test_requires_matching_headers_counts_and_crop_order(self):
        item = segment((0, 0, 2, 2), np.ones((2, 2), dtype=np.float32), "a")
        with self.assertRaisesRegex(ValueError, "matching source shapes"):
            MODULE.normalize_feathered_segs(((2, 2), [item]), ((3, 2), [item]))
        with self.assertRaisesRegex(ValueError, "matching entry counts"):
            MODULE.normalize_feathered_segs(((2, 2), [item]), ((2, 2), []))
        other = segment((1, 0, 2, 2), np.ones((2, 1), dtype=np.float32), "b")
        with self.assertRaisesRegex(ValueError, "matching crop_region"):
            MODULE.normalize_feathered_segs(((2, 2), [item]), ((2, 2), [other]))

    def test_feathered_metadata_and_inputs_are_preserved(self):
        header = (3, 4)
        feather_mask = np.full((3, 4), 0.5, dtype=np.float32)
        reference_mask = np.ones((3, 4), dtype=np.float32)
        feathered = segment((0, 0, 4, 3), feather_mask, "feathered")
        reference = segment((0, 0, 4, 3), reference_mask, "reference")
        result = MODULE.normalize_feathered_segs((header, [feathered]), (header, [reference]))
        self.assertIs(result[0], header)
        self.assertEqual(result[1][0][:1] + result[1][0][2:], feathered[:1] + feathered[2:])
        np.testing.assert_array_equal(feathered.cropped_mask, feather_mask)
        np.testing.assert_array_equal(reference.cropped_mask, reference_mask)

    def test_empty_segs_preserves_feathered_header(self):
        header = (5, 7)
        result = MODULE.normalize_feathered_segs((header, []), ((5, 7), []))
        self.assertIs(result[0], header)
        self.assertEqual(result[1], [])

    def test_schema_and_registration(self):
        schema = MODULE.NormalizeFeatheredSEGS.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteNormalizeFeatheredSEGS")
        self.assertEqual(schema.display_name, "Normalize Feathered SEGS")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual([item.id for item in schema.inputs], ["feathered_segs", "reference_segs"])
        self.assertEqual([item.io_type for item in schema.outputs], ["SEGS"])
        self.assertIn(MODULE.NormalizeFeatheredSEGS, asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list()))


if __name__ == "__main__":
    unittest.main()
