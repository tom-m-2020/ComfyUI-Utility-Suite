from __future__ import annotations

import asyncio
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
    "utility_suite_set_default_mask_segs_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.set_default_mask_segs"]

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)


def segment(crop, label):
    height = crop[3] - crop[1]
    width = crop[2] - crop[0]
    return SEG(
        object(),
        np.zeros((height, width), dtype=np.float32),
        0.75,
        crop,
        tuple(crop),
        label,
        object(),
    )


class SetDefaultMaskForSEGSTests(unittest.TestCase):
    def test_crops_full_canvas_mask_into_every_entry(self):
        mask = torch.arange(30, dtype=torch.float32).reshape(5, 6) / 30
        entries = [segment((0, 0, 3, 2), "a"), segment((2, 1, 6, 5), "b")]

        result = MODULE.set_default_mask_for_segs(((5, 6), entries), mask)

        np.testing.assert_array_equal(result[1][0].cropped_mask, mask[0:2, 0:3].numpy())
        np.testing.assert_array_equal(result[1][1].cropped_mask, mask[1:5, 2:6].numpy())

    def test_preserves_crop_regions_and_all_non_mask_fields(self):
        entries = [segment((1, 1, 4, 3), "a"), segment((0, 0, 2, 2), "b")]
        result = MODULE.set_default_mask_for_segs(((3, 4), entries), torch.ones((1, 3, 4)))
        for original, updated in zip(entries, result[1]):
            self.assertEqual(updated[:1] + updated[2:], original[:1] + original[2:])
            self.assertEqual(updated.crop_region, original.crop_region)

    def test_preserves_soft_values_without_thresholding(self):
        mask = torch.tensor([[0.0, 0.001, 0.5], [0.75, 0.999, 1.0]])
        result = MODULE.set_default_mask_for_segs(
            ((2, 3), [segment((0, 0, 3, 2), "soft")]), mask
        )
        np.testing.assert_array_equal(result[1][0].cropped_mask, mask.numpy())

    def test_empty_segs_remains_empty(self):
        header = (2, 3)
        result = MODULE.set_default_mask_for_segs((header, []), torch.zeros((2, 3)))
        self.assertIs(result[0], header)
        self.assertEqual(result[1], [])

    def test_rejects_batch_and_canvas_mismatch(self):
        with self.assertRaisesRegex(ValueError, r"\[H,W\] or \[1,H,W\]"):
            MODULE.set_default_mask_for_segs(((2, 3), []), torch.zeros((2, 2, 3)))
        with self.assertRaisesRegex(ValueError, "MASK canvas to match"):
            MODULE.set_default_mask_for_segs(((2, 3), []), torch.zeros((2, 4)))

    def test_inputs_are_not_mutated(self):
        entry = segment((0, 0, 3, 2), "a")
        original_mask = entry.cropped_mask.copy()
        mask = torch.full((2, 3), 0.4)
        mask_before = mask.clone()
        MODULE.set_default_mask_for_segs(((2, 3), [entry]), mask)
        np.testing.assert_array_equal(entry.cropped_mask, original_mask)
        torch.testing.assert_close(mask, mask_before)

    def test_schema_and_registration(self):
        schema = MODULE.SetDefaultMaskForSEGS.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteSetDefaultMaskForSEGS")
        self.assertEqual(schema.display_name, "Set Default Mask for SEGS")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual([item.id for item in schema.inputs], ["segs", "mask"])
        self.assertEqual([item.io_type for item in schema.outputs], ["SEGS"])
        self.assertIn(
            MODULE.SetDefaultMaskForSEGS,
            asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list()),
        )


if __name__ == "__main__":
    unittest.main()
