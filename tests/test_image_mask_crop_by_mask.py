from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_image_mask_crop_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.image_mask_crop_by_mask"]


class ImageAndMaskCropByMaskTests(unittest.TestCase):
    def test_schema_and_registration(self):
        schema = MODULE.ImageAndMaskCropByMask.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteImageAndMaskCropByMask")
        self.assertEqual(schema.display_name, "Image And Mask Crop By Mask")
        self.assertEqual([item.id for item in schema.inputs], ["mask", "image", "mask_to_be_cropped"])
        self.assertFalse(schema.inputs[0].optional)
        self.assertTrue(schema.inputs[1].optional)
        self.assertTrue(schema.inputs[2].optional)
        self.assertEqual([item.id for item in schema.outputs], ["image", "mask"])
        self.assertIn(
            MODULE.ImageAndMaskCropByMask,
            asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list()),
        )

    def test_crops_image_and_mask_to_same_bounds(self):
        crop_mask = torch.zeros(1, 7, 9)
        crop_mask[:, 2:6, 3:8] = 1
        image = torch.arange(7 * 9 * 3).reshape(1, 7, 9, 3)
        target_mask = torch.arange(7 * 9).reshape(1, 7, 9)

        cropped_image, cropped_mask = MODULE.image_and_mask_crop_by_mask(crop_mask, image, target_mask)

        self.assertTrue(torch.equal(cropped_image, image[:, 2:6, 3:8, :]))
        self.assertTrue(torch.equal(cropped_mask, target_mask[:, 2:6, 3:8]))

    def test_each_optional_target_works_alone(self):
        crop_mask = torch.zeros(1, 5, 6)
        crop_mask[:, 1:4, 2:5] = 1
        image = torch.rand(1, 5, 6, 3)
        target_mask = torch.rand(1, 5, 6)

        cropped_image, absent_mask = MODULE.image_and_mask_crop_by_mask(crop_mask, image=image)
        absent_image, cropped_mask = MODULE.image_and_mask_crop_by_mask(
            crop_mask, mask_to_be_cropped=target_mask
        )

        self.assertEqual(cropped_image.shape, (1, 3, 3, 3))
        self.assertIsNone(absent_mask)
        self.assertIsNone(absent_image)
        self.assertEqual(cropped_mask.shape, (1, 3, 3))

    def test_repeats_last_crop_mask_for_longer_target_batches(self):
        crop_mask = torch.zeros(2, 6, 7)
        crop_mask[0, 1:4, 2:5] = 1
        crop_mask[1, 2:5, 1:4] = 1
        image = torch.rand(3, 6, 7, 3)
        result, _ = MODULE.image_and_mask_crop_by_mask(crop_mask, image=image)
        self.assertEqual(result.shape, (3, 3, 3, 3))
        self.assertTrue(torch.equal(result[2], image[2, 2:5, 1:4]))

    def test_preserves_kjnodes_rounded_mask_threshold(self):
        crop_mask = torch.zeros(1, 5, 6)
        crop_mask[0, 0, 0] = 0.5
        crop_mask[0, 2:4, 3:5] = 0.5001
        image = torch.rand(1, 5, 6, 3)
        result, _ = MODULE.image_and_mask_crop_by_mask(crop_mask, image=image)
        self.assertEqual(result.shape, (1, 2, 2, 3))
        self.assertTrue(torch.equal(result, image[:, 2:4, 3:5, :]))

    def test_rejects_missing_targets_empty_mask_and_mismatched_canvas(self):
        mask = torch.ones(1, 4, 5)
        with self.assertRaisesRegex(ValueError, "requires image or mask_to_be_cropped"):
            MODULE.image_and_mask_crop_by_mask(mask)
        with self.assertRaisesRegex(ValueError, "is empty"):
            MODULE.image_and_mask_crop_by_mask(torch.zeros(1, 4, 5), image=torch.rand(1, 4, 5, 3))
        with self.assertRaisesRegex(ValueError, "must match crop mask canvas"):
            MODULE.image_and_mask_crop_by_mask(mask, image=torch.rand(1, 3, 5, 3))

    def test_rejects_different_per_item_crop_sizes(self):
        mask = torch.zeros(2, 6, 7)
        mask[0, 1:3, 1:3] = 1
        mask[1, 1:4, 1:4] = 1
        with self.assertRaisesRegex(ValueError, "equal crop sizes"):
            MODULE.image_and_mask_crop_by_mask(mask, image=torch.rand(2, 6, 7, 3))


if __name__ == "__main__":
    unittest.main()
