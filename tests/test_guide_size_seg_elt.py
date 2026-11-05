import asyncio
import importlib.util
import math
import sys
import unittest
from collections import namedtuple
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_guide_size_seg_elt_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.guide_size_seg_elt"]

SEG = namedtuple(
    "SEG", "cropped_image cropped_mask confidence crop_region bbox label control_net_wrapper"
)


def segment(image=None, mask=None, crop=(0, 0, 64, 32), bbox=(16, 8, 48, 24)):
    if image is None:
        image = torch.zeros((1, 32, 64, 3))
    if mask is None:
        mask = torch.linspace(0, 1, 32 * 64).reshape(1, 32, 64)
    return SEG(image, mask, 0.75, crop, bbox, "x", object())


def run(seg, **overrides):
    values = {
        "seg_elt": seg,
        "guide_size": 32,
        "guide_size_for": "crop_region",
        "scale_to_guide_size": "scale",
        "max_size": 1024,
    }
    values.update(overrides)
    return MODULE.guide_size_seg_elt(**values)


class GuideSizeSEGELTTests(unittest.TestCase):
    def test_schema_defaults_and_registration(self):
        schema = MODULE.GuideSizeSEGELT.define_schema()
        self.assertEqual(schema.display_name, "Guide Size (SEG_ELT)")
        defaults = {item.id: getattr(item, "default", None) for item in schema.inputs}
        self.assertEqual(defaults["guide_size"], 512)
        self.assertEqual(defaults["guide_size_for"], "bbox")
        self.assertEqual(defaults["scale_to_guide_size"], "upscale")
        self.assertEqual(defaults["max_size"], 1024)
        nodes = asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list())
        self.assertIn(MODULE.GuideSizeSEGELT, nodes)

    def test_crop_and_bbox_reference_geometry(self):
        seg = segment()
        crop_image, crop_mask = run(seg)
        bbox_image, bbox_mask = run(seg, guide_size_for="bbox")
        self.assertEqual(tuple(crop_image.shape[1:3]), (24, 48))
        self.assertEqual(tuple(crop_mask.shape[-2:]), (24, 48))
        self.assertEqual(tuple(bbox_image.shape[1:3]), (48, 88))
        self.assertEqual(tuple(bbox_mask.shape[-2:]), (48, 88))

    def test_all_scaling_modes_and_max_size(self):
        seg = segment()
        self.assertEqual(tuple(run(seg, guide_size=128, scale_to_guide_size="upscale")[0].shape[1:3]), (88, 184))
        skipped_up = run(seg, guide_size=8, scale_to_guide_size="upscale", max_size=8)[0]
        skipped_down = run(seg, guide_size=128, scale_to_guide_size="downscale", max_size=8)[0]
        self.assertIs(skipped_up, seg.cropped_image)
        self.assertIs(skipped_down, seg.cropped_image)
        downscaled = run(seg, guide_size=8, scale_to_guide_size="downscale")[0]
        self.assertEqual(tuple(downscaled.shape[1:3]), (8, 8))
        capped = run(seg, guide_size=1024, max_size=40)[0]
        self.assertLessEqual(math.sqrt(capped.shape[1] * capped.shape[2]), 40)

    def test_none_returns_original_tensors_and_ignores_max_size(self):
        seg = segment()
        image, mask = run(seg, scale_to_guide_size="none", max_size=8)
        self.assertIs(image, seg.cropped_image)
        self.assertIs(mask, seg.cropped_mask)

    def test_overrides_take_precedence_without_mutation(self):
        seg = segment()
        original = tuple(seg)
        image_override = torch.ones((1, 16, 24, 3))
        mask_override = torch.full((1, 16, 24), 0.25)
        image, mask = run(
            seg,
            image=image_override,
            mask=mask_override,
            scale_to_guide_size="none",
        )
        self.assertIs(image, image_override)
        self.assertIs(mask, mask_override)
        self.assertEqual(tuple(seg), original)

    def test_mismatched_image_and_mask_are_resized_to_one_output_size(self):
        seg = segment()
        image, mask = run(seg, image=torch.zeros((1, 16, 24, 3)))
        self.assertEqual(tuple(image.shape[1:3]), (8, 16))
        self.assertEqual(tuple(mask.shape[-2:]), (8, 16))

    def test_image_override_and_seg_mask_resolve_independently(self):
        seg = segment()
        image_override = torch.ones((1, 16, 24, 3))
        image, mask = run(seg, image=image_override, scale_to_guide_size="none")
        self.assertIs(image, image_override)
        self.assertEqual(tuple(mask.shape[-2:]), (16, 24))
        self.assertIsNot(mask, seg.cropped_mask)

    def test_mask_override_and_seg_image_resolve_independently(self):
        seg = segment()
        mask_override = torch.full((1, 12, 20), 0.25)
        image, mask = run(seg, mask=mask_override, scale_to_guide_size="none")
        self.assertIs(image, seg.cropped_image)
        self.assertEqual(tuple(mask.shape[-2:]), (32, 64))
        self.assertTrue(torch.allclose(mask, torch.full_like(mask, 0.25)))

    def test_soft_mask_is_interpolated_without_binarizing(self):
        _, mask = run(segment())
        self.assertTrue(torch.any((mask > 0) & (mask < 1)))
        self.assertGreater(torch.unique(mask).numel(), 2)

    def test_missing_seg_payloads_require_overrides(self):
        seg = segment(image=None, mask=None)._replace(cropped_image=None, cropped_mask=None)
        with self.assertRaisesRegex(ValueError, "requires image"):
            run(seg)
        with self.assertRaisesRegex(ValueError, "requires mask"):
            run(seg, image=torch.zeros((1, 32, 64, 3)))


if __name__ == "__main__":
    unittest.main()
