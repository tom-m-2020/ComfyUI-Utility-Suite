import importlib.util
import sys
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

import comfy.utils
import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_seg_paste_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.seg_paste"]

SEG = namedtuple(
    "SEG", "cropped_image cropped_mask confidence crop_region bbox label control_net_wrapper"
)


def segment(region, value, mask=None):
    x1, y1, x2, y2 = region
    if mask is None:
        mask = torch.ones((y2 - y1, x2 - x1))
    crop = np.full((1, y2 - y1, x2 - x1, 3), value, dtype=np.float32)
    return SEG(crop, mask, 0.75, region, region, "label", object())


class SEGPasteTests(unittest.TestCase):
    def test_schema(self):
        schema = MODULE.SEGPaste.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteSEGPaste")
        self.assertEqual(schema.display_name, "SEGPaste")
        self.assertEqual([item.id for item in schema.inputs], ["cropped_image", "segs", "index", "feather", "image"])
        self.assertEqual([item.id for item in schema.outputs], ["image", "segs"])

    def test_updates_only_selected_crop_and_uses_lanczos(self):
        first = segment((0, 0, 2, 2), 0.1)
        second = segment((2, 1, 6, 4), 0.2)
        segs = ((4, 6), [first, second])
        processed = torch.ones((1, 2, 3, 3))
        calls = []

        def resize(value, width, height, method, crop):
            calls.append((width, height, method, crop))
            return torch.ones((value.shape[0], value.shape[1], height, width), dtype=value.dtype)

        with patch.object(comfy.utils, "common_upscale", side_effect=resize):
            image_out, segs_out = MODULE.seg_paste(processed, segs, 1, 5)

        self.assertIsNone(image_out)
        self.assertEqual(calls, [(4, 3, "lanczos", "disabled")])
        self.assertIs(segs_out[1][0], first)
        self.assertIsNot(segs_out[1][1], second)
        self.assertEqual(segs_out[1][1]._replace(cropped_image=None), second._replace(cropped_image=None))
        self.assertEqual(segs_out[1][1].cropped_image.shape, (1, 3, 4, 3))
        self.assertIs(segs[1][1], second)

    def test_optional_image_composites_with_soft_mask_without_mutating_input(self):
        mask = torch.full((2, 2), 0.25)
        original_segment = segment((1, 1, 3, 3), 0.0, mask)
        segs = ((4, 4), [original_segment])
        image = torch.zeros((1, 4, 4, 3))
        processed = torch.ones((1, 2, 2, 3))

        image_out, segs_out = MODULE.seg_paste(processed, segs, 0, 0, image)

        self.assertTrue(torch.all(image == 0))
        self.assertTrue(torch.allclose(image_out[:, 1:3, 1:3], torch.full((1, 2, 2, 3), 0.25)))
        self.assertTrue(torch.all(image_out[:, :1] == 0))
        self.assertTrue(np.array_equal(segs_out[1][0].cropped_image, processed.numpy()))

    def test_invalid_inputs_raise(self):
        segs = ((4, 4), [segment((0, 0, 2, 2), 0.0)])
        crop = torch.zeros((1, 2, 2, 3))
        cases = [
            ((crop[0], segs, 0, 0, None), "one cropped IMAGE"),
            ((crop, segs, -1, 0, None), "outside"),
            ((crop, segs, 1, 0, None), "outside"),
            ((crop, segs, 0, 101, None), "0..100"),
            ((crop, segs, 0, 0, torch.zeros((1, 5, 4, 3))), "must match"),
        ]
        for arguments, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex((TypeError, ValueError, IndexError), message):
                MODULE.seg_paste(*arguments)


if __name__ == "__main__":
    unittest.main()
