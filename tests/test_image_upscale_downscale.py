import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_upscale_downscale_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.image_upscale_downscale"]


class UpscaleAndDownscaleTests(unittest.TestCase):
    def test_schema(self):
        schema = MODULE.UpscaleAndDownscale.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteUpscaleAndDownscale")
        self.assertEqual(schema.display_name, "Upscale and Downscale")
        self.assertEqual([item.id for item in schema.inputs], ["image", "upscale_method", "scale_by", "cycle"])
        self.assertEqual(schema.inputs[1].options, MODULE.UPSCALE_METHODS)
        self.assertEqual(schema.inputs[1].default, "bilinear")
        self.assertEqual(schema.inputs[2].default, 8.0)
        self.assertEqual(schema.inputs[2].min, 0.01)
        self.assertEqual(schema.inputs[2].max, 8.0)
        self.assertEqual(schema.inputs[3].default, 1)
        self.assertEqual(schema.inputs[3].min, 1)
        self.assertEqual(schema.inputs[3].max, 5)
        self.assertEqual(schema.outputs[0].id, "image")

    def test_upscale_then_restore_repeats_for_each_cycle(self):
        image = torch.rand((2, 3, 5, 3))
        calls = []

        def resize(value, width, height, method, crop):
            calls.append((tuple(value.shape), width, height, method, crop))
            return torch.zeros((value.shape[0], value.shape[1], height, width), dtype=value.dtype)

        with patch.object(MODULE.comfy.utils, "common_upscale", side_effect=resize):
            result = MODULE.upscale_and_downscale(image, "bicubic", 2.0, 2)

        self.assertEqual(result.shape, image.shape)
        self.assertEqual([(call[1], call[2]) for call in calls], [(10, 6), (5, 3), (10, 6), (5, 3)])
        self.assertTrue(all(call[3:] == ("bicubic", "disabled") for call in calls))

    def test_factor_below_one_downscales_then_restores(self):
        image = torch.rand((1, 7, 9, 4))
        sizes = []

        def resize(value, width, height, method, crop):
            sizes.append((width, height))
            return torch.zeros((value.shape[0], value.shape[1], height, width), dtype=value.dtype)

        with patch.object(MODULE.comfy.utils, "common_upscale", side_effect=resize):
            result = MODULE.upscale_and_downscale(image, "area", 0.5, 1)

        self.assertEqual(sizes, [(4, 4), (9, 7)])
        self.assertEqual(result.shape, image.shape)

    def test_small_scaled_dimensions_are_clamped_to_one(self):
        image = torch.rand((1, 2, 3, 3))
        sizes = []

        def resize(value, width, height, method, crop):
            sizes.append((width, height))
            return torch.zeros((value.shape[0], value.shape[1], height, width), dtype=value.dtype)

        with patch.object(MODULE.comfy.utils, "common_upscale", side_effect=resize):
            MODULE.upscale_and_downscale(image, "nearest-exact", 0.01, 1)

        self.assertEqual(sizes, [(1, 1), (3, 2)])

    def test_real_core_resize_preserves_batch_channels_and_original_size(self):
        image = torch.rand((2, 4, 6, 4))
        result = MODULE.upscale_and_downscale(image, "bilinear", 1.5, 2)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, image.dtype)

    def test_invalid_arguments_raise(self):
        image = torch.rand((1, 4, 4, 3))
        cases = [
            ((image[0], "bilinear", 2.0, 1), "nonempty NHWC"),
            ((image, "invalid", 2.0, 1), "upscale_method"),
            ((image, "bilinear", 0.0, 1), "0.01..8.0"),
            ((image, "bilinear", 8.01, 1), "0.01..8.0"),
            ((image, "bilinear", 2.0, 0), "1..5"),
            ((image, "bilinear", 2.0, 6), "1..5"),
        ]
        for arguments, message in cases:
            with self.subTest(arguments=arguments), self.assertRaisesRegex((TypeError, ValueError), message):
                MODULE.upscale_and_downscale(*arguments)


if __name__ == "__main__":
    unittest.main()
