from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_segs_upscaler_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.segs_upscaler"]

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)


def segment(region, mask=None, label="seg"):
    x1, y1, x2, y2 = region
    if mask is None:
        mask = np.ones((y2 - y1, x2 - x1), dtype=np.float32)
    return SEG(None, mask, 0.75, region, region, label, object())


def native_two_x(_model, image):
    return F.interpolate(image.movedim(-1, 1), scale_factor=2, mode="nearest").movedim(1, -1)


class SEGSUpscalerTests(unittest.TestCase):
    def setUp(self):
        self.model = object()
        self.image = torch.arange(4 * 4 * 3, dtype=torch.float32).reshape(1, 4, 4, 3) / 48.0

    def upscale(self, segs, upscaler=native_two_x, **overrides):
        arguments = {
            "image": self.image,
            "segs": segs,
            "upscale_model": self.model,
            "start_index": 0,
            "max_iteration": -1,
            "rescale_factor": 2.0,
            "resampling_method": "lanczos",
            "whole_upscale_first": True,
            "overscale_smoothing": False,
            "use_guide_size": False,
            "guide_size": 512,
            "feather": 0,
        }
        arguments.update(overrides)
        with mock.patch.object(MODULE, "_upscale_with_model", side_effect=upscaler) as upscale:
            result = MODULE.segs_upscaler(**arguments)
        return result, upscale

    def test_whole_image_is_model_upscaled_before_requested_resize(self):
        segs = ((4, 4), [])
        (image, output_segs), upscale = self.upscale(segs)
        self.assertEqual(tuple(image.shape), (1, 8, 8, 3))
        self.assertEqual(output_segs, ((8, 8), []))
        self.assertEqual(upscale.call_count, 1)
        self.assertIs(upscale.call_args.args[0], self.model)
        self.assertIs(upscale.call_args.args[1], self.image)

    def test_whole_upscale_false_uses_model_only_for_selected_segs(self):
        entries = [segment((0, 0, 2, 2))]
        (_, _), upscale = self.upscale(((4, 4), entries), whole_upscale_first=False)
        self.assertEqual(upscale.call_count, 1)
        self.assertEqual(tuple(upscale.call_args.args[1].shape), (1, 4, 4, 3))

    def test_overscale_smoothing_uses_five_times_requested_canvas(self):
        calls = []
        original_resize = MODULE._resize_image

        def record_resize(image, width, height, method):
            calls.append((width, height, method))
            return original_resize(image, width, height, method)

        with (
            mock.patch.object(MODULE, "_resize_image", side_effect=record_resize),
            mock.patch.object(MODULE, "_upscale_with_model", side_effect=native_two_x),
        ):
            image, _ = MODULE.segs_upscaler(
                self.image,
                ((4, 4), []),
                self.model,
                0,
                -1,
                2.0,
                "bicubic",
                False,
                True,
                False,
                512,
                0,
            )
        self.assertEqual(calls, [(40, 40, "bicubic"), (8, 8, "bicubic")])
        self.assertEqual(tuple(image.shape), (1, 8, 8, 3))

    def test_geometry_masks_and_unprocessed_crops_match_scaled_canvas(self):
        entry = segment((1, 1, 3, 3), label="keep")
        (image, output_segs), upscale = self.upscale(
            ((4, 4), [entry]),
            whole_upscale_first=False,
            start_index=1,
        )
        output = output_segs[1][0]
        self.assertEqual(output_segs[0], (8, 8))
        self.assertEqual(output.crop_region, (2, 2, 6, 6))
        self.assertEqual(output.bbox, (2, 2, 6, 6))
        self.assertEqual(np.shape(output.cropped_mask), (4, 4))
        self.assertEqual(np.shape(output.cropped_image), (1, 4, 4, 3))
        np.testing.assert_allclose(output.cropped_image, image[:, 2:6, 2:6, :].numpy())
        self.assertEqual(output.label, "keep")
        self.assertIs(output.control_net_wrapper, entry.control_net_wrapper)
        self.assertEqual(upscale.call_count, 0)

    def test_guide_size_uses_square_root_area_and_eight_pixel_alignment(self):
        image = torch.zeros((1, 4, 8, 3), dtype=torch.float32)
        entry = segment((0, 0, 8, 4))
        captured = []

        def capture(_model, value):
            captured.append(tuple(value.shape))
            return value

        with mock.patch.object(MODULE, "_upscale_with_model", side_effect=capture):
            MODULE.segs_upscaler(
                image,
                ((4, 8), [entry]),
                self.model,
                0,
                -1,
                1.0,
                "lanczos",
                False,
                False,
                True,
                16,
                0,
            )
        self.assertEqual(captured, [(1, 8, 24, 3)])

    def test_each_seg_is_upscaled_from_unchanged_base_canvas(self):
        image = torch.zeros((1, 3, 4, 3), dtype=torch.float32)
        image[:, :, 2:, :] = 0.4
        entries = [segment((0, 0, 3, 3), label="first"), segment((1, 0, 4, 3), label="second")]
        captured = []

        def enhance(_model, value):
            captured.append(value.clone())
            return value + len(captured) * 0.1

        with mock.patch.object(MODULE, "_upscale_with_model", side_effect=enhance):
            canvas, output_segs = MODULE.segs_upscaler(
                image,
                ((3, 4), entries),
                self.model,
                0,
                -1,
                1.0,
                "nearest",
                False,
                False,
                False,
                512,
                0,
            )
        torch.testing.assert_close(captured[0], image[:, :, 0:3, :])
        torch.testing.assert_close(captured[1], image[:, :, 1:4, :])
        np.testing.assert_allclose(output_segs[1][0].cropped_image, (captured[0] + 0.1).numpy())
        np.testing.assert_allclose(output_segs[1][1].cropped_image, (captured[1] + 0.2).numpy())
        torch.testing.assert_close(canvas[:, :, 1:3, :], captured[1][:, :, 0:2, :] + 0.2)

    def test_soft_mask_controls_stitching_but_not_stored_enhanced_crop(self):
        image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
        mask = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
        entry = segment((0, 0, 2, 2), mask=mask)

        with mock.patch.object(MODULE, "_upscale_with_model", side_effect=lambda _model, value: value + 1.0):
            canvas, output_segs = MODULE.segs_upscaler(
                image,
                ((2, 2), [entry]),
                self.model,
                0,
                -1,
                1.0,
                "nearest",
                False,
                False,
                False,
                512,
                0,
            )
        np.testing.assert_allclose(output_segs[1][0].cropped_image, np.ones((1, 2, 2, 3)))
        expected = torch.as_tensor(mask).unsqueeze(0).unsqueeze(-1).expand_as(canvas)
        torch.testing.assert_close(canvas, expected)

    def test_start_index_and_max_iteration_limit_seg_processing(self):
        entries = [segment((0, 0, 2, 2), label="A"), segment((2, 0, 4, 2), label="B")]
        (_, output_segs), upscale = self.upscale(
            ((4, 4), entries),
            whole_upscale_first=False,
            start_index=1,
            max_iteration=1,
        )
        self.assertEqual(upscale.call_count, 1)
        base_a = output_segs[1][0].cropped_image
        enhanced_b = output_segs[1][1].cropped_image
        self.assertEqual(np.shape(base_a), (1, 4, 4, 3))
        self.assertEqual(np.shape(enhanced_b), (1, 4, 4, 3))

    def test_empty_masks_are_retained_and_not_model_upscaled(self):
        empty = segment((0, 0, 2, 2), mask=np.zeros((2, 2), dtype=np.float32))
        (_, output_segs), upscale = self.upscale(((4, 4), [empty]), whole_upscale_first=False)
        self.assertEqual(upscale.call_count, 0)
        self.assertEqual(len(output_segs[1]), 1)
        self.assertEqual(np.shape(output_segs[1][0].cropped_image), (1, 4, 4, 3))

    def test_iteration_bounds_match_sampler_contract(self):
        self.assertEqual(MODULE.iteration_bounds(10, 0, -1), (0, 10))
        self.assertEqual(MODULE.iteration_bounds(10, 3, 0), (3, 10))
        self.assertEqual(MODULE.iteration_bounds(10, 3, 4), (3, 7))
        self.assertEqual(MODULE.iteration_bounds(10, 8, 10), (8, 10))
        self.assertEqual(MODULE.iteration_bounds(10, 10, -1), (10, 10))

    def test_invalid_inputs_fail_clearly(self):
        segs = ((4, 4), [])
        invalid = (
            ({"start_index": -1}, "start_index"),
            ({"max_iteration": -2}, "max_iteration"),
            ({"rescale_factor": 0}, "positive"),
            ({"resampling_method": "area"}, "resampling_method"),
            ({"guide_size": 7}, "guide_size"),
            ({"feather": 101}, "feather"),
        )
        for override, message in invalid:
            with self.subTest(override=override), self.assertRaisesRegex((TypeError, ValueError), message):
                self.upscale(segs, **override)

    def test_schema_defaults_and_outputs(self):
        schema = MODULE.SEGSUpscaler.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteSEGSUpscaler")
        self.assertEqual(schema.display_name, "SEGS Upscaler")
        self.assertEqual(
            [entry.id for entry in schema.inputs],
            [
                "image",
                "segs",
                "upscale_model",
                "start_index",
                "max_iteration",
                "rescale_factor",
                "resampling_method",
                "whole_upscale_first",
                "overscale_smoothing",
                "use_guide_size",
                "guide_size",
                "feather",
            ],
        )
        self.assertEqual(schema.inputs[3].default, 0)
        self.assertEqual(schema.inputs[4].default, -1)
        self.assertEqual(schema.inputs[5].default, 2.0)
        self.assertEqual(schema.inputs[6].options, MODULE.RESAMPLING_METHODS)
        self.assertTrue(schema.inputs[7].default)
        self.assertFalse(schema.inputs[8].default)
        self.assertTrue(schema.inputs[9].default)
        self.assertEqual(schema.inputs[10].default, 512)
        self.assertEqual(schema.inputs[11].default, 5)
        self.assertEqual([output.io_type for output in schema.outputs], ["IMAGE", "SEGS"])


if __name__ == "__main__":
    unittest.main()
