import asyncio
import importlib.util
import math
import sys
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_segs_sampler_custom_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.segs_sampler_custom"]

SEG = namedtuple(
    "SEG", "cropped_image cropped_mask confidence crop_region bbox label control_net_wrapper"
)


class FakeVAE:
    def __init__(self, compression=1):
        self.compression = compression
        self.calls = []

    def spacial_compression_encode(self):
        return self.compression

    def spacial_compression_decode(self):
        return self.compression

    def encode(self, image):
        self.calls.append(("encode", tuple(image.shape)))
        return image.clone()

    def encode_tiled(self, image, **kwargs):
        self.calls.append(("encode_tiled", kwargs))
        return image.clone()

    def decode(self, samples):
        self.calls.append(("decode", tuple(samples.shape)))
        return samples

    def decode_tiled(self, samples, **kwargs):
        self.calls.append(("decode_tiled", kwargs))
        return samples


def segment(crop, mask=None, bbox=None, cropped_image=None, label="x", wrapper=None):
    x1, y1, x2, y2 = crop
    if mask is None:
        mask = torch.ones((y2 - y1, x2 - x1))
    return SEG(cropped_image, mask, 0.75, crop, bbox or crop, label, wrapper)


def arguments(image, entries, **overrides):
    values = {
        "image": image,
        "segs": ((image.shape[1], image.shape[2]), entries),
        "noise": object(),
        "guider": object(),
        "sampler": object(),
        "sigmas": torch.tensor([1.0, 0.0]),
        "vae": FakeVAE(),
        "guide_size": 4,
        "guide_size_for": "crop_region",
        "scale_to_guide_size": "none",
        "max_size": 16,
        "force_inpaint": "enabled",
        "feather": 0,
        "noise_mask": "disabled",
        "noise_mask_feather": 0,
        "tiled_encode": "disabled",
        "tiled_decode": "disabled",
        "operation": "sequential",
    }
    values.update(overrides)
    return values


def add_sampler(amount=1.0, seen=None):
    def sample(noise, guider, sampler, sigmas, latent):
        if seen is not None:
            seen.append({key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in latent.items()})
        result = latent.copy()
        result["samples"] = latent["samples"] + amount
        return result

    return sample


class ResolutionTests(unittest.TestCase):
    def size(self, **overrides):
        values = {
            "crop_width": 800,
            "crop_height": 200,
            "reference_width": 800,
            "reference_height": 200,
            "guide_size": 400,
            "scale_mode": "scale",
            "max_size": 2000,
            "multiple": 1,
        }
        values.update(overrides)
        return MODULE.processing_size(**values)

    def test_square_root_area_guide_and_non_square_aspect(self):
        self.assertEqual(self.size(), (800, 200))
        width, height = self.size(guide_size=200)
        self.assertEqual((width, height), (400, 100))
        self.assertEqual(width / height, 4)

    def test_reference_region_changes_scaling(self):
        crop_reference = self.size(reference_width=800, reference_height=200)
        bbox_reference = self.size(reference_width=200, reference_height=50)
        self.assertEqual(crop_reference, (800, 200))
        self.assertEqual(bbox_reference, (3200, 800))

    def test_all_scaling_modes(self):
        self.assertEqual(self.size(guide_size=800, scale_mode="upscale"), (1600, 400))
        self.assertEqual(self.size(guide_size=200, scale_mode="upscale"), (800, 200))
        self.assertEqual(self.size(guide_size=800, scale_mode="downscale"), (800, 200))
        self.assertEqual(self.size(guide_size=200, scale_mode="downscale"), (400, 100))
        self.assertEqual(self.size(guide_size=200, scale_mode="none", max_size=10), (800, 200))

    def test_max_size_and_vae_rounding(self):
        width, height = self.size(guide_size=2000, max_size=500, multiple=8)
        self.assertLessEqual(math.sqrt(width * height), 500)
        self.assertEqual((width % 8, height % 8), (0, 0))

    def test_force_inpaint_uses_original_reference(self):
        self.assertFalse(MODULE.should_sample(1024, 1024, 512, "disabled"))
        self.assertTrue(MODULE.should_sample(1024, 1024, 512, "enabled"))
        self.assertTrue(MODULE.should_sample(256, 256, 512, "disabled"))


class ProcessingTests(unittest.TestCase):
    def test_schema_defaults_tooltips_and_registration(self):
        schema = MODULE.SEGSSamplerCustom.define_schema()
        self.assertEqual(schema.display_name, "SEGS Sampler Custom")
        defaults = {item.id: getattr(item, "default", None) for item in schema.inputs}
        self.assertEqual(
            {
                key: defaults[key]
                for key in defaults
                if key not in {"image", "segs", "noise", "guider", "sampler", "sigmas", "vae", "unsample"}
            },
            {
                "guide_size": 512,
                "guide_size_for": "bbox",
                "scale_to_guide_size": "upscale",
                "max_size": 1024,
                "force_inpaint": "enabled",
                "feather": 5,
                "noise_mask": "enabled",
                "noise_mask_feather": 20,
                "tiled_encode": "disabled",
                "tiled_decode": "disabled",
                "operation": "sequential",
            },
        )
        self.assertIn("Square root", schema.inputs[7].tooltip)
        nodes = asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list())
        self.assertIn(MODULE.SEGSSamplerCustom, nodes)

    def test_sequential_overlap_reads_updated_canvas_and_final_crops(self):
        image = torch.zeros((1, 4, 6, 3))
        entries = [segment((0, 0, 4, 4), label="first"), segment((2, 0, 6, 4), label="second")]
        seen = []
        with patch.object(MODULE, "_sample_once", add_sampler(seen=seen)):
            output, segs = MODULE.segs_sampler_custom(**arguments(image, entries))
        self.assertEqual(float(seen[1]["samples"][:, :, :2].mean()), 1.0)
        self.assertTrue(torch.all(output[:, :, :2] == 1))
        self.assertTrue(torch.all(output[:, :, 2:4] == 2))
        self.assertTrue(torch.all(output[:, :, 4:] == 1))
        self.assertTrue(np.all(segs[1][0].cropped_image[:, :, 2:] == 2))
        self.assertEqual([entry.label for entry in segs[1]], ["first", "second"])

    def test_individual_crops_are_independent_and_paste_order_wins(self):
        image = torch.zeros((1, 4, 6, 3))
        entries = [segment((0, 0, 4, 4)), segment((2, 0, 6, 4))]
        seen = []
        with patch.object(MODULE, "_sample_once", add_sampler(seen=seen)):
            output, segs = MODULE.segs_sampler_custom(**arguments(image, entries, operation="individual"))
        self.assertTrue(torch.all(seen[0]["samples"] == 0))
        self.assertTrue(torch.all(seen[1]["samples"] == 0))
        self.assertTrue(torch.all(output == 1))
        self.assertTrue(np.all(segs[1][0].cropped_image == 1))
        self.assertTrue(np.all(segs[1][1].cropped_image == 1))

    def test_individual_prefers_existing_crop(self):
        image = torch.zeros((1, 2, 2, 3))
        source = np.full((1, 2, 2, 3), 4.0, dtype=np.float32)
        seen = []
        with patch.object(MODULE, "_sample_once", add_sampler(0, seen)):
            MODULE.segs_sampler_custom(**arguments(image, [segment((0, 0, 2, 2), cropped_image=source)], operation="individual"))
        self.assertTrue(torch.all(seen[0]["samples"] == 4))

    def test_skipped_and_empty_masks_are_retained_with_metadata(self):
        image = torch.rand((1, 4, 4, 3))
        wrapper = object()
        entries = [
            segment(
                (0, 0, 2, 2),
                mask=torch.zeros((2, 2)),
                bbox=(0, 0, 0, 0),
                label="empty",
                wrapper=wrapper,
            ),
            segment((2, 0, 4, 2), label="skip", wrapper=wrapper),
        ]
        with patch.object(MODULE, "_sample_once", side_effect=AssertionError("must skip")):
            output, segs = MODULE.segs_sampler_custom(
                **arguments(image, entries, guide_size=1, force_inpaint="disabled")
            )
        self.assertTrue(torch.equal(output, image))
        self.assertEqual(len(segs[1]), 2)
        for original, result in zip(entries, segs[1]):
            self.assertEqual(result.crop_region, original.crop_region)
            self.assertEqual(result.bbox, original.bbox)
            self.assertEqual(result.confidence, original.confidence)
            self.assertIs(result.control_net_wrapper, wrapper)

    def test_zero_bbox_requires_an_empty_mask(self):
        image = torch.zeros((1, 2, 2, 3))
        entry = segment((0, 0, 2, 2), bbox=(0, 0, 0, 0))
        with self.assertRaisesRegex(ValueError, "invalid bbox"):
            MODULE.segs_sampler_custom(**arguments(image, [entry]))

    def test_noise_mask_and_composite_feather_are_independent(self):
        image = torch.zeros((1, 5, 5, 3))
        mask = torch.zeros((5, 5))
        mask[2, 2] = 1
        seen = []
        with patch.object(MODULE, "_sample_once", add_sampler(seen=seen)):
            output, _ = MODULE.segs_sampler_custom(
                **arguments(
                    image,
                    [segment((0, 0, 5, 5), mask=mask)],
                    noise_mask="enabled",
                    noise_mask_feather=0,
                    feather=1,
                )
            )
        self.assertTrue(torch.equal(seen[0]["noise_mask"], mask.unsqueeze(0)))
        self.assertGreater(int(torch.count_nonzero(output)), 3)

    def test_sampling_executes_once_per_segment(self):
        image = torch.zeros((1, 2, 2, 3))
        seen = []
        with patch.object(MODULE, "_sample_once", add_sampler(seen=seen)):
            output, _ = MODULE.segs_sampler_custom(**arguments(image, [segment((0, 0, 2, 2))]))
        self.assertEqual([float(item["samples"].mean()) for item in seen], [0.0])
        self.assertTrue(torch.all(output == 1))

    def test_normal_and_tiled_vae_paths(self):
        image = torch.zeros((1, 2, 2, 3))
        for tiled, expected in (("disabled", ["encode", "decode"]), ("enabled", ["encode_tiled", "decode_tiled"])):
            vae = FakeVAE()
            with self.subTest(tiled=tiled), patch.object(MODULE, "_sample_once", add_sampler(0)):
                MODULE.segs_sampler_custom(
                    **arguments(image, [segment((0, 0, 2, 2))], vae=vae, tiled_encode=tiled, tiled_decode=tiled)
                )
            self.assertEqual([call[0] for call in vae.calls], expected)


if __name__ == "__main__":
    unittest.main()
