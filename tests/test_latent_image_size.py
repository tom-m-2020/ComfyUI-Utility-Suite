from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_latent_image_size_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.latent_image_size"]


class MetadataOnlyVAE:
    def __init__(self, compression: int):
        self.compression = compression
        self.metadata_calls = 0

    def spacial_compression_decode(self) -> int:
        self.metadata_calls += 1
        return self.compression

    def decode(self, _samples):
        raise AssertionError("size inspection must not decode")


class GetImageSizeFromLatentTests(unittest.TestCase):
    def test_conventional_scale_eight_latent(self):
        samples = torch.empty((2, 4, 128, 256))
        vae = MetadataOnlyVAE(8)
        self.assertEqual(MODULE.get_latent_image_size({"samples": samples}, vae), (2048, 1024, 2, 4))
        self.assertEqual(vae.metadata_calls, 1)

    def test_z_image_scale_eight_and_channels(self):
        samples = torch.empty((1, 16, 128, 256))
        self.assertEqual(
            MODULE.get_latent_image_size({"samples": samples}, MetadataOnlyVAE(8)),
            (2048, 1024, 1, 16),
        )

    def test_flux2_klein_scale_sixteen_and_channels(self):
        samples = torch.empty((1, 128, 64, 128))
        self.assertEqual(
            MODULE.get_latent_image_size({"samples": samples}, MetadataOnlyVAE(16)),
            (2048, 1024, 1, 128),
        )

    def test_five_dimensional_anima_style_latent_uses_final_spatial_axes(self):
        samples = torch.empty((3, 16, 1, 128, 256))
        self.assertEqual(
            MODULE.get_latent_image_size({"samples": samples}, MetadataOnlyVAE(8)),
            (2048, 1024, 3, 16),
        )

    def test_same_shape_can_represent_different_sizes(self):
        latent = {"samples": torch.empty((1, 16, 10, 20))}
        self.assertEqual(MODULE.get_latent_image_size(latent, MetadataOnlyVAE(8))[:2], (160, 80))
        self.assertEqual(MODULE.get_latent_image_size(latent, MetadataOnlyVAE(42))[:2], (840, 420))

    def test_optional_latent_scale_metadata_is_not_treated_as_authoritative(self):
        latent = {
            "samples": torch.empty((1, 128, 10, 20)),
            "downscale_ratio_spacial": 8,
        }
        self.assertEqual(MODULE.get_latent_image_size(latent, MetadataOnlyVAE(16))[:2], (320, 160))

    def test_execution_only_reads_shape_and_vae_metadata(self):
        samples = torch.empty((1, 128, 2, 3), device="meta")
        vae = MetadataOnlyVAE(16)
        result = MODULE.GetImageSizeFromLatent.execute({"samples": samples}, vae).result
        self.assertEqual(result, (48, 32, 1, 128))
        self.assertEqual(samples.device.type, "meta")
        self.assertEqual(vae.metadata_calls, 1)

    def test_input_is_not_mutated(self):
        samples = torch.randn((1, 4, 3, 5))
        latent = {"samples": samples, "noise_mask": torch.ones((1, 24, 40))}
        keys = tuple(latent.keys())
        MODULE.get_latent_image_size(latent, MetadataOnlyVAE(8))
        self.assertEqual(tuple(latent.keys()), keys)
        self.assertIs(latent["samples"], samples)

    def test_invalid_latent_structures_fail_clearly(self):
        vae = MetadataOnlyVAE(8)
        cases = (
            (None, TypeError, "dictionary"),
            ({}, TypeError, "torch.Tensor"),
            ({"samples": [1]}, TypeError, "torch.Tensor"),
            ({"samples": torch.empty((1, 4, 8))}, ValueError, "4-D"),
            ({"samples": torch.empty((1, 4, 0, 8))}, ValueError, "nonempty"),
            ({"samples": torch.empty((1, 4, 1, 2, 3, 4))}, ValueError, "5-D"),
        )
        for latent, error_type, message in cases:
            with self.subTest(latent=latent), self.assertRaisesRegex(error_type, message):
                MODULE.get_latent_image_size(latent, vae)

    def test_invalid_vae_metadata_fails_clearly(self):
        latent = {"samples": torch.empty((1, 4, 8, 8))}
        with self.assertRaisesRegex(TypeError, "spatial compression metadata"):
            MODULE.get_latent_image_size(latent, object())
        for compression in (0, -1, 8.0, True, None):
            with self.subTest(compression=compression), self.assertRaisesRegex(ValueError, "positive integer"):
                MODULE.get_latent_image_size(latent, MetadataOnlyVAE(compression))

    def test_schema_and_output_order(self):
        schema = MODULE.GetImageSizeFromLatent.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteGetImageSizeFromLatent")
        self.assertEqual(schema.display_name, "Get Image Size from Latent")
        self.assertEqual(schema.category, "Utility Suite/Latent")
        self.assertEqual([item.id for item in schema.inputs], ["latent", "vae"])
        self.assertEqual([item.io_type for item in schema.inputs], ["LATENT", "VAE"])
        self.assertEqual(
            [item.id for item in schema.outputs],
            ["width", "height", "batch_size", "channels"],
        )
        self.assertTrue(all(item.io_type == "INT" for item in schema.outputs))


if __name__ == "__main__":
    unittest.main()
