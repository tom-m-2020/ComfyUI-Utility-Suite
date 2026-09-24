from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_empty_latent_vae_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.empty_latent_vae"]

import nodes
from comfy_extras.nodes_flux import EmptyFlux2LatentImage
from comfy_extras.nodes_sd3 import EmptySD3LatentImage


class MetadataOnlyVAE:
    def __init__(self, channels: int, compression: int, dimensions: int = 2):
        self.latent_channels = channels
        self.latent_dim = dimensions
        self.compression = compression
        self.metadata_calls = 0

    def spacial_compression_encode(self) -> int:
        self.metadata_calls += 1
        return self.compression

    def encode(self, _image):
        raise AssertionError("empty latent construction must not encode")

    def decode(self, _latent):
        raise AssertionError("empty latent construction must not decode")


class EmptyLatentFromVAETests(unittest.TestCase):
    def create(self, vae, width=1024, height=1024, batch_size=1):
        with (
            patch.object(MODULE.comfy.model_management, "intermediate_device", return_value=torch.device("cpu")),
            patch.object(MODULE.comfy.model_management, "intermediate_dtype", return_value=torch.float32),
        ):
            return MODULE.empty_latent_from_vae(vae, width, height, batch_size)

    def assert_empty(self, result, shape):
        self.assertEqual(set(result), {"samples"})
        self.assertEqual(tuple(result["samples"].shape), shape)
        self.assertEqual(result["samples"].dtype, torch.float32)
        self.assertEqual(result["samples"].device.type, "cpu")
        self.assertEqual(torch.count_nonzero(result["samples"]), 0)

    def test_sdxl_four_channel_scale_eight(self):
        self.assert_empty(self.create(MetadataOnlyVAE(4, 8), 1024, 512, 2), (2, 4, 64, 128))

    def test_sd3_flux_and_z_image_sixteen_channel_scale_eight(self):
        for width, height in ((1024, 1024), (512, 1024), (2048, 1024)):
            with self.subTest(size=(width, height)):
                self.assert_empty(
                    self.create(MetadataOnlyVAE(16, 8), width, height, 3),
                    (3, 16, height // 8, width // 8),
                )

    def test_flux2_klein_128_channel_scale_sixteen(self):
        self.assert_empty(self.create(MetadataOnlyVAE(128, 16), 2048, 1024, 2), (2, 128, 64, 128))

    def test_pixel_space_vae_scale_one(self):
        self.assert_empty(self.create(MetadataOnlyVAE(3, 1), 37, 23, 1), (1, 3, 23, 37))

    def test_three_dimensional_anima_wan_style_uses_one_temporal_latent(self):
        self.assert_empty(self.create(MetadataOnlyVAE(16, 8, 3), 1024, 512, 2), (2, 16, 1, 64, 128))

    def test_non_multiple_dimensions_use_existing_floor_semantics(self):
        self.assert_empty(self.create(MetadataOnlyVAE(16, 8), 1023, 517, 1), (1, 16, 64, 127))
        self.assert_empty(self.create(MetadataOnlyVAE(128, 16), 1031, 525, 1), (1, 128, 32, 64))

    def test_reference_equivalence_conventional_sd3_and_flux2(self):
        references = (
            (nodes.EmptyLatentImage().generate(2048, 1024, 2)[0], MetadataOnlyVAE(4, 8)),
            (EmptySD3LatentImage.execute(2048, 1024, 2).result[0], MetadataOnlyVAE(16, 8)),
            (EmptyFlux2LatentImage.execute(2048, 1024, 2).result[0], MetadataOnlyVAE(128, 16)),
        )
        for reference, vae in references:
            with self.subTest(shape=tuple(reference["samples"].shape)):
                actual = self.create(vae, 2048, 1024, 2)
                torch.testing.assert_close(actual["samples"], reference["samples"], rtol=0, atol=0)
                self.assertEqual(actual["samples"].dtype, reference["samples"].dtype)
                self.assertEqual(actual["samples"].device, reference["samples"].device)

    def test_fresh_dictionary_has_no_stale_fields_and_calls_only_metadata(self):
        vae = MetadataOnlyVAE(4, 8)
        first = self.create(vae)
        second = self.create(vae)
        self.assertEqual(set(first), {"samples"})
        self.assertEqual(set(second), {"samples"})
        self.assertIsNot(first, second)
        self.assertIsNot(first["samples"], second["samples"])
        self.assertEqual(vae.metadata_calls, 2)

    def test_audio_and_invalid_vae_metadata_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "latent_dim 2 or 3"):
            self.create(MetadataOnlyVAE(64, 2048, 1))
        for channels in (0, -1, 4.0, True, None):
            with self.subTest(channels=channels), self.assertRaisesRegex(ValueError, "latent_channels"):
                self.create(MetadataOnlyVAE(channels, 8))
        with self.assertRaisesRegex(TypeError, "spatial compression metadata"):
            self.create(type("VAE", (), {"latent_channels": 4, "latent_dim": 2})())
        for compression in (0, -1, 8.0, True, None):
            with self.subTest(compression=compression), self.assertRaisesRegex(ValueError, "compression ratio"):
                self.create(MetadataOnlyVAE(4, compression))

    def test_invalid_dimensions_and_too_small_request_fail_clearly(self):
        vae = MetadataOnlyVAE(4, 8)
        for name, values in (
            ("width", (0, 8, 1)),
            ("height", (8, 0, 1)),
            ("batch_size", (8, 8, 0)),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self.create(vae, *values)
        with self.assertRaisesRegex(ValueError, "smaller than"):
            self.create(vae, 7, 8, 1)

    def test_schema(self):
        schema = MODULE.EmptyLatentFromVAE.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteEmptyLatentFromVAE")
        self.assertEqual(schema.display_name, "Empty Latent from VAE")
        self.assertEqual(schema.category, "Utility Suite/Latent")
        self.assertEqual([item.id for item in schema.inputs], ["vae", "width", "height", "batch_size"])
        self.assertEqual([item.io_type for item in schema.inputs], ["VAE", "INT", "INT", "INT"])
        self.assertEqual(schema.outputs[0].io_type, "LATENT")


if __name__ == "__main__":
    unittest.main()
