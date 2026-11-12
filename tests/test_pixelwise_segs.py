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
    "utility_suite_pixelwise_segs_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.pixelwise_segs"]
COMBINE_FEATHER_MODULE = sys.modules[f"{SPEC.name}.combine_feather_segs"]

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)


def segment(crop, mask, label):
    return SEG(object(), mask, 0.75, crop, tuple(crop), label, object())


class PixelwiseSEGSTests(unittest.TestCase):
    def test_in_segs_max_uses_full_canvas_coordinates(self):
        first = segment((0, 0, 3, 2), np.full((2, 3), 0.4, dtype=np.float32), "first")
        second = segment((2, 0, 5, 2), np.full((2, 3), 0.7, dtype=np.float32), "second")

        result = MODULE.pixelwise_in_segs(((2, 5), [first, second]), "max")

        np.testing.assert_array_equal(
            result[1][0].cropped_mask,
            np.array([[0.4, 0.4, 0.7], [0.4, 0.4, 0.7]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            result[1][1].cropped_mask,
            np.array([[0.7, 0.7, 0.7], [0.7, 0.7, 0.7]], dtype=np.float32),
        )

    def test_in_segs_min_uses_one_outside_each_crop(self):
        first = segment((0, 0, 3, 2), np.full((2, 3), 0.8, dtype=np.float32), "first")
        second = segment((2, 0, 5, 2), np.full((2, 3), 0.5, dtype=np.float32), "second")

        result = MODULE.pixelwise_in_segs(((2, 5), [first, second]), "min")

        expected_first = np.full((2, 3), 0.8, dtype=np.float32)
        expected_first[:, 2] = 0.5
        expected_second = np.full((2, 3), 0.5, dtype=np.float32)
        np.testing.assert_array_equal(result[1][0].cropped_mask, expected_first)
        np.testing.assert_array_equal(result[1][1].cropped_mask, expected_second)

    def test_in_segs_empty_and_singleton(self):
        header = (2, 3)
        self.assertEqual(MODULE.pixelwise_in_segs((header, []), "max"), (header, []))
        mask = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        result = MODULE.pixelwise_in_segs((header, [segment((1, 0, 3, 2), mask, "only")]), "min")
        np.testing.assert_array_equal(result[1][0].cropped_mask, mask)

    def test_segs_and_segs_compares_each_segs1_entry_with_all_segs2_masks(self):
        segs1 = (
            (2, 6),
            [
                segment((0, 0, 3, 2), np.full((2, 3), 0.2, dtype=np.float32), "a"),
                segment((3, 0, 6, 2), np.full((2, 3), 0.4, dtype=np.float32), "b"),
            ],
        )
        segs2 = (
            (2, 6),
            [
                segment((1, 0, 4, 2), np.full((2, 3), 0.6, dtype=np.float32), "source-1"),
                segment((2, 0, 5, 2), np.full((2, 3), 0.8, dtype=np.float32), "source-2"),
            ],
        )

        maximum = MODULE.pixelwise_segs_and_segs(segs1, segs2, "max")
        minimum = MODULE.pixelwise_segs_and_segs(segs1, segs2, "min")

        np.testing.assert_array_equal(
            maximum[1][0].cropped_mask[0], np.array([0.2, 0.6, 0.8], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            maximum[1][1].cropped_mask[0], np.array([0.8, 0.8, 0.4], dtype=np.float32)
        )
        expected_first = np.full((2, 3), 0.2, dtype=np.float32)
        expected_second = np.full((2, 3), 0.4, dtype=np.float32)
        np.testing.assert_array_equal(minimum[1][0].cropped_mask, expected_first)
        np.testing.assert_array_equal(minimum[1][1].cropped_mask, expected_second)
        self.assertEqual([entry.label for entry in maximum[1]], ["a", "b"])

    def test_segs_and_segs_empty_second_group_is_identity(self):
        mask = np.full((2, 3), 0.4, dtype=np.float32)
        segs1 = ((2, 3), [segment((0, 0, 3, 2), mask, "a")])
        for operation in ("min", "max"):
            result = MODULE.pixelwise_segs_and_segs(segs1, ((2, 3), []), operation)
            np.testing.assert_array_equal(result[1][0].cropped_mask, mask)

    def test_standalone_max_then_min_matches_combine_feather(self):
        destination = (
            (3, 8),
            [segment((1, 0, 7, 3), np.full((3, 6), 0.8, dtype=np.float32), "destination")],
        )
        source = (
            (3, 8),
            [
                segment((1, 0, 5, 3), np.full((3, 4), 0.3, dtype=np.float32), "source-a"),
                segment((3, 0, 7, 3), np.full((3, 4), 0.6, dtype=np.float32), "source-b"),
            ],
        )

        source_max = MODULE.pixelwise_in_segs(source, "max")
        standalone = MODULE.pixelwise_segs_and_segs(destination, source_max, "min")
        combined = COMBINE_FEATHER_MODULE.combine_feather_segs(destination, source)

        np.testing.assert_array_equal(standalone[1][0].cropped_mask, combined[1][0].cropped_mask)

    def test_segs_and_segs_requires_matching_canvas(self):
        with self.assertRaisesRegex(ValueError, "matching SEGS source shapes"):
            MODULE.pixelwise_segs_and_segs(((2, 3), []), ((3, 3), []), "max")

    def test_segs_and_mask_uses_full_canvas_coordinates(self):
        mask = np.full((2, 3), 0.6, dtype=np.float32)
        segs = ((2, 5), [segment((1, 0, 4, 2), mask, "destination")])
        source = torch.tensor([[0.1, 0.2, 0.3, 0.8, 0.9], [0.1, 0.2, 0.3, 0.8, 0.9]])

        minimum = MODULE.pixelwise_segs_and_mask(segs, source, "min")
        maximum = MODULE.pixelwise_segs_and_mask(segs, source.unsqueeze(0), "max")

        np.testing.assert_array_equal(
            minimum[1][0].cropped_mask[0], np.array([0.2, 0.3, 0.6], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            maximum[1][0].cropped_mask[0], np.array([0.6, 0.6, 0.8], dtype=np.float32)
        )

    def test_segs_and_mask_rejects_batch_and_wrong_canvas(self):
        segs = ((2, 3), [])
        with self.assertRaisesRegex(ValueError, r"\[H,W\] or \[1,H,W\]"):
            MODULE.pixelwise_segs_and_mask(segs, torch.zeros((2, 2, 3)), "max")
        with self.assertRaisesRegex(ValueError, "mask canvas must match"):
            MODULE.pixelwise_segs_and_mask(segs, torch.zeros((2, 4)), "max")

    def test_preserves_metadata_and_does_not_mutate_inputs(self):
        original_mask = np.full((2, 3), 0.5, dtype=np.float32)
        entry = segment((0, 0, 3, 2), original_mask, "destination")
        source = torch.full((2, 3), 0.25)

        result = MODULE.pixelwise_segs_and_mask(((2, 3), [entry]), source, "min")

        self.assertEqual(result[1][0][:1] + result[1][0][2:], entry[:1] + entry[2:])
        np.testing.assert_array_equal(original_mask, np.full((2, 3), 0.5, dtype=np.float32))
        self.assertTrue(torch.equal(source, torch.full((2, 3), 0.25)))

    def test_all_schemas_are_registered_with_exact_socket_names(self):
        expected = [
            (MODULE.PixelwiseMinInSEGS, "Pixelwise Min (in SEGS)", ["segs"]),
            (MODULE.PixelwiseMaxInSEGS, "Pixelwise Max (in SEGS)", ["segs"]),
            (MODULE.PixelwiseMinSEGSAndSEGS, "Pixelwise Min (SEGS & SEGS)", ["segs1", "segs2"]),
            (MODULE.PixelwiseMaxSEGSAndSEGS, "Pixelwise Max (SEGS & SEGS)", ["segs1", "segs2"]),
            (MODULE.PixelwiseMinSEGSAndMASK, "Pixelwise Min (SEGS & MASK)", ["segs", "mask"]),
            (MODULE.PixelwiseMaxSEGSAndMASK, "Pixelwise Max (SEGS & MASK)", ["segs", "mask"]),
        ]
        registered = asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list())
        for node_class, display_name, inputs in expected:
            schema = node_class.define_schema()
            self.assertEqual(schema.display_name, display_name)
            self.assertEqual(schema.category, "Utility Suite/SEGS")
            self.assertEqual([item.id for item in schema.inputs], inputs)
            self.assertEqual([item.io_type for item in schema.outputs], ["SEGS"])
            self.assertIn(node_class, registered)


if __name__ == "__main__":
    unittest.main()
