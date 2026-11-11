import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_mask_change_ratio_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.mask_change_ratio"]


def run(mask, **overrides):
    values = {
        "ratio": 1.0,
        "tiling": False,
        "overlap_mode": "pixel",
        "min_overlap": 16,
        "feather_overlap": False,
        "max_tile_size": 2048,
        "shape": "square",
    }
    values.update(overrides)
    return MODULE.mask_change_ratio(mask, **values)


class MASKChangeRatioTests(unittest.TestCase):
    def test_schema_and_registration(self):
        schema = MODULE.MASKChangeRatio.define_schema()

        self.assertEqual(schema.node_id, "UtilitySuiteMASKChangeRatio")
        self.assertEqual(schema.display_name, "MASK Change Ratio")
        self.assertEqual(schema.category, "Utility Suite/Mask")
        self.assertEqual(
            [item.id for item in schema.inputs],
            [
                "mask",
                "ratio",
                "tiling",
                "overlap_mode",
                "min_overlap",
                "feather_overlap",
                "max_tile_size",
                "shape",
            ],
        )
        self.assertEqual(schema.inputs[3].default, "pixel")
        self.assertEqual(schema.inputs[6].min, 512)
        self.assertEqual(schema.inputs[7].default, "square")
        nodes = asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list())
        self.assertIn(MODULE.MASKChangeRatio, nodes)

    def test_scattered_input_becomes_one_filled_union_boundary(self):
        mask = torch.zeros(1, 100, 160)
        mask[0, 20:30, 10:20] = 1
        mask[0, 60:80, 130:150] = 1

        result = run(mask, ratio=140 / 60)

        self.assertEqual(result.shape, (1, 100, 160))
        self.assertTrue(torch.all(result[0, 20:80, 10:150] == 1))
        self.assertEqual(int(torch.count_nonzero(result)), 140 * 60)

    def test_ratio_expands_short_axis_and_respects_canvas(self):
        mask = torch.zeros(1, 100, 160)
        mask[0, 10:30, 0:80] = 1

        result = run(mask, ratio=1.0)

        self.assertTrue(torch.all(result[0, 0:80, 0:80] == 1))
        self.assertEqual(int(torch.count_nonzero(result)), 80 * 80)

    def test_batch_items_are_processed_independently_and_preserve_order(self):
        mask = torch.zeros(2, 100, 160)
        mask[0, 10:30, 10:50] = 1
        mask[1, 60:80, 100:120] = 1

        result = run(mask)

        self.assertEqual(result.shape, (2, 100, 160))
        self.assertEqual(tuple(torch.nonzero(result[0]).amin(0).tolist()), (0, 10))
        self.assertEqual(tuple(torch.nonzero(result[1]).amin(0).tolist()), (60, 100))

    def test_tiling_emits_separate_full_canvas_masks(self):
        mask = torch.ones(1, 2048, 2048)

        result = run(
            mask,
            tiling=True,
            min_overlap=32,
            max_tile_size=512,
        )

        self.assertEqual(result.shape, (25, 2048, 2048))
        self.assertTrue(torch.all(result.sum(dim=0) >= 1))
        self.assertTrue(
            all(int(torch.count_nonzero(item)) <= 512 * 512 for item in result)
        )

    def test_overlap_feathers_are_complementary(self):
        mask = torch.ones(1, 512, 1024)

        result = run(
            mask,
            tiling=True,
            min_overlap=32,
            feather_overlap=True,
            max_tile_size=512,
        )

        self.assertEqual(result.shape[0], 3)
        self.assertTrue(torch.allclose(result.sum(dim=0), torch.ones(512, 1024)))

    def test_circle_is_inscribed_in_resolved_boundary(self):
        mask = torch.zeros(1, 80, 120)
        mask[0, 20:60, 20:100] = 1

        result = run(mask, ratio=2.0, shape="circle")

        self.assertEqual(float(result[0, 40, 60]), 1.0)
        self.assertEqual(float(result[0, 20, 20]), 0.0)
        self.assertEqual(float(result[0, 20, 60]), 1.0)
        self.assertEqual(float(result[0, 40, 20]), 1.0)
        self.assertTrue(torch.all(result[:, :20] == 0))

    def test_circle_tiling_is_supported(self):
        result = run(
            torch.ones(1, 512, 1024),
            tiling=True,
            max_tile_size=512,
            shape="circle",
        )

        self.assertGreater(result.shape[0], 1)
        self.assertTrue(torch.all(result[:, 0, 0] == 0))

    def test_empty_batch_item_is_preserved_as_empty_mask(self):
        mask = torch.zeros(2, 20, 30)
        mask[1, 5:10, 8:14] = 1

        result = run(mask)

        self.assertEqual(result.shape, (2, 20, 30))
        self.assertEqual(int(torch.count_nonzero(result[0])), 0)
        self.assertGreater(int(torch.count_nonzero(result[1])), 0)

    def test_percent_overlap_and_invalid_inputs(self):
        mask = torch.ones(1, 512, 1024)
        percent = run(
            mask,
            tiling=True,
            overlap_mode="percent",
            min_overlap=10,
            max_tile_size=512,
        )
        pixel = run(
            mask,
            tiling=True,
            overlap_mode="pixel",
            min_overlap=52,
            max_tile_size=512,
        )
        self.assertEqual(percent.shape, pixel.shape)

        with self.assertRaisesRegex(ValueError, "cannot exceed 90"):
            run(mask, overlap_mode="percent", min_overlap=91)
        with self.assertRaisesRegex(ValueError, "shape must be square or circle"):
            run(mask, shape="triangle")
        with self.assertRaisesRegex(ValueError, r"\[B,H,W\]"):
            run(torch.ones(1, 1, 20, 30))


if __name__ == "__main__":
    unittest.main()
