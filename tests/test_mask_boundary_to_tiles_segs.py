import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_mask_boundary_tiles_segs_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.mask_boundary_to_tiles_segs"]


def run(mask, **overrides):
    values = {
        "ratio": 1.0,
        "overlap_mode": "pixel",
        "min_overlap": 16,
        "feather_overlap": False,
        "max_tile_size": 2048,
    }
    values.update(overrides)
    return MODULE.mask_boundary_to_tiles_segs(mask, **values)


def covered_boundary(entries):
    return [
        min(entry.crop_region[0] for entry in entries),
        min(entry.crop_region[1] for entry in entries),
        max(entry.crop_region[2] for entry in entries),
        max(entry.crop_region[3] for entry in entries),
    ]


class MaskBoundaryToTilesSEGSTests(unittest.TestCase):
    def test_schema_and_registration(self):
        schema = MODULE.MaskBoundaryToTilesSEGS.define_schema()

        self.assertEqual(schema.node_id, "UtilitySuiteMaskBoundaryToTilesSEGS")
        self.assertEqual(schema.display_name, "Mask Boundary to Tiles (SEGS)")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual(
            [item.id for item in schema.inputs],
            [
                "mask",
                "ratio",
                "overlap_mode",
                "min_overlap",
                "feather_overlap",
                "max_tile_size",
            ],
        )
        self.assertEqual(schema.inputs[2].default, "pixel")
        self.assertEqual(schema.inputs[5].default, 2048)
        self.assertEqual(schema.inputs[5].min, 128)
        self.assertEqual(schema.inputs[5].max, 2048)
        nodes = asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list())
        self.assertIn(MODULE.MaskBoundaryToTilesSEGS, nodes)

    def test_disconnected_regions_share_one_fixed_union_boundary(self):
        mask = torch.zeros(1, 100, 160)
        mask[0, 20:30, 10:20] = 1
        mask[0, 60:80, 130:150] = 1

        header, entries = run(mask, max_tile_size=128)

        self.assertEqual(header, (100, 160))
        self.assertEqual(covered_boundary(entries), [10, 20, 150, 80])
        self.assertTrue(all(entry.bbox == (10, 20, 150, 80) for entry in entries))
        self.assertTrue(all(np.all(entry.cropped_mask == 1) for entry in entries))

    def test_ratio_tiles_stay_inside_boundary_without_expansion_or_shrink(self):
        mask = torch.zeros(1, 120, 180)
        mask[0, 30:70, 20:140] = 1

        _, entries = run(mask, ratio=1.0, max_tile_size=128)

        self.assertEqual(covered_boundary(entries), [20, 30, 140, 70])
        for entry in entries:
            x1, y1, x2, y2 = entry.crop_region
            self.assertTrue(20 <= x1 < x2 <= 140)
            self.assertTrue(30 <= y1 < y2 <= 70)
            self.assertEqual(x2 - x1, y2 - y1)

    def test_tile_axes_are_capped_by_crop_for_tall_ratio(self):
        mask = torch.ones(1, 1024, 768)

        _, entries = run(mask, ratio=9 / 16, max_tile_size=2048)

        self.assertEqual(len(entries), 2)
        self.assertEqual(covered_boundary(entries), [0, 0, 768, 1024])
        for entry in entries:
            width = entry.crop_region[2] - entry.crop_region[0]
            height = entry.crop_region[3] - entry.crop_region[1]
            self.assertEqual((width, height), (576, 1024))

    def test_max_tile_size_is_square_root_area_limit(self):
        mask = torch.ones(1, 400, 400)

        _, entries = run(mask, max_tile_size=128)

        self.assertEqual(covered_boundary(entries), [0, 0, 400, 400])
        self.assertTrue(
            all(
                (entry.crop_region[2] - entry.crop_region[0])
                * (entry.crop_region[3] - entry.crop_region[1])
                <= 128 * 128
                for entry in entries
            )
        )

    def test_feathers_are_complementary_across_shared_edge(self):
        mask = torch.ones(1, 50, 100)

        _, entries = run(
            mask,
            min_overlap=10,
            feather_overlap=True,
            max_tile_size=128,
        )

        left, right = entries[:2]
        overlap_start = right.crop_region[0]
        overlap_end = left.crop_region[2]
        left_values = left.cropped_mask[
            25, overlap_start - left.crop_region[0] : overlap_end - left.crop_region[0]
        ]
        right_values = right.cropped_mask[25, : overlap_end - overlap_start]
        self.assertTrue(np.allclose(left_values + right_values, 1.0))

    def test_percent_overlap_resolves_from_each_tile_axis(self):
        mask = torch.ones(1, 50, 100)

        _, percent = run(mask, overlap_mode="percent", min_overlap=20)
        _, pixel = run(mask, overlap_mode="pixel", min_overlap=10)

        self.assertEqual(
            [entry.crop_region for entry in percent],
            [entry.crop_region for entry in pixel],
        )

    def test_percent_overlap_rejects_values_above_ninety_only_in_percent_mode(self):
        mask = torch.ones(1, 400, 400)

        with self.assertRaisesRegex(ValueError, "cannot exceed 90"):
            run(mask, overlap_mode="percent", min_overlap=91, max_tile_size=128)
        _, entries = run(
            mask,
            overlap_mode="pixel",
            min_overlap=91,
            max_tile_size=128,
        )

        self.assertGreater(len(entries), 1)

    def test_batch_items_resolve_boundaries_independently(self):
        masks = torch.zeros(2, 120, 180)
        masks[0, 10:50, 10:130] = 1
        masks[1, 70:110, 80:160] = 1

        _, entries = run(masks, max_tile_size=128)

        first = [entry for entry in entries if entry.bbox == (10, 10, 130, 50)]
        second = [entry for entry in entries if entry.bbox == (80, 70, 160, 110)]
        self.assertEqual(covered_boundary(first), [10, 10, 130, 50])
        self.assertEqual(covered_boundary(second), [80, 70, 160, 110])

    def test_empty_items_contribute_no_segments(self):
        masks = torch.zeros(2, 20, 30)
        masks[1, 5:10, 8:14] = 1

        header, entries = run(masks, max_tile_size=128)
        empty_header, empty_entries = run(torch.zeros(1, 20, 30))

        self.assertEqual(header, (20, 30))
        self.assertEqual(len(entries), 2)
        self.assertTrue(all(entry.bbox == (8, 5, 14, 10) for entry in entries))
        self.assertEqual(empty_header, (20, 30))
        self.assertEqual(empty_entries, [])

    def test_rejects_invalid_inputs(self):
        mask = torch.ones(1, 20, 30)
        with self.assertRaisesRegex(ValueError, "ratio must be finite and positive"):
            run(mask, ratio=0)
        with self.assertRaisesRegex(ValueError, "min_overlap must be positive"):
            run(mask, min_overlap=0)
        with self.assertRaisesRegex(ValueError, "overlap_mode must be pixel or percent"):
            run(mask, overlap_mode="other")
        with self.assertRaisesRegex(ValueError, "between 128 and 2048"):
            run(mask, max_tile_size=127)
        with self.assertRaisesRegex(ValueError, r"\[B,H,W\]"):
            run(torch.ones(1, 1, 20, 30))


if __name__ == "__main__":
    unittest.main()
