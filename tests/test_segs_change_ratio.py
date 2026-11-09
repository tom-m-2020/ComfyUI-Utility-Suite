import asyncio
import importlib.util
import sys
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_segs_change_ratio_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.segs_change_ratio"]

SEG = namedtuple(
    "SEG",
    [
        "cropped_image",
        "cropped_mask",
        "confidence",
        "crop_region",
        "bbox",
        "label",
        "control_net_wrapper",
    ],
)


def segment(crop, mask=None, bbox=None):
    x1, y1, x2, y2 = crop
    if mask is None:
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
    return SEG(object(), mask, 0.75, list(crop), bbox or tuple(crop), "test", object())


def run(segs, **overrides):
    values = {
        "ratio": 1.0,
        "tiling": False,
        "overlap_mode": "pixel",
        "min_overlap": 16,
        "feather_overlap": False,
        "max_tile_size": 2048,
    }
    values.update(overrides)
    return MODULE.segs_change_ratio(segs, **values)


class SEGSChangeRatioTests(unittest.TestCase):
    def test_schema_and_registration(self):
        schema = MODULE.SEGSChangeRatio.define_schema()

        self.assertEqual(schema.node_id, "UtilitySuiteSEGSChangeRatio")
        self.assertEqual(schema.display_name, "SEGS Change Ratio")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual(
            [item.id for item in schema.inputs],
            [
                "segs",
                "ratio",
                "tiling",
                "overlap_mode",
                "min_overlap",
                "feather_overlap",
                "max_tile_size",
            ],
        )
        self.assertEqual(schema.inputs[3].default, "pixel")
        self.assertEqual(schema.inputs[6].default, 2048)
        self.assertEqual(schema.inputs[6].min, 512)
        self.assertEqual(schema.inputs[6].max, 2048)
        nodes = asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list())
        self.assertIn(MODULE.SEGSChangeRatio, nodes)

    def test_square_expands_only_shorter_axis(self):
        source = segment((100, 200, 612, 456))

        result = run(((800, 1000), [source]))

        self.assertEqual(result[1][0].crop_region, [100, 72, 612, 584])
        self.assertEqual(result[1][0].bbox, source.bbox)
        self.assertEqual(result[1][0].confidence, source.confidence)
        self.assertEqual(result[1][0].label, source.label)

    def test_fitting_expansion_shifts_away_from_canvas_edge(self):
        source = segment((0, 10, 200, 110))

        result = run(((300, 400), [source]))

        self.assertEqual(result[1][0].crop_region, [0, 0, 200, 200])

    def test_non_tiled_expansion_truncates_without_shrinking_original_axis(self):
        source = segment((256, 108, 768, 620))

        result = run(((728, 1024), [source]), ratio=9 / 16)

        crop = result[1][0].crop_region
        self.assertEqual(crop, [256, 0, 768, 728])
        self.assertEqual(crop[2] - crop[0], 512)
        self.assertGreaterEqual(crop[3] - crop[1], 512)

    def test_example_tiles_full_height_into_two_ratio_tiles(self):
        source = segment((256, 108, 768, 620))

        result = run(
            ((728, 1024), [source]),
            ratio=9 / 16,
            tiling=True,
            min_overlap=32,
        )

        self.assertEqual(len(result[1]), 2)
        self.assertEqual(
            [entry.crop_region[1::2] for entry in result[1]], [[0, 728], [0, 728]]
        )
        self.assertTrue(
            all(
                entry.crop_region[2] - entry.crop_region[0] == 409
                for entry in result[1]
            )
        )

    def test_max_tile_area_triggers_tiling_only_when_enabled(self):
        source = segment((0, 0, 2048, 2048))
        segs = ((2048, 2048), [source])

        untiled = run(segs, max_tile_size=512)
        tiled = run(segs, tiling=True, min_overlap=32, max_tile_size=512)

        self.assertEqual(len(untiled[1]), 1)
        self.assertEqual(len(tiled[1]), 25)
        for entry in tiled[1]:
            width = entry.crop_region[2] - entry.crop_region[0]
            height = entry.crop_region[3] - entry.crop_region[1]
            self.assertLessEqual(width * height, 512 * 512)
            self.assertEqual(width, height)

    def test_percent_overlap_resolves_from_each_tile_axis(self):
        source = segment((0, 0, 1024, 1024))

        with mock.patch.object(
            MODULE,
            "build_tile_layout",
            wraps=MODULE.build_tile_layout,
        ) as build:
            run(
                ((1024, 1024), [source]),
                ratio=0.5,
                tiling=True,
                overlap_mode="percent",
                min_overlap=20,
                max_tile_size=512,
            )

        args = build.call_args.args
        self.assertEqual((args[2], args[3]), (362, 724))
        self.assertEqual((args[8], args[9]), (73, 145))

    def test_percent_overlap_rejects_values_above_ninety_only_in_percent_mode(
        self,
    ):
        source = segment((0, 0, 1024, 1024))
        segs = ((1024, 1024), [source])

        with self.assertRaisesRegex(ValueError, "cannot exceed 90"):
            run(segs, overlap_mode="percent", min_overlap=91)
        result = run(
            segs,
            tiling=True,
            overlap_mode="pixel",
            min_overlap=91,
            max_tile_size=512,
        )
        self.assertGreater(len(result[1]), 1)

    def test_tile_limit_caps_expansion_to_original_short_side(self):
        source = segment((0, 198, 1477, 712))

        result = run(
            ((1108, 1478), [source]),
            tiling=True,
            min_overlap=96,
            max_tile_size=514,
        )

        self.assertEqual(len(result[1]), 4)
        self.assertEqual(
            [
                min(entry.crop_region[0] for entry in result[1]),
                min(entry.crop_region[1] for entry in result[1]),
                max(entry.crop_region[2] for entry in result[1]),
                max(entry.crop_region[3] for entry in result[1]),
            ],
            [0, 198, 1477, 712],
        )
        self.assertTrue(
            all(
                entry.crop_region[2] - entry.crop_region[0] == 514
                and entry.crop_region[3] - entry.crop_region[1] == 514
                for entry in result[1]
            )
        )

    def test_larger_tile_limit_caps_expansion_at_that_limit(self):
        source = segment((0, 198, 1477, 712))

        result = run(
            ((1108, 1478), [source]),
            tiling=True,
            min_overlap=96,
            max_tile_size=1024,
        )

        self.assertEqual(len(result[1]), 2)
        self.assertEqual(
            [
                min(entry.crop_region[0] for entry in result[1]),
                min(entry.crop_region[1] for entry in result[1]),
                max(entry.crop_region[2] for entry in result[1]),
                max(entry.crop_region[3] for entry in result[1]),
            ],
            [0, 0, 1477, 1024],
        )
        self.assertTrue(
            all(
                entry.crop_region[2] - entry.crop_region[0] == 1024
                and entry.crop_region[3] - entry.crop_region[1] == 1024
                for entry in result[1]
            )
        )

    def test_existing_large_crop_never_shrinks_for_either_tile_limit(self):
        source = segment((0, 0, 1477, 1108))
        segs = ((1108, 1478), [source])

        small = run(segs, tiling=True, min_overlap=96, max_tile_size=514)
        large = run(segs, tiling=True, min_overlap=96, max_tile_size=2048)

        for result in (small, large):
            self.assertEqual(
                [
                    min(entry.crop_region[0] for entry in result[1]),
                    min(entry.crop_region[1] for entry in result[1]),
                    max(entry.crop_region[2] for entry in result[1]),
                    max(entry.crop_region[3] for entry in result[1]),
                ],
                [0, 0, 1477, 1108],
            )
        self.assertEqual(len(large[1]), 2)
        self.assertTrue(
            all(
                entry.crop_region[2] - entry.crop_region[0] == 1108
                and entry.crop_region[3] - entry.crop_region[1] == 1108
                for entry in large[1]
            )
        )

    def test_original_mask_stays_in_global_position_across_tiles(self):
        mask = np.zeros((512, 512), dtype=np.float32)
        mask[200:312, 240:272] = 0.625
        source = segment((256, 108, 768, 620), mask, bbox=(496, 308, 528, 420))

        result = run(
            ((728, 1024), [source]),
            ratio=9 / 16,
            tiling=True,
            min_overlap=32,
        )

        reconstructed = np.zeros((728, 1024), dtype=np.float32)
        for entry in result[1]:
            x1, y1, x2, y2 = entry.crop_region
            reconstructed[y1:y2, x1:x2] = np.maximum(
                reconstructed[y1:y2, x1:x2], entry.cropped_mask
            )
            self.assertEqual(entry.bbox, source.bbox)
        expected = np.zeros_like(reconstructed)
        expected[308:420, 496:528] = 0.625
        self.assertTrue(np.array_equal(reconstructed, expected))

    def test_feather_overlap_multiplies_existing_soft_mask(self):
        mask = np.full((512, 512), 0.8, dtype=np.float32)
        source = segment((256, 108, 768, 620), mask)

        result = run(
            ((728, 1024), [source]),
            ratio=9 / 16,
            tiling=True,
            min_overlap=32,
            feather_overlap=True,
        )

        left, right = result[1]
        overlap_start = right.crop_region[0]
        overlap_end = left.crop_region[2]
        y = 300
        left_values = left.cropped_mask[
            y, overlap_start - left.crop_region[0] : overlap_end - left.crop_region[0]
        ]
        right_values = right.cropped_mask[y, : overlap_end - overlap_start]
        self.assertTrue(np.allclose(left_values + right_values, 0.8))
        self.assertEqual(float(left_values[0]), float(np.float32(0.8)))
        self.assertEqual(float(left_values[-1]), 0.0)
        self.assertEqual(float(right_values[0]), 0.0)
        self.assertEqual(float(right_values[-1]), float(np.float32(0.8)))

    def test_feather_overlap_multiplies_ramps_at_two_axis_corner(self):
        record = PACKAGE.tiling.SpatialTileRecord(
            row=0,
            column=0,
            source_rect=PACKAGE.tiling.Rect(0, 0, 5, 5),
            valid_rect=PACKAGE.tiling.Rect(0, 0, 5, 5),
            neighbor_overlaps=PACKAGE.tiling.NeighborOverlaps(0, 0, 3, 3),
        )
        feather = MODULE._feather_mask(record)
        ramp = np.array([1.0, 0.5, 0.0], dtype=np.float32)

        self.assertTrue(np.array_equal(feather[:2, :2], np.ones((2, 2))))
        self.assertTrue(
            np.array_equal(feather[-3:, -3:], ramp[:, None] * ramp[None, :])
        )

    def test_multiple_entries_expand_in_place_and_preserve_order(self):
        first = segment((10, 20, 110, 70), bbox=(20, 25, 30, 35))
        second = segment((200, 100, 260, 220), bbox=(210, 110, 220, 120))

        result = run(((400, 500), [first, second]))

        self.assertEqual(len(result[1]), 2)
        self.assertEqual([entry.bbox for entry in result[1]], [first.bbox, second.bbox])
        self.assertEqual(result[1][0].crop_region, [10, 0, 110, 100])
        self.assertEqual(result[1][1].crop_region, [170, 100, 290, 220])

    def test_rejects_invalid_ratio_and_malformed_mask(self):
        source = segment((0, 0, 10, 10), np.zeros((9, 10), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "ratio must be finite and positive"):
            run(((20, 20), [source]), ratio=0)
        with self.assertRaisesRegex(ValueError, "overlap_mode must be pixel or percent"):
            run(((20, 20), [source]), overlap_mode="other")
        with self.assertRaisesRegex(ValueError, "does not match crop_region"):
            run(((20, 20), [source]))


if __name__ == "__main__":
    unittest.main()
