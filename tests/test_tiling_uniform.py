from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest
from itertools import pairwise

import torch

COMFY_ROOT = pathlib.Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
MODULE_PATH = pathlib.Path(__file__).parents[1] / "tiling.py"
sys.path.insert(0, str(COMFY_ROOT))
SPEC = importlib.util.spec_from_file_location("utility_suite_uniform_tiling", MODULE_PATH)
tiling = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tiling
SPEC.loader.exec_module(tiling)


def uniform_layout(
    shape,
    tile=(512, 512),
    minimum_overlap=(64, 64),
    minimum_count=(1, 1),
):
    return tiling.build_tile_layout(
        shape,
        "uniform_grid",
        tile[0],
        tile[1],
        0,
        0,
        1,
        1,
        minimum_overlap[0],
        minimum_overlap[1],
        minimum_count[0],
        minimum_count[1],
    )


def axis_starts(layout, axis):
    if axis == "x":
        return tuple(layout.spatial_records[column].source_rect.x0 for column in range(layout.columns))
    return tuple(layout.spatial_records[row * layout.columns].source_rect.y0 for row in range(layout.rows))


def patterned_image(batch, height, width, channels=3):
    values = torch.arange(batch * height * width * channels, dtype=torch.float32)
    return ((values % 251) / 250).reshape(batch, height, width, channels)


class UniformGridGeometryTests(unittest.TestCase):
    def test_exact_divisible_distribution(self):
        layout = uniform_layout((1, 512, 1024, 3), minimum_count=(3, 1))
        self.assertEqual(layout.columns, 3)
        self.assertEqual(axis_starts(layout, "x"), (0, 256, 512))
        self.assertEqual(
            tuple(record.neighbor_overlaps.right for record in layout.spatial_records[:-1]),
            (256, 256),
        )

    def test_nondivisible_balanced_integer_distribution(self):
        layout = uniform_layout(
            (1, 400, 1001, 3),
            tile=(400, 400),
            minimum_overlap=(100, 0),
            minimum_count=(4, 1),
        )
        starts = axis_starts(layout, "x")
        strides = tuple(right - left for left, right in pairwise(starts))
        overlaps = tuple(400 - stride for stride in strides)
        self.assertEqual(starts, (0, 200, 401, 601))
        self.assertEqual(strides, (200, 201, 200))
        self.assertEqual(overlaps, (200, 199, 200))

    def test_minimum_count_and_coverage_independently_raise_count(self):
        minimum_count_layout = uniform_layout((1, 512, 1024, 3), minimum_count=(4, 1))
        self.assertEqual(minimum_count_layout.columns, 4)
        coverage_layout = uniform_layout((1, 512, 2000, 3), minimum_count=(2, 1))
        self.assertEqual(coverage_layout.columns, 5)

    def test_minimum_overlap_is_guaranteed_for_every_neighbor(self):
        layout = uniform_layout(
            (1, 777, 1234, 3),
            tile=(400, 300),
            minimum_overlap=(173, 151),
            minimum_count=(5, 4),
        )
        for record in layout.spatial_records:
            for overlap in (record.neighbor_overlaps.left, record.neighbor_overlaps.right):
                if overlap:
                    self.assertGreaterEqual(overlap, 173)
            for overlap in (record.neighbor_overlaps.top, record.neighbor_overlaps.bottom):
                if overlap:
                    self.assertGreaterEqual(overlap, 151)

    def test_fixed_tensor_dimensions_and_independent_axes(self):
        image = patterned_image(1, 701, 1001)
        layout = uniform_layout(
            tuple(image.shape),
            tile=(400, 300),
            minimum_overlap=(100, 80),
            minimum_count=(4, 2),
        )
        tiles = tiling.tile_image_batch(image, layout)
        self.assertEqual((layout.rows, layout.columns), (3, 4))
        self.assertEqual(tuple(tiles.shape), (12, 300, 400, 3))
        self.assertEqual((layout.tile_tensor_height, layout.tile_tensor_width), (300, 400))

    def test_smaller_equal_and_impossible_counts(self):
        smaller = uniform_layout(
            (1, 300, 350, 3),
            tile=(512, 400),
            minimum_overlap=(64, 64),
            minimum_count=(1, 1),
        )
        self.assertEqual((smaller.rows, smaller.columns), (1, 1))
        self.assertEqual(smaller.spatial_records[0].valid_rect, tiling.Rect(0, 0, 350, 300))
        equal = uniform_layout((1, 512, 512, 3), minimum_count=(1, 1))
        self.assertEqual(axis_starts(equal, "x"), (0,))
        with self.assertRaisesRegex(ValueError, "only one distinct tile position"):
            uniform_layout((1, 512, 512, 3), minimum_count=(2, 1))
        with self.assertRaisesRegex(ValueError, "only one distinct tile position"):
            uniform_layout((1, 300, 350, 3), tile=(512, 400), minimum_count=(2, 1))
        with self.assertRaisesRegex(ValueError, "distinct starts"):
            uniform_layout((1, 1, 10, 3), tile=(8, 1), minimum_overlap=(0, 0), minimum_count=(4, 1))

    def test_minimum_overlap_may_exceed_half_tile(self):
        layout = uniform_layout(
            (1, 14, 14, 1),
            tile=(10, 10),
            minimum_overlap=(8, 8),
            minimum_count=(3, 3),
        )
        self.assertEqual(axis_starts(layout, "x"), (0, 2, 4))
        self.assertEqual(axis_starts(layout, "y"), (0, 2, 4))
        self.assertTrue(
            all(
                overlap == 0 or overlap == 8
                for record in layout.spatial_records
                for overlap in (
                    record.neighbor_overlaps.left,
                    record.neighbor_overlaps.top,
                    record.neighbor_overlaps.right,
                    record.neighbor_overlaps.bottom,
                )
            )
        )
        with self.assertRaisesRegex(ValueError, "smaller"):
            uniform_layout((1, 14, 14, 1), tile=(10, 10), minimum_overlap=(10, 8), minimum_count=(3, 3))

    def test_requested_metadata_uses_explicit_minimum_fields(self):
        layout = uniform_layout(
            (1, 600, 900, 3),
            tile=(400, 300),
            minimum_overlap=(111, 77),
            minimum_count=(4, 3),
        )
        requested = layout.requested_parameters
        self.assertEqual(requested.mode, "uniform_grid")
        self.assertEqual((requested.min_overlap_x, requested.min_overlap_y), (111, 77))
        self.assertEqual((requested.min_columns, requested.min_rows), (4, 3))
        self.assertIsNone(requested.overlap_x)
        self.assertIsNone(requested.max_columns)


class UniformGridIntegrationTests(unittest.TestCase):
    def test_source_batch_order_and_bbox_cardinality(self):
        image = torch.zeros((2, 6, 10, 1))
        for source in range(2):
            for y in range(6):
                for x in range(10):
                    image[source, y, x, 0] = source * 1000 + y * 100 + x
        layout = uniform_layout(
            tuple(image.shape),
            tile=(6, 4),
            minimum_overlap=(2, 1),
            minimum_count=(3, 2),
        )
        tiles = tiling.tile_image_batch(image, layout)
        origins = [int(tile[0, 0, 0].item()) for tile in tiles]
        self.assertEqual(origins, [0, 2, 4, 200, 202, 204, 1000, 1002, 1004, 1200, 1202, 1204])
        bboxes, bounding_boxes = tiling.tile_bounding_boxes(layout)
        expected = [(0, 0, 6, 4), (2, 0, 6, 4), (4, 0, 6, 4), (0, 2, 6, 4), (2, 2, 6, 4), (4, 2, 6, 4)]
        self.assertEqual(bboxes, expected + expected)
        self.assertEqual(
            bounding_boxes,
            [{"x": x, "y": y, "width": width, "height": height} for x, y, width, height in expected] * 2,
        )
        self.assertEqual(len(bboxes), tiles.shape[0])

    def test_all_merge_modes_reconstruct_varying_odd_even_overlaps(self):
        image = patterned_image(2, 701, 1001)
        layout = uniform_layout(
            tuple(image.shape),
            tile=(400, 300),
            minimum_overlap=(100, 80),
            minimum_count=(4, 3),
        )
        tiles = tiling.tile_image_batch(image, layout)
        x_overlaps = tuple(400 - (right - left) for left, right in pairwise(axis_starts(layout, "x")))
        y_overlaps = tuple(300 - (bottom - top) for top, bottom in pairwise(axis_starts(layout, "y")))
        self.assertLessEqual(max(x_overlaps) - min(x_overlaps), 1)
        self.assertLessEqual(max(y_overlaps) - min(y_overlaps), 1)
        self.assertTrue(any(overlap % 2 for overlap in x_overlaps + y_overlaps))
        self.assertTrue(any(overlap % 2 == 0 for overlap in x_overlaps + y_overlaps))
        for merge_mode in ("linear", "hard_cut", "limited_linear"):
            with self.subTest(merge_mode=merge_mode):
                result = tiling.untile_image_batch(tiles, layout, merge_mode, 31)
                self.assertEqual(tuple(result.shape), tuple(image.shape))
                self.assertLessEqual((result - image).abs().max().item(), 1e-6)
        hard_cut = tiling.untile_image_batch(tiles, layout, "hard_cut")
        limited_zero = tiling.untile_image_batch(tiles, layout, "limited_linear", 0)
        self.assertTrue(torch.equal(hard_cut, limited_zero))

    def test_all_merge_modes_support_overlap_above_half_tile(self):
        image = patterned_image(1, 14, 14, 1)
        layout = uniform_layout(
            tuple(image.shape),
            tile=(10, 10),
            minimum_overlap=(8, 8),
            minimum_count=(3, 3),
        )
        tiles = tiling.tile_image_batch(image, layout)
        for merge_mode in ("linear", "hard_cut", "limited_linear"):
            with self.subTest(merge_mode=merge_mode):
                result = tiling.untile_image_batch(tiles, layout, merge_mode, 7)
                self.assertLessEqual((result - image).abs().max().item(), 1e-6)

    def test_node_schema_and_execute(self):
        schema = tiling.ImageTileBatch.define_schema()
        self.assertEqual(schema.inputs[1].options, ["fixed_tile", "bounded_grid", "uniform_grid"])
        self.assertEqual(
            [item.id for item in schema.inputs[-4:]],
            ["min_overlap_x", "min_overlap_y", "min_columns", "min_rows"],
        )
        image = patterned_image(1, 512, 1024)
        output = tiling.ImageTileBatch.execute(image, "uniform_grid", 512, 512, 0, 0, 1, 1, 64, 0, 3, 1)
        tiles, layout, bboxes, bounding_boxes = output.result
        self.assertEqual(tuple(tiles.shape), (3, 512, 512, 3))
        self.assertEqual(len(bboxes), 3)
        self.assertEqual(len(bounding_boxes), 3)
        self.assertEqual(axis_starts(layout, "x"), (0, 256, 512))


if __name__ == "__main__":
    unittest.main()
