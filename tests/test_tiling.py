from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import pathlib
import sys
import unittest

import torch

COMFY_ROOT = pathlib.Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
MODULE_PATH = pathlib.Path(__file__).parents[1] / "tiling.py"
sys.path.insert(0, str(COMFY_ROOT))
SPEC = importlib.util.spec_from_file_location("utility_suite_tiling", MODULE_PATH)
tiling = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tiling
SPEC.loader.exec_module(tiling)


def layout_for(shape, mode="fixed_tile", tile=(8, 8), overlap=(2, 2), maximum=(3, 3)):
    return tiling.build_tile_layout(
        shape,
        mode,
        tile[0],
        tile[1],
        overlap[0],
        overlap[1],
        maximum[0],
        maximum[1],
    )


def patterned_image(batch, height, width, channels=3, dtype=torch.float32):
    values = torch.arange(batch * height * width * channels, dtype=torch.float32)
    values = (values % 251) / 250.0
    return values.reshape(batch, height, width, channels).to(dtype)


class GeometryTests(unittest.TestCase):
    def test_fixed_exactly_one_tile(self):
        layout = layout_for((1, 8, 8, 3), tile=(8, 8), overlap=(2, 2))
        self.assertEqual((layout.rows, layout.columns), (1, 1))
        self.assertEqual(layout.spatial_records[0].source_rect, tiling.Rect(0, 0, 8, 8))

    def test_fixed_source_smaller_than_tile(self):
        layout = layout_for((1, 5, 6, 3), tile=(9, 10), overlap=(2, 3))
        self.assertEqual((layout.rows, layout.columns), (1, 1))
        self.assertEqual((layout.tile_tensor_height, layout.tile_tensor_width), (10, 9))
        self.assertEqual(layout.spatial_records[0].valid_rect, tiling.Rect(0, 0, 6, 5))

    def test_fixed_larger_nondivisible_rectangular_and_edge_overlap(self):
        layout = layout_for((1, 15, 20, 3), tile=(8, 7), overlap=(2, 1))
        self.assertEqual((layout.rows, layout.columns), (3, 3))
        last = layout.spatial_records[-1]
        self.assertEqual(last.source_rect, tiling.Rect(12, 8, 20, 15))
        self.assertEqual(last.neighbor_overlaps.left, 2)
        self.assertEqual(last.neighbor_overlaps.top, 5)
        self.assertGreater(last.neighbor_overlaps.top, 1)

    def test_fixed_zero_overlap(self):
        layout = layout_for((1, 8, 12, 3), tile=(4, 4), overlap=(0, 0))
        self.assertEqual((layout.rows, layout.columns), (2, 3))
        self.assertTrue(
            all(record.neighbor_overlaps == tiling.NeighborOverlaps(0, 0, 0, 0)
                for record in layout.spatial_records)
        )

    def test_fixed_zero_nominal_overlap_can_gain_edge_overlap(self):
        layout = layout_for((1, 5, 10, 3), tile=(6, 5), overlap=(0, 0))
        self.assertEqual(layout.columns, 2)
        self.assertEqual(layout.spatial_records[0].neighbor_overlaps.right, 2)
        self.assertEqual(layout.spatial_records[1].neighbor_overlaps.left, 2)

    def test_invalid_requested_overlap(self):
        with self.assertRaisesRegex(ValueError, "half"):
            layout_for((1, 8, 8, 3), tile=(8, 8), overlap=(5, 0))
        with self.assertRaisesRegex(ValueError, "smaller"):
            layout_for((1, 8, 8, 3), tile=(8, 8), overlap=(8, 0))

    def test_bounded_count_below_limit(self):
        layout = layout_for((1, 12, 12, 3), "bounded_grid", tile=(8, 8), overlap=(2, 2), maximum=(4, 4))
        self.assertEqual((layout.rows, layout.columns), (2, 2))
        self.assertEqual((layout.tile_tensor_height, layout.tile_tensor_width), (7, 7))

    def test_bounded_reaches_limit_and_grows_rectangular_tiles(self):
        layout = layout_for((1, 23, 40, 3), "bounded_grid", tile=(10, 8), overlap=(2, 1), maximum=(3, 2))
        self.assertEqual((layout.rows, layout.columns), (2, 3))
        self.assertEqual(layout.tile_tensor_width, 15)
        self.assertEqual(layout.tile_tensor_height, 12)

    def test_bounded_different_xy_overlap(self):
        layout = layout_for((1, 20, 30, 3), "bounded_grid", tile=(12, 9), overlap=(4, 2), maximum=(3, 3))
        self.assertEqual(layout.requested_parameters.overlap_x, 4)
        self.assertEqual(layout.requested_parameters.overlap_y, 2)

    def test_bounded_invalid_overlap_after_derivation(self):
        with self.assertRaisesRegex(ValueError, "half"):
            layout_for((1, 5, 5, 3), "bounded_grid", tile=(10, 10), overlap=(4, 4), maximum=(3, 3))


class BatchAndIdentityTests(unittest.TestCase):
    def assert_identity(self, image, layout, tolerance=1e-6):
        tiles = tiling.tile_image_batch(image, layout)
        result = tiling.untile_image_batch(tiles, layout)
        self.assertEqual(tuple(result.shape), tuple(image.shape))
        self.assertLessEqual((result.float() - image.float()).abs().max().item(), tolerance)

    def test_source_major_row_major_order(self):
        image = torch.zeros((2, 5, 8, 1))
        for source in range(2):
            for y in range(5):
                for x in range(8):
                    image[source, y, x, 0] = source * 1000 + y * 100 + x
        layout = layout_for(tuple(image.shape), tile=(4, 3), overlap=(0, 0))
        tiles = tiling.tile_image_batch(image, layout)
        origins = [int(tile[0, 0, 0].item()) for tile in tiles]
        self.assertEqual(origins, [0, 4, 200, 204, 1000, 1004, 1200, 1204])

    def test_batch_one_overlap_identity(self):
        image = patterned_image(1, 15, 20)
        self.assert_identity(image, layout_for(tuple(image.shape), tile=(8, 7), overlap=(2, 1)))

    def test_batch_two_independent_identity(self):
        image = patterned_image(2, 13, 17)
        self.assert_identity(image, layout_for(tuple(image.shape), tile=(9, 8), overlap=(3, 2)))

    def test_smaller_than_tile_padding_identity(self):
        image = patterned_image(2, 3, 5)
        layout = layout_for(tuple(image.shape), tile=(9, 7), overlap=(2, 2))
        tiles = tiling.tile_image_batch(image, layout)
        self.assertTrue(torch.equal(tiles[:, :3, 5:, :], tiles[:, :3, 4:5, :].expand(-1, -1, 4, -1)))
        self.assertTrue(torch.equal(tiles[:, 3:, :, :], tiles[:, 2:3, :, :].expand(-1, 4, -1, -1)))
        self.assert_identity(image, layout)

    def test_bounded_identity(self):
        image = patterned_image(1, 23, 40)
        layout = layout_for(tuple(image.shape), "bounded_grid", tile=(10, 8), overlap=(2, 1), maximum=(3, 2))
        self.assert_identity(image, layout)

    def test_float16_identity_and_dtype(self):
        image = patterned_image(1, 13, 17, dtype=torch.float16)
        layout = layout_for(tuple(image.shape), tile=(9, 8), overlap=(3, 2))
        tiles = tiling.tile_image_batch(image, layout)
        result = tiling.untile_image_batch(tiles, layout)
        self.assertEqual(result.dtype, torch.float16)
        self.assertLessEqual((result.float() - image.float()).abs().max().item(), 5e-4)

    def test_bfloat16_identity_and_dtype(self):
        image = patterned_image(1, 13, 17, dtype=torch.bfloat16)
        layout = layout_for(tuple(image.shape), tile=(9, 8), overlap=(3, 2))
        tiles = tiling.tile_image_batch(image, layout)
        result = tiling.untile_image_batch(tiles, layout)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertLessEqual((result.float() - image.float()).abs().max().item(), 4e-3)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.image = patterned_image(1, 10, 12)
        self.layout = layout_for(tuple(self.image.shape), tile=(7, 6), overlap=(2, 2))
        self.tiles = tiling.tile_image_batch(self.image, self.layout)

    def test_rank(self):
        with self.assertRaisesRegex(ValueError, "rank-4"):
            tiling.untile_image_batch(self.tiles[0], self.layout)

    def test_missing_and_extra_tiles(self):
        with self.assertRaisesRegex(ValueError, "Tile count mismatch"):
            tiling.untile_image_batch(self.tiles[:-1], self.layout)
        extra = torch.cat((self.tiles, self.tiles[:1]), dim=0)
        with self.assertRaisesRegex(ValueError, "Tile count mismatch"):
            tiling.untile_image_batch(extra, self.layout)

    def test_wrong_width_height_and_channels(self):
        with self.assertRaisesRegex(ValueError, "height mismatch"):
            tiling.untile_image_batch(self.tiles[:, :-1], self.layout)
        with self.assertRaisesRegex(ValueError, "width mismatch"):
            tiling.untile_image_batch(self.tiles[:, :, :-1], self.layout)
        with self.assertRaisesRegex(ValueError, "channel mismatch"):
            tiling.untile_image_batch(self.tiles[..., :-1], self.layout)

    def test_malformed_record_extent(self):
        record = self.layout.spatial_records[0]
        bad_record = dataclasses.replace(record, valid_rect=tiling.Rect(0, 0, record.valid_rect.x1 - 1, record.valid_rect.y1))
        bad_layout = dataclasses.replace(
            self.layout,
            spatial_records=(bad_record,) + self.layout.spatial_records[1:],
        )
        with self.assertRaisesRegex(ValueError, "extents disagree"):
            tiling.untile_image_batch(self.tiles, bad_layout)

    def test_malformed_overlap(self):
        record = self.layout.spatial_records[1]
        bad_record = dataclasses.replace(record, neighbor_overlaps=tiling.NeighborOverlaps(99, 0, 0, 0))
        bad_layout = dataclasses.replace(
            self.layout,
            spatial_records=(self.layout.spatial_records[0], bad_record) + self.layout.spatial_records[2:],
        )
        with self.assertRaisesRegex(ValueError, "overlaps do not match"):
            tiling.untile_image_batch(self.tiles, bad_layout)

    def test_nodes_2_registration_and_schema(self):
        extension = asyncio.run(tiling.comfy_entrypoint())
        node_list = asyncio.run(extension.get_node_list())
        self.assertEqual(node_list, [tiling.ImageTileBatch, tiling.ImageUntileBatch])
        tile_schema = tiling.ImageTileBatch.define_schema()
        untile_schema = tiling.ImageUntileBatch.define_schema()
        self.assertEqual(tile_schema.node_id, "UtilitySuiteImageTileBatch")
        self.assertEqual(untile_schema.node_id, "UtilitySuiteImageUntileBatch")


if __name__ == "__main__":
    unittest.main()
