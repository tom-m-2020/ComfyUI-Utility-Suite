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

    def test_explicit_linear_matches_default_and_previous_values(self):
        layout = layout_for((1, 1, 9, 1), tile=(6, 1), overlap=(2, 0))
        tiles = torch.zeros((2, 1, 6, 1))
        tiles[1] = 1
        default = tiling.untile_image_batch(tiles, layout)
        explicit = tiling.untile_image_batch(tiles, layout, "linear")
        expected = torch.tensor([0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1]).reshape(1, 1, 9, 1)
        self.assertTrue(torch.equal(default, explicit))
        self.assertTrue(torch.equal(explicit, expected))


class HardCutTests(unittest.TestCase):
    @staticmethod
    def constant_tiles(layout, values, dtype=torch.float32, device="cpu"):
        tiles = torch.empty(
            (
                layout.required_tile_count,
                layout.tile_tensor_height,
                layout.tile_tensor_width,
                layout.channels,
            ),
            dtype=dtype,
            device=device,
        )
        for index, value in enumerate(values):
            tiles[index].fill_(value)
        return tiles

    def test_one_by_one_and_padding_exclusion(self):
        image = patterned_image(1, 3, 4)
        layout = layout_for(tuple(image.shape), tile=(7, 6), overlap=(2, 2))
        tiles = tiling.tile_image_batch(image, layout)
        tiles[:, 3:, :, :] = 99
        tiles[:, :, 4:, :] = 99
        output = tiling.untile_image_batch(tiles, layout, "hard_cut")
        self.assertTrue(torch.equal(output, image))
        self.assertEqual(output.dtype, image.dtype)
        self.assertEqual(output.device, image.device)

    def test_one_row_two_columns_even_overlap(self):
        layout = layout_for((1, 1, 10, 1), tile=(6, 1), overlap=(2, 0))
        tiles = self.constant_tiles(layout, [0, 1])
        output = tiling.untile_image_batch(tiles, layout, "hard_cut").flatten()
        self.assertTrue(torch.equal(output, torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])))

    def test_one_row_two_columns_odd_overlap_extra_goes_right(self):
        layout = layout_for((1, 1, 9, 1), tile=(6, 1), overlap=(2, 0))
        self.assertEqual(layout.spatial_records[0].neighbor_overlaps.right, 3)
        tiles = self.constant_tiles(layout, [0, 1])
        output = tiling.untile_image_batch(tiles, layout, "hard_cut").flatten()
        self.assertTrue(torch.equal(output, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1])))

    def test_two_rows_one_column_even_overlap(self):
        layout = layout_for((1, 10, 1, 1), tile=(1, 6), overlap=(0, 2))
        tiles = self.constant_tiles(layout, [2, 8])
        output = tiling.untile_image_batch(tiles, layout, "hard_cut").flatten()
        self.assertTrue(torch.equal(output, torch.tensor([2, 2, 2, 2, 2, 8, 8, 8, 8, 8])))

    def test_two_rows_one_column_odd_overlap_extra_goes_bottom(self):
        layout = layout_for((1, 9, 1, 1), tile=(1, 6), overlap=(0, 2))
        tiles = self.constant_tiles(layout, [2, 8])
        output = tiling.untile_image_batch(tiles, layout, "hard_cut").flatten()
        self.assertTrue(torch.equal(output, torch.tensor([2, 2, 2, 2, 8, 8, 8, 8, 8])))

    def test_two_by_two_four_way_intersection(self):
        layout = layout_for((1, 9, 9, 1), tile=(6, 6), overlap=(2, 2))
        tiles = self.constant_tiles(layout, [0, 1, 2, 3])
        output = tiling.untile_image_batch(tiles, layout, "hard_cut")[0, :, :, 0]
        expected = torch.empty((9, 9))
        expected[:4, :4] = 0
        expected[:4, 4:] = 1
        expected[4:, :4] = 2
        expected[4:, 4:] = 3
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(set(output.unique().tolist()), {0.0, 1.0, 2.0, 3.0})

    def test_three_by_three_zero_nominal_overlap_rectangular_tiles(self):
        layout = layout_for((1, 11, 14, 1), tile=(6, 5), overlap=(0, 0))
        self.assertEqual((layout.rows, layout.columns), (3, 3))
        tiles = self.constant_tiles(layout, range(9))
        output = tiling.untile_image_batch(tiles, layout, "hard_cut")
        self.assertEqual(tuple(output.shape), (1, 11, 14, 1))
        self.assertEqual(set(output.unique().tolist()), {float(value) for value in range(9)})

    def test_rectangular_tiles_and_different_xy_overlaps_identity(self):
        image = patterned_image(1, 17, 23)
        layout = layout_for(tuple(image.shape), tile=(10, 7), overlap=(4, 2))
        tiles = tiling.tile_image_batch(image, layout)
        output = tiling.untile_image_batch(tiles, layout, "hard_cut")
        self.assertTrue(torch.equal(output, image))

    def test_batch_two_source_major_and_independent(self):
        layout = layout_for((2, 1, 9, 1), tile=(6, 1), overlap=(2, 0))
        tiles = self.constant_tiles(layout, [0, 1, 10, 11])
        output = tiling.untile_image_batch(tiles, layout, "hard_cut")[:, 0, :, 0]
        expected = torch.tensor(
            [
                [0, 0, 0, 0, 1, 1, 1, 1, 1],
                [10, 10, 10, 10, 11, 11, 11, 11, 11],
            ]
        )
        self.assertTrue(torch.equal(output, expected))

    def test_hard_cut_preserves_low_precision_dtype(self):
        layout = layout_for((1, 1, 9, 1), tile=(6, 1), overlap=(2, 0))
        for dtype in (torch.float16, torch.bfloat16):
            tiles = self.constant_tiles(layout, [0, 1], dtype=dtype)
            output = tiling.untile_image_batch(tiles, layout, "hard_cut")
            self.assertEqual(output.dtype, dtype)
            self.assertEqual(output.device, tiles.device)

    def test_legacy_linear_layout_supports_both_modes(self):
        layout = layout_for((1, 1, 9, 1), tile=(6, 1), overlap=(2, 0))
        legacy = dataclasses.replace(layout, merge_policy=tiling.LEGACY_LINEAR_MERGE_POLICY)
        tiles = self.constant_tiles(layout, [0, 1])
        tiling.untile_image_batch(tiles, legacy, "linear")
        hard = tiling.untile_image_batch(tiles, legacy, "hard_cut")
        self.assertEqual(tuple(hard.shape), (1, 1, 9, 1))


class LimitedLinearTests(unittest.TestCase):
    @staticmethod
    def constant_tiles(layout, values, dtype=torch.float32, device="cpu"):
        return HardCutTests.constant_tiles(layout, values, dtype=dtype, device=device)

    def test_one_by_one_and_padding_exclusion(self):
        image = patterned_image(1, 3, 4)
        layout = layout_for(tuple(image.shape), tile=(7, 6), overlap=(2, 2))
        tiles = tiling.tile_image_batch(image, layout)
        tiles[:, 3:, :, :] = 99
        tiles[:, :, 4:, :] = 99
        output = tiling.untile_image_batch(tiles, layout, "limited_linear", 3)
        self.assertTrue(torch.equal(output, image))

    def test_horizontal_limited_even_width(self):
        layout = layout_for((1, 1, 18, 1), tile=(12, 1), overlap=(6, 0))
        tiles = self.constant_tiles(layout, [0, 1])
        output = tiling.untile_image_batch(tiles, layout, "limited_linear", 2).flatten()
        expected = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1 / 3, 2 / 3, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertTrue(torch.allclose(output, expected))
        self.assertTrue(torch.equal(output[:8], torch.zeros(8)))
        self.assertTrue(torch.equal(output[10:], torch.ones(8)))

    def test_horizontal_limited_odd_width_extra_goes_after_cut(self):
        layout = layout_for((1, 1, 18, 1), tile=(12, 1), overlap=(6, 0))
        tiles = self.constant_tiles(layout, [0, 1])
        output = tiling.untile_image_batch(tiles, layout, "limited_linear", 3).flatten()
        expected = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1, 1, 1, 1])
        self.assertTrue(torch.equal(output, expected))

    def test_vertical_limited_width(self):
        layout = layout_for((1, 18, 1, 1), tile=(1, 12), overlap=(0, 6))
        tiles = self.constant_tiles(layout, [0, 1])
        output = tiling.untile_image_batch(tiles, layout, "limited_linear", 2).flatten()
        expected = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1 / 3, 2 / 3, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertTrue(torch.allclose(output, expected))

    def test_width_zero_equals_hard_cut_exactly(self):
        layout = layout_for((1, 9, 9, 1), tile=(6, 6), overlap=(2, 2))
        tiles = self.constant_tiles(layout, [0, 1, 2, 3])
        hard = tiling.untile_image_batch(tiles, layout, "hard_cut")
        limited = tiling.untile_image_batch(tiles, layout, "limited_linear", 0)
        self.assertTrue(torch.equal(limited, hard))

    def test_width_one_blends_only_midpoint_pixel(self):
        layout = layout_for((1, 1, 18, 1), tile=(12, 1), overlap=(6, 0))
        tiles = self.constant_tiles(layout, [0, 1])
        output = tiling.untile_image_batch(tiles, layout, "limited_linear", 1).flatten()
        self.assertTrue(torch.equal(output[:9], torch.zeros(9)))
        self.assertEqual(output[9].item(), 0.5)
        self.assertTrue(torch.equal(output[10:], torch.ones(8)))

    def test_width_equal_to_overlap_matches_linear(self):
        layout = layout_for((1, 1, 18, 1), tile=(12, 1), overlap=(6, 0))
        tiles = self.constant_tiles(layout, [0, 1])
        linear = tiling.untile_image_batch(tiles, layout, "linear")
        limited = tiling.untile_image_batch(tiles, layout, "limited_linear", 6)
        self.assertTrue(torch.equal(limited, linear))

    def test_width_greater_than_overlap_uses_effective_overlap(self):
        layout = layout_for((1, 1, 18, 1), tile=(12, 1), overlap=(6, 0))
        tiles = self.constant_tiles(layout, [0, 1])
        equal = tiling.untile_image_batch(tiles, layout, "limited_linear", 6)
        greater = tiling.untile_image_batch(tiles, layout, "limited_linear", 99)
        self.assertTrue(torch.equal(greater, equal))

    def test_two_by_two_four_way_band_locality(self):
        layout = layout_for((1, 18, 18, 1), tile=(12, 12), overlap=(6, 6))
        tiles = self.constant_tiles(layout, [0, 10, 30, 70])
        output = tiling.untile_image_batch(tiles, layout, "limited_linear", 2)[0, :, :, 0]
        self.assertEqual(output[7, 7].item(), 0)
        self.assertAlmostEqual(output[7, 8].item(), 10 / 3, places=5)
        self.assertAlmostEqual(output[8, 7].item(), 10, places=5)
        self.assertAlmostEqual(output[8, 8].item(), 50 / 3, places=5)
        self.assertEqual(output[10, 10].item(), 70)

    def test_three_by_three_and_zero_nominal_edge_overlap(self):
        layout = layout_for((1, 22, 22, 1), tile=(10, 10), overlap=(4, 4))
        tiles = self.constant_tiles(layout, range(9))
        output = tiling.untile_image_batch(tiles, layout, "limited_linear", 2)
        self.assertEqual(tuple(output.shape), (1, 22, 22, 1))

        edge_layout = layout_for((1, 1, 19, 1), tile=(10, 1), overlap=(0, 0))
        self.assertEqual(edge_layout.spatial_records[0].neighbor_overlaps.right, 1)
        edge_tiles = self.constant_tiles(edge_layout, [0, 1])
        edge_output = tiling.untile_image_batch(edge_tiles, edge_layout, "limited_linear", 8).flatten()
        self.assertEqual(edge_output[9].item(), 0.5)

    def test_rectangular_different_xy_actual_overlaps_identity(self):
        image = patterned_image(1, 13, 14)
        layout = layout_for(tuple(image.shape), tile=(10, 9), overlap=(4, 2))
        self.assertNotEqual(
            layout.spatial_records[0].neighbor_overlaps.right,
            layout.spatial_records[0].neighbor_overlaps.bottom,
        )
        tiles = tiling.tile_image_batch(image, layout)
        output = tiling.untile_image_batch(tiles, layout, "limited_linear", 3)
        self.assertLessEqual((output - image).abs().max().item(), 2e-7)

    def test_batch_two_independent(self):
        layout = layout_for((2, 1, 18, 1), tile=(12, 1), overlap=(6, 0))
        tiles = self.constant_tiles(layout, [0, 1, 10, 11])
        output = tiling.untile_image_batch(tiles, layout, "limited_linear", 2)[:, 0, :, 0]
        self.assertTrue(torch.allclose(output[1] - output[0], torch.full((18,), 10.0), atol=1e-6))

    def test_low_precision_float32_accumulation_and_return_dtype(self):
        layout = layout_for((1, 1, 18, 1), tile=(12, 1), overlap=(6, 0))
        for dtype in (torch.float16, torch.bfloat16):
            tiles = self.constant_tiles(layout, [0, 1], dtype=dtype)
            output = tiling.untile_image_batch(tiles, layout, "limited_linear", 3)
            self.assertEqual(output.dtype, dtype)
            self.assertEqual(output.device, tiles.device)
            expected = torch.tensor([0.25, 0.5, 0.75], dtype=dtype)
            self.assertTrue(torch.equal(output.flatten()[8:11], expected))

    def test_legacy_layout_support(self):
        layout = layout_for((1, 1, 18, 1), tile=(12, 1), overlap=(6, 0))
        legacy = dataclasses.replace(layout, merge_policy=tiling.LEGACY_LINEAR_MERGE_POLICY)
        tiles = self.constant_tiles(layout, [0, 1])
        output = tiling.untile_image_batch(tiles, legacy, "limited_linear", 2)
        self.assertEqual(tuple(output.shape), (1, 1, 18, 1))

    def test_negative_width_rejected_only_for_limited_mode(self):
        layout = layout_for((1, 1, 18, 1), tile=(12, 1), overlap=(6, 0))
        tiles = self.constant_tiles(layout, [0, 1])
        with self.assertRaisesRegex(ValueError, "blend_width must be nonnegative"):
            tiling.untile_image_batch(tiles, layout, "limited_linear", -1)
        tiling.untile_image_batch(tiles, layout, "linear", -1)
        tiling.untile_image_batch(tiles, layout, "hard_cut", -1)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.image = patterned_image(1, 10, 12)
        self.layout = layout_for(tuple(self.image.shape), tile=(7, 6), overlap=(2, 2))
        self.tiles = tiling.tile_image_batch(self.image, self.layout)

    def test_rank(self):
        with self.assertRaisesRegex(ValueError, "rank-4"):
            tiling.untile_image_batch(self.tiles[0], self.layout)

    def test_missing_and_extra_tiles(self):
        for merge_mode in ("linear", "hard_cut"):
            with self.assertRaisesRegex(ValueError, "Tile count mismatch"):
                tiling.untile_image_batch(self.tiles[:-1], self.layout, merge_mode)
            extra = torch.cat((self.tiles, self.tiles[:1]), dim=0)
            with self.assertRaisesRegex(ValueError, "Tile count mismatch"):
                tiling.untile_image_batch(extra, self.layout, merge_mode)

    def test_invalid_merge_mode(self):
        with self.assertRaisesRegex(ValueError, "Unsupported merge_mode"):
            tiling.untile_image_batch(self.tiles, self.layout, "unknown")

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
        self.assertEqual(untile_schema.inputs[2].options, ["linear", "hard_cut", "limited_linear"])
        self.assertEqual(untile_schema.inputs[2].default, "linear")
        self.assertEqual(untile_schema.inputs[3].id, "blend_width")
        self.assertEqual(untile_schema.inputs[3].default, 32)
        self.assertEqual([output.io_type for output in tile_schema.outputs], ["IMAGE", "TILE_LAYOUT", "BBOX", "BOUNDING_BOX"])
        self.assertFalse(tile_schema.outputs[0].is_output_list)
        self.assertTrue(tile_schema.outputs[2].is_output_list)
        self.assertTrue(tile_schema.outputs[3].is_output_list)


class TileBoundingBoxTests(unittest.TestCase):
    def _resolve(self, image_shape, mode, tile_width, tile_height, overlap_x, overlap_y, max_columns=3, max_rows=3):
        layout = tiling.build_tile_layout(
            image_shape,
            mode,
            tile_width,
            tile_height,
            overlap_x,
            overlap_y,
            max_columns,
            max_rows,
        )
        return layout, tiling.tile_bounding_boxes(layout)

    def test_one_image_one_tile(self):
        layout, (bboxes, bounding_boxes) = self._resolve((1, 64, 96, 3), "fixed_tile", 96, 64, 0, 0)
        self.assertEqual(layout.required_tile_count, 1)
        self.assertEqual(bboxes, [(0, 0, 96, 64)])
        self.assertEqual(bounding_boxes, [{"x": 0, "y": 0, "width": 96, "height": 64}])

    def test_edge_aligned_two_by_two_exact_values(self):
        layout, (bboxes, bounding_boxes) = self._resolve((1, 90, 110, 3), "fixed_tile", 64, 56, 16, 12)
        self.assertEqual((layout.rows, layout.columns), (2, 2))
        self.assertEqual(bboxes, [(0, 0, 64, 56), (46, 0, 64, 56), (0, 34, 64, 56), (46, 34, 64, 56)])
        self.assertEqual(
            bounding_boxes,
            [{"x": x, "y": y, "width": width, "height": height} for x, y, width, height in bboxes],
        )

    def test_source_batch_duplicates_spatial_order(self):
        layout, (bboxes, bounding_boxes) = self._resolve((2, 90, 110, 3), "fixed_tile", 64, 56, 16, 12)
        spatial = [(0, 0, 64, 56), (46, 0, 64, 56), (0, 34, 64, 56), (46, 34, 64, 56)]
        self.assertEqual(bboxes, spatial + spatial)
        self.assertEqual(len(bboxes), layout.required_tile_count)
        self.assertEqual(len(bounding_boxes), layout.required_tile_count)

    def test_bbox_excludes_replicate_padding(self):
        layout, (bboxes, bounding_boxes) = self._resolve((1, 30, 40, 3), "fixed_tile", 64, 56, 16, 12)
        self.assertEqual((layout.tile_tensor_height, layout.tile_tensor_width), (56, 64))
        self.assertEqual(bboxes, [(0, 0, 40, 30)])
        self.assertEqual(bounding_boxes[0], {"x": 0, "y": 0, "width": 40, "height": 30})

    def test_bounded_grid_uses_resolved_source_rectangles(self):
        layout, (bboxes, bounding_boxes) = self._resolve((1, 140, 220, 3), "bounded_grid", 80, 72, 16, 8, 2, 2)
        expected = [
            (record.source_rect.x0, record.source_rect.y0, record.source_rect.width, record.source_rect.height)
            for record in layout.spatial_records
        ]
        self.assertEqual(bboxes, expected)
        self.assertEqual(
            bounding_boxes,
            [{"x": x, "y": y, "width": width, "height": height} for x, y, width, height in expected],
        )

    def test_node_execute_cardinality_matches_tiles(self):
        image = torch.rand((2, 90, 110, 3))
        output = tiling.ImageTileBatch.execute(image, "fixed_tile", 64, 56, 16, 12, 3, 3)
        tiles, layout, bboxes, bounding_boxes = output.result
        self.assertEqual(tiles.shape[0], layout.required_tile_count)
        self.assertEqual(tiles.shape[0], len(bboxes))
        self.assertEqual(tiles.shape[0], len(bounding_boxes))


if __name__ == "__main__":
    unittest.main()
