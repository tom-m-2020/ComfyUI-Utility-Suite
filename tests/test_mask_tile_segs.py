from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_mask_tile_segs_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.mask_tile_segs"]
TILING = sys.modules[f"{SPEC.name}.tiling"]


def convert(
    mask: torch.Tensor,
    mode: str = "fixed_tile",
    tile: tuple[int, int] = (8, 8),
    overlap: tuple[int, int] = (2, 2),
    maximum: tuple[int, int] = (3, 3),
    minimum_overlap: tuple[int, int] = (2, 2),
    minimum_count: tuple[int, int] = (1, 1),
    image: torch.Tensor | None = None,
):
    return MODULE.mask_to_tile_segs(
        mask,
        mode,
        tile[0],
        tile[1],
        overlap[0],
        overlap[1],
        maximum[0],
        maximum[1],
        minimum_overlap[0],
        minimum_overlap[1],
        minimum_count[0],
        minimum_count[1],
        image,
    )


class MaskToTileSEGSTests(unittest.TestCase):
    def test_required_2048_by_1024_acceptance_case(self):
        mask = torch.zeros((1, 1024, 2048), dtype=torch.float32)
        mask[0, :, :694] = 0.25
        mask[0, :, 1354:] = 0.75

        segs, tiles = convert(mask, tile=(1024, 1024), overlap=(0, 0))

        self.assertEqual(segs[0], (1024, 2048))
        self.assertEqual(len(segs[1]), 2)
        self.assertEqual(segs[1][0].crop_region, [0, 0, 1024, 1024])
        self.assertEqual(segs[1][1].crop_region, [1024, 0, 2048, 1024])
        self.assertEqual(segs[1][0].bbox, (0, 0, 694, 1024))
        self.assertEqual(segs[1][1].bbox, (1354, 0, 2048, 1024))
        self.assertEqual(segs[1][0].cropped_mask.shape, (1024, 1024))
        self.assertEqual(segs[1][1].cropped_mask.shape, (1024, 1024))
        np.testing.assert_array_equal(segs[1][0].cropped_mask, mask[0, :, :1024].numpy())
        np.testing.assert_array_equal(segs[1][1].cropped_mask, mask[0, :, 1024:].numpy())
        self.assertEqual(tuple(tiles.shape), (2, 1024, 1024))
        self.assertTrue(torch.equal(tiles[0], mask[0, :, :1024]))
        self.assertTrue(torch.equal(tiles[1], mask[0, :, 1024:]))

    def test_geometry_matches_authoritative_layout_for_all_modes(self):
        cases = [
            ("fixed_tile", (17, 25), (10, 8), (2, 3), (4, 4), (1, 1), (1, 1)),
            ("bounded_grid", (23, 40), (10, 8), (2, 1), (3, 2), (1, 1), (1, 1)),
            ("uniform_grid", (21, 31), (12, 9), (0, 0), (2, 2), (3, 2), (4, 3)),
        ]
        for mode, (height, width), tile, overlap, maximum, minimum_overlap, minimum_count in cases:
            with self.subTest(mode=mode):
                mask = torch.ones((1, height, width))
                segs, tiles = convert(mask, mode, tile, overlap, maximum, minimum_overlap, minimum_count)
                layout = TILING.build_tile_layout(
                    (1, height, width, 1),
                    mode,
                    tile[0],
                    tile[1],
                    overlap[0],
                    overlap[1],
                    maximum[0],
                    maximum[1],
                    minimum_overlap[0],
                    minimum_overlap[1],
                    minimum_count[0],
                    minimum_count[1],
                )
                expected = [
                    [r.source_rect.x0, r.source_rect.y0, r.source_rect.x1, r.source_rect.y1]
                    for r in layout.spatial_records
                ]
                self.assertEqual([seg.crop_region for seg in segs[1]], expected)
                self.assertEqual(len(segs[1]), len(layout.spatial_records))
                self.assertEqual(
                    tuple(tiles.shape),
                    (len(layout.spatial_records), layout.tile_tensor_height, layout.tile_tensor_width),
                )

    def test_rectangular_tiles_xy_overlap_and_row_major_order(self):
        mask = torch.ones((1, 13, 22))
        segs, _ = convert(mask, tile=(9, 7), overlap=(3, 2), maximum=(8, 8))
        origins = [(seg.crop_region[0], seg.crop_region[1]) for seg in segs[1]]
        self.assertEqual(origins, sorted(origins, key=lambda p: (p[1], p[0])))
        self.assertGreater(len({x for x, _ in origins}), 1)
        self.assertGreater(len({y for _, y in origins}), 1)

    def test_grayscale_values_are_preserved_and_quantization_only_controls_bbox(self):
        mask = torch.zeros((1, 6, 8), dtype=torch.float32)
        mask[0, 1, 1] = 0.001
        mask[0, 2, 3] = 0.004
        mask[0, 4, 6] = 0.8
        seg, tile = convert(mask, tile=(8, 6), overlap=(0, 0))
        entry = seg[1][0]
        self.assertEqual(entry.bbox, (3, 2, 7, 5))
        np.testing.assert_array_equal(entry.cropped_mask, mask[0].numpy())
        self.assertTrue(torch.equal(tile[0], mask[0]))
        self.assertEqual(entry.cropped_mask[1, 1], np.float32(0.001))

    def test_bbox_is_source_offset_and_uses_exclusive_edges(self):
        mask = torch.zeros((1, 6, 12))
        mask[0, 0:6, 0:2] = 1
        mask[0, 1:6, 10:12] = 1
        segs, _ = convert(mask, tile=(6, 6), overlap=(0, 0))
        self.assertEqual(segs[1][0].bbox, (0, 0, 2, 6))
        self.assertEqual(segs[1][1].bbox, (10, 1, 12, 6))

    def test_empty_tile_is_retained_with_crop_region_bbox_fallback(self):
        mask = torch.zeros((1, 6, 12))
        mask[0, 1:3, 1:3] = 1
        segs, tiles = convert(mask, tile=(6, 6), overlap=(0, 0))
        self.assertEqual(len(segs[1]), 2)
        self.assertEqual(segs[1][1].crop_region, [6, 0, 12, 6])
        self.assertEqual(segs[1][1].bbox, (6, 0, 12, 6))
        self.assertEqual(np.count_nonzero(segs[1][1].cropped_mask), 0)
        self.assertEqual(torch.count_nonzero(tiles[1]).item(), 0)

    def test_optional_image_absent_and_present_exact_crop(self):
        mask = torch.ones((1, 5, 10))
        without_image, _ = convert(mask, tile=(6, 5), overlap=(2, 0))
        self.assertTrue(all(seg.cropped_image is None for seg in without_image[1]))

        image = torch.arange(1 * 5 * 10 * 3, dtype=torch.float32).reshape(1, 5, 10, 3)
        with_image, _ = convert(mask, tile=(6, 5), overlap=(2, 0), image=image)
        for seg in with_image[1]:
            x0, y0, x1, y1 = seg.crop_region
            np.testing.assert_array_equal(seg.cropped_image, image[:, y0:y1, x0:x1, :].numpy())
            self.assertEqual(seg.cropped_image.shape, (1, y1 - y0, x1 - x0, 3))

    def test_image_validation(self):
        mask = torch.ones((1, 5, 10))
        with self.assertRaisesRegex(ValueError, "spatial dimensions to match"):
            convert(mask, image=torch.ones((1, 6, 10, 3)))
        with self.assertRaisesRegex(ValueError, "IMAGE batch size 1"):
            convert(mask, image=torch.ones((2, 5, 10, 3)))
        with self.assertRaisesRegex(ValueError, r"IMAGE \[1,H,W,C\]"):
            convert(mask, image=torch.ones((5, 10, 3)))

    def test_mask_rank_and_batch_validation(self):
        segs, tiles = convert(torch.ones((5, 7)), tile=(7, 5), overlap=(0, 0))
        self.assertEqual(segs[0], (5, 7))
        self.assertEqual(tuple(tiles.shape), (1, 5, 7))
        with self.assertRaisesRegex(ValueError, "MASK batch size 1"):
            convert(torch.ones((2, 5, 7)))
        with self.assertRaisesRegex(ValueError, r"MASK \[H,W\]"):
            convert(torch.ones((1, 1, 5, 7)))

    def test_source_smaller_than_tile_zero_pads_only_mask_batch(self):
        mask = torch.tensor([[[0.2, 0.4, 0.6], [0.8, 1.0, 0.3]]], dtype=torch.float32)
        segs, tiles = convert(mask, tile=(6, 5), overlap=(1, 1))
        entry = segs[1][0]
        self.assertEqual(entry.crop_region, [0, 0, 3, 2])
        self.assertEqual(entry.cropped_mask.shape, (2, 3))
        np.testing.assert_array_equal(entry.cropped_mask, mask[0].numpy())
        self.assertEqual(tuple(tiles.shape), (1, 5, 6))
        self.assertTrue(torch.equal(tiles[0, :2, :3], mask[0]))
        self.assertEqual(torch.count_nonzero(tiles[0, 2:, :]).item(), 0)
        self.assertEqual(torch.count_nonzero(tiles[0, :2, 3:]).item(), 0)

    def test_mask_content_changes_payload_and_bbox_not_geometry(self):
        first = torch.zeros((1, 9, 15))
        second = torch.zeros_like(first)
        first[0, 1:3, 1:4] = 1
        second[0, 5:8, 11:14] = 0.5
        segs_a, _ = convert(first, tile=(8, 6), overlap=(2, 2))
        segs_b, _ = convert(second, tile=(8, 6), overlap=(2, 2))
        self.assertEqual([s.crop_region for s in segs_a[1]], [s.crop_region for s in segs_b[1]])
        self.assertEqual(len(segs_a[1]), len(segs_b[1]))
        self.assertNotEqual([s.bbox for s in segs_a[1]], [s.bbox for s in segs_b[1]])
        self.assertTrue(any(not np.array_equal(a.cropped_mask, b.cropped_mask) for a, b in zip(segs_a[1], segs_b[1])))

    def test_mask_output_preserves_dtype_and_device(self):
        mask = torch.linspace(0, 1, 70, dtype=torch.float16).reshape(1, 7, 10)
        segs, tiles = convert(mask, tile=(6, 5), overlap=(2, 2))
        self.assertEqual(tiles.dtype, mask.dtype)
        self.assertEqual(tiles.device, mask.device)
        self.assertEqual(segs[1][0].cropped_mask.dtype, np.float16)

    def test_local_seg_has_exact_impact_field_order_and_neutral_values(self):
        self.assertEqual(
            MODULE.SEG._fields,
            (
                "cropped_image",
                "cropped_mask",
                "confidence",
                "crop_region",
                "bbox",
                "label",
                "control_net_wrapper",
            ),
        )
        segs, _ = convert(torch.ones((1, 4, 4)), tile=(4, 4), overlap=(0, 0))
        entry = segs[1][0]
        self.assertEqual(entry.confidence, 1.0)
        self.assertEqual(entry.label, "")
        self.assertIsNone(entry.control_net_wrapper)

    def test_schema_is_nodes_2_segs_plus_ordinary_mask(self):
        schema = MODULE.MaskToTileSEGS.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteMaskToTileSEGS")
        self.assertEqual(schema.display_name, "MASK to Tile SEGS")
        self.assertEqual(schema.category, "Utility Suite/Mask/Tiling")
        self.assertEqual([output.io_type for output in schema.outputs], ["SEGS", "MASK"])
        self.assertTrue(all(not output.is_output_list for output in schema.outputs))
        names = [entry.id for entry in schema.inputs]
        self.assertEqual(
            names,
            [
                "mask",
                "image",
                "mode",
                "tile_width",
                "tile_height",
                "overlap_x",
                "overlap_y",
                "max_columns",
                "max_rows",
                "min_overlap_x",
                "min_overlap_y",
                "min_columns",
                "min_rows",
            ],
        )


if __name__ == "__main__":
    unittest.main()
