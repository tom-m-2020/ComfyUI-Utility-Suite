from __future__ import annotations

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
    "utility_suite_filter_tile_segs_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.filter_tile_segs"]
MASK_TILE = sys.modules[f"{SPEC.name}.mask_tile_segs"]

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)


def segment(
    crop: tuple[int, int, int, int],
    mask: np.ndarray | torch.Tensor | None = None,
    bbox: tuple[int, int, int, int] | None = None,
    label: str = "",
) -> SEG:
    x1, y1, x2, y2 = crop
    if mask is None:
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
    if bbox is None:
        bbox = crop
    return SEG(None, mask, 1.0, crop, bbox, label, None)


def partition(shape, entries, mode="mask"):
    return MODULE.partition_tile_segs((shape, entries), mode)


class FilterTileSEGSTests(unittest.TestCase):
    def test_one_segment_with_mask_is_kept(self):
        item = segment((0, 0, 5, 4), np.ones((4, 5), dtype=np.float32), label="one")
        kept, excluded = partition((4, 5), [item])
        self.assertEqual(kept[1], [item])
        self.assertEqual(excluded[1], [])

    def test_one_segment_with_empty_mask_is_excluded_in_both_modes(self):
        item = segment((0, 0, 5, 4), label="empty")
        for mode in ("mask", "bbox"):
            with self.subTest(mode=mode):
                kept, excluded = partition((4, 5), [item], mode)
                self.assertEqual(kept[1], [])
                self.assertEqual(excluded[1], [item])

    def test_three_tile_acceptance_excludes_nonempty_center(self):
        left_mask = np.zeros((4, 5), dtype=np.float32)
        left_mask[:, 1] = 1
        center_mask = np.ones((4, 4), dtype=np.float32)
        right_mask = np.zeros((4, 5), dtype=np.float32)
        right_mask[:, 3] = 1
        left = segment((0, 0, 5, 4), left_mask, label="left")
        center = segment((3, 0, 7, 4), center_mask, label="center")
        right = segment((5, 0, 10, 4), right_mask, label="right")
        self.assertTrue(center.cropped_mask.any())

        kept, excluded = partition((4, 10), [left, center, right])

        self.assertEqual([item.label for item in kept[1]], ["left", "right"])
        self.assertEqual([item.label for item in excluded[1]], ["center"])

    def test_mask_only_in_horizontal_overlap_is_excluded(self):
        mask = np.zeros((4, 6), dtype=np.float32)
        mask[:, 4:6] = 1
        first = segment((0, 0, 6, 4), mask, label="overlap-only")
        second = segment((4, 0, 10, 4), label="other")
        kept, excluded = partition((4, 10), [first, second])
        self.assertEqual(kept[1], [])
        self.assertEqual(excluded[1], [first, second])

    def test_any_positive_stride_value_is_kept_without_tolerance(self):
        mask = np.zeros((4, 6), dtype=np.float32)
        mask[1, 1] = 1e-12
        first = segment((0, 0, 6, 4), mask)
        second = segment((4, 0, 10, 4))
        kept, _ = partition((4, 10), [first, second])
        self.assertEqual(kept[1], [first])

    def test_grayscale_mask_is_inspected_without_modification(self):
        values = np.linspace(0, 0.8, 20, dtype=np.float32).reshape(4, 5)
        original = values.copy()
        item = segment((0, 0, 5, 4), values)
        partition((4, 5), [item])
        np.testing.assert_array_equal(item.cropped_mask, original)

    def test_bbox_intersects_stride_or_only_overlap(self):
        first = segment((0, 0, 6, 4), np.ones((4, 6), dtype=np.float32), bbox=(1, 1, 3, 3), label="stride")
        second = segment(
            (4, 0, 10, 4), np.ones((4, 6), dtype=np.float32), bbox=(4, 0, 6, 4), label="overlap"
        )
        kept, excluded = partition((4, 10), [first, second], "bbox")
        self.assertEqual(kept[1], [first])
        self.assertEqual(excluded[1], [second])

    def test_mask_and_bbox_modes_can_differ(self):
        mask = np.zeros((4, 6), dtype=np.float32)
        mask[:, 4:6] = 1
        first = segment((0, 0, 6, 4), mask, bbox=(0, 0, 6, 4), label="different")
        second = segment((4, 0, 10, 4), label="other")
        mask_kept, _ = partition((4, 10), [first, second], "mask")
        bbox_kept, _ = partition((4, 10), [first, second], "bbox")
        self.assertNotIn(first, mask_kept[1])
        self.assertIn(first, bbox_kept[1])

    def test_fully_overlapped_entries_have_no_stride(self):
        first = segment((0, 0, 5, 4), np.ones((4, 5), dtype=np.float32), label="a")
        second = segment((0, 0, 5, 4), np.ones((4, 5), dtype=np.float32), label="b")
        for mode in ("mask", "bbox"):
            with self.subTest(mode=mode):
                kept, excluded = partition((4, 5), [first, second], mode)
                self.assertEqual(kept[1], [])
                self.assertEqual(excluded[1], [first, second])

    def test_vertical_overlap(self):
        top_mask = np.zeros((5, 4), dtype=np.float32)
        top_mask[1, 1] = 1
        bottom_mask = np.zeros((5, 4), dtype=np.float32)
        bottom_mask[3, 2] = 1
        top = segment((0, 0, 4, 5), top_mask)
        bottom = segment((0, 3, 4, 8), bottom_mask)
        kept, _ = partition((8, 4), [top, bottom])
        self.assertEqual(kept[1], [top, bottom])

    def test_two_dimensional_four_way_intersection(self):
        crops = [(0, 0, 4, 4), (2, 0, 6, 4), (0, 2, 4, 6), (2, 2, 6, 6)]
        masks = []
        points = [(0, 0), (3, 0), (0, 3), (3, 3)]
        for point in points:
            mask = np.zeros((4, 4), dtype=np.float32)
            mask[point[1], point[0]] = 1
            masks.append(mask)
        entries = [segment(crop, mask, label=str(index)) for index, (crop, mask) in enumerate(zip(crops, masks))]
        kept, excluded = partition((6, 6), entries)
        self.assertEqual(kept[1], entries)
        self.assertEqual(excluded[1], [])

    def test_touching_crops_do_not_overlap(self):
        left = segment((0, 0, 4, 3), np.ones((3, 4), dtype=np.float32))
        right = segment((4, 0, 8, 3), np.ones((3, 4), dtype=np.float32))
        kept, _ = partition((3, 8), [left, right])
        self.assertEqual(kept[1], [left, right])

    def test_empty_segs_preserves_outer_shape(self):
        source_shape = (9, 13)
        kept, excluded = partition(source_shape, [])
        self.assertIs(kept[0], source_shape)
        self.assertIs(excluded[0], source_shape)
        self.assertEqual(kept[1], [])
        self.assertEqual(excluded[1], [])

    def test_order_identity_fields_and_header_are_preserved(self):
        source_shape = (4, 10)
        entries = [
            segment((0, 0, 6, 4), np.ones((4, 6), dtype=np.float32), label="keep-0"),
            segment((4, 0, 10, 4), label="exclude-1"),
            segment((0, 0, 3, 4), np.ones((4, 3), dtype=np.float32), label="exclude-2"),
        ]
        snapshots = [tuple(item) for item in entries]
        kept, excluded = partition(source_shape, entries)
        self.assertIs(kept[0], source_shape)
        self.assertIs(excluded[0], source_shape)
        self.assertTrue(all(any(output is original for original in entries) for output in kept[1] + excluded[1]))
        self.assertEqual([tuple(item) for item in entries], snapshots)
        positions = {id(item): index for index, item in enumerate(entries)}
        self.assertEqual(
            [positions[id(item)] for item in kept[1]],
            sorted(positions[id(item)] for item in kept[1]),
        )
        self.assertEqual(
            [positions[id(item)] for item in excluded[1]],
            sorted(positions[id(item)] for item in excluded[1]),
        )

    def test_singleton_rank_three_mask_is_supported(self):
        mask = torch.ones((1, 4, 5))
        item = segment((0, 0, 5, 4), mask)
        kept, _ = partition((4, 5), [item])
        self.assertEqual(kept[1], [item])

    def test_bbox_is_clipped_to_crop_and_canvas(self):
        item = segment((0, 0, 5, 4), np.ones((4, 5), dtype=np.float32), bbox=(-10, -10, 2, 2))
        kept, _ = partition((4, 5), [item], "bbox")
        self.assertEqual(kept[1], [item])

    def test_mask_to_tile_segs_uniform_grid_interoperability(self):
        source = torch.zeros((1, 4, 16), dtype=torch.float32)
        source[:, :, :4] = 1
        source[:, :, 12:] = 1
        segs, _ = MASK_TILE.mask_to_tile_segs(
            source,
            "uniform_grid",
            8,
            4,
            0,
            0,
            3,
            3,
            0,
            0,
            3,
            1,
        )
        self.assertEqual(segs[1][1].bbox, tuple(segs[1][1].crop_region))
        self.assertEqual(np.count_nonzero(segs[1][1].cropped_mask), 0)
        kept, excluded = MODULE.partition_tile_segs(segs, "bbox")
        self.assertEqual([entry.crop_region for entry in kept[1]], [[0, 0, 8, 4], [8, 0, 16, 4]])
        self.assertEqual([entry.crop_region for entry in excluded[1]], [[4, 0, 12, 4]])
        self.assertIs(kept[1][0], segs[1][0])

    def test_invalid_structures_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "expected 'mask' or 'bbox'"):
            partition((4, 5), [], "other")
        with self.assertRaisesRegex(TypeError, "shaped as"):
            MODULE.partition_tile_segs([], "mask")
        with self.assertRaisesRegex(ValueError, "positive source"):
            partition((0, 5), [])
        with self.assertRaisesRegex(TypeError, "no crop_region"):
            partition((4, 5), [object()])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            partition((4, 5), [segment((1, 1, 1, 3))])
        with self.assertRaisesRegex(ValueError, "outside source"):
            partition((4, 5), [segment((0, 0, 6, 4))])

    def test_invalid_mask_and_bbox_fail_clearly(self):
        crop = (0, 0, 5, 4)
        with self.assertRaisesRegex(TypeError, "no cropped_mask"):
            partition((4, 5), [segment(crop)._replace(cropped_mask=None)])
        with self.assertRaisesRegex(ValueError, "one leading item"):
            partition((4, 5), [segment(crop, np.ones((2, 4, 5), dtype=np.float32))])
        with self.assertRaisesRegex(ValueError, "does not match"):
            partition((4, 5), [segment(crop, np.ones((3, 5), dtype=np.float32))])
        with self.assertRaisesRegex(TypeError, "no bbox"):
            partition(
                (4, 5),
                [segment(crop, np.ones((4, 5), dtype=np.float32))._replace(bbox=None)],
                "bbox",
            )
        with self.assertRaisesRegex(ValueError, "reversed"):
            partition(
                (4, 5),
                [segment(crop, np.ones((4, 5), dtype=np.float32), bbox=(4, 0, 2, 3))],
                "bbox",
            )

    def test_schema_declares_two_ordinary_segs_outputs(self):
        schema = MODULE.FilterTileSEGS.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteFilterTileSEGS")
        self.assertEqual(schema.display_name, "Filter Tile SEGS")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual([entry.io_type for entry in schema.inputs], ["SEGS", "COMBO"])
        self.assertEqual([output.io_type for output in schema.outputs], ["SEGS", "SEGS"])
        self.assertTrue(all(not output.is_output_list for output in schema.outputs))


if __name__ == "__main__":
    unittest.main()
