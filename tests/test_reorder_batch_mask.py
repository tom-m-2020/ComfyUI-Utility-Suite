from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import namedtuple
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_reorder_batch_mask_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.reorder_batch_mask"]
REORDER_SEGS = sys.modules[f"{SPEC.name}.reorder_segs"]

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)


def rectangle_mask(
    bbox: tuple[int, int, int, int],
    value: float = 1.0,
    shape: tuple[int, int] = (80, 100),
) -> torch.Tensor:
    x1, y1, x2, y2 = bbox
    result = torch.zeros(shape, dtype=torch.float32)
    result[y1:y2, x1:x2] = value
    return result


def indexed_grid(rows: int, columns: int, varying: bool = False) -> tuple[torch.Tensor, list[int]]:
    items = []
    for row in range(rows):
        for column in range(columns):
            x1 = 5 + column * 20
            y1 = 4 + row * 20
            width = 8 + ((row + column) % 3 if varying else 0)
            height = 7 + ((2 * row + column) % 3 if varying else 0)
            item = rectangle_mask((x1, y1, x1 + width, y1 + height))
            items.append(item)
    permutation = list(range(len(items) - 1, -1, -1))
    return torch.stack([items[index] for index in permutation]), permutation


def source_index(output: torch.Tensor, source: torch.Tensor) -> list[int]:
    return [next(index for index, item in enumerate(source) if torch.equal(plane, item)) for plane in output]


class ReorderBatchMaskTests(unittest.TestCase):
    def test_schema_and_single_item_identity(self):
        schema = MODULE.ReorderBatchMask.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteReorderBatchMask")
        self.assertEqual(schema.display_name, "Reorder Batch Mask")
        self.assertEqual(schema.category, "Utility Suite/Mask")
        self.assertEqual(schema.inputs[1].options, MODULE.ORDERS)
        self.assertEqual([output.io_type for output in schema.outputs], ["MASK"])

        mask = rectangle_mask((3, 4, 9, 12)).unsqueeze(0)
        for order in MODULE.ORDERS:
            self.assertTrue(torch.equal(MODULE.reorder_batch_mask(mask, order), mask))

    def test_two_by_two_all_spatial_orders(self):
        source, permutation = indexed_grid(2, 2, varying=True)
        expected_grid = {
            "left_to_right_top_to_bottom": [0, 1, 2, 3],
            "clockwise": [0, 1, 3, 2],
            "counter_clockwise": [0, 2, 3, 1],
            "clockwise_outward": [0, 1, 3, 2],
            "counter_clockwise_outward": [0, 2, 3, 1],
        }
        for order, expected_original in expected_grid.items():
            with self.subTest(order=order):
                actual = MODULE.reorder_batch_mask(source, order)
                original_for_source = {original: current for current, original in enumerate(permutation)}
                expected = [original_for_source[index] for index in expected_original]
                self.assertEqual(source_index(actual, source), expected)

    def test_larger_rectangular_grid_matches_shared_ring_traversal(self):
        source, permutation = indexed_grid(3, 4, varying=True)
        original_for_source = {original: current for current, original in enumerate(permutation)}
        for order in REORDER_SEGS.SPATIAL_ORDERS:
            with self.subTest(order=order):
                expected = [
                    original_for_source[row * 4 + column]
                    for row, column in REORDER_SEGS.grid_coordinates(3, 4, order)
                ]
                actual = MODULE.reorder_batch_mask(source, order)
                self.assertEqual(source_index(actual, source), expected)

    def test_spatial_order_agrees_with_reorder_segs_for_equivalent_geometry(self):
        boxes = [
            (5 + column * 20, 4 + row * 20, 13 + column * 20, 11 + row * 20)
            for row in range(3)
            for column in range(4)
        ]
        permutation = [7, 0, 11, 2, 9, 4, 1, 10, 5, 3, 8, 6]
        mask = torch.stack([rectangle_mask(boxes[index]) for index in permutation])
        entries = [SEG(None, None, 1.0, boxes[index], boxes[index], str(index), None) for index in permutation]
        for order in REORDER_SEGS.SPATIAL_ORDERS:
            with self.subTest(order=order):
                expected = [int(entry.label) for entry in REORDER_SEGS.reorder_segs(((80, 100), entries), order)[1]]
                actual = source_index(MODULE.reorder_batch_mask(mask, order), mask)
                self.assertEqual([permutation[index] for index in actual], expected)

    def test_differently_sized_objects_share_position_rows_and_columns(self):
        boxes = [(3, 4, 10, 11), (24, 2, 36, 13), (1, 25, 12, 34), (25, 23, 34, 36)]
        mask = torch.stack([rectangle_mask(boxes[index]) for index in (3, 1, 2, 0)])
        actual = MODULE.reorder_batch_mask(mask, "left_to_right_top_to_bottom")
        self.assertEqual(source_index(actual, mask), [3, 1, 2, 0])

    def test_biggest_smallest_use_bbox_area_and_keep_equal_area_stable(self):
        boxes = [(2, 2, 7, 12), (15, 3, 25, 13), (30, 4, 35, 14), (45, 2, 53, 10)]
        mask = torch.stack([rectangle_mask(box) for box in boxes])
        biggest = MODULE.reorder_batch_mask(mask, "biggest")
        smallest = MODULE.reorder_batch_mask(mask, "smallest")
        self.assertEqual(source_index(biggest, mask), [1, 3, 0, 2])
        self.assertEqual(source_index(smallest, mask), [0, 2, 3, 1])

    def test_thin_large_span_beats_dense_small_block(self):
        outline = torch.zeros((240, 240))
        outline[10, 10:210] = 1
        outline[209, 10:210] = 1
        outline[10:210, 10] = 1
        outline[10:210, 209] = 1
        block = torch.zeros_like(outline)
        block[100:200, 100:200] = 1
        self.assertLess(torch.count_nonzero(outline), torch.count_nonzero(block))
        source = torch.stack((block, outline))
        self.assertEqual(source_index(MODULE.reorder_batch_mask(source, "biggest"), source), [1, 0])

    def test_empty_masks_are_area_zero_and_spatially_appended_stably(self):
        empty_a = torch.zeros((30, 40))
        nonempty = rectangle_mask((3, 4, 8, 10), shape=(30, 40))
        empty_b = torch.zeros((30, 40))
        source = torch.stack((empty_a, nonempty, empty_b))
        smallest = MODULE.reorder_batch_mask(source, "smallest")
        biggest = MODULE.reorder_batch_mask(source, "biggest")
        self.assertEqual([torch.count_nonzero(item).item() for item in smallest], [0, 0, 30])
        self.assertEqual([torch.count_nonzero(item).item() for item in biggest], [30, 0, 0])
        for order in REORDER_SEGS.SPATIAL_ORDERS:
            actual = MODULE.reorder_batch_mask(source, order)
            self.assertEqual([torch.count_nonzero(item).item() for item in actual], [30, 0, 0])

    def test_quantized_bounds_match_tile_segs_tail_semantics(self):
        invisible = torch.zeros((20, 30))
        invisible[1, 1] = 0.001
        invisible[7:10, 12:16] = 0.5
        visible_tail = invisible.clone()
        visible_tail[1, 1] = 1 / 255
        self.assertEqual(MODULE.quantized_mask_bounds(invisible), (12, 7, 16, 10))
        self.assertEqual(MODULE.quantized_mask_bounds(visible_tail), (1, 1, 16, 10))

    def test_soft_values_dtype_device_pixels_and_cardinality_are_preserved(self):
        first = rectangle_mask((2, 3, 8, 10), 0.37).to(torch.float64)
        second = rectangle_mask((20, 15, 31, 27), 0.83).to(torch.float64)
        source = torch.stack((second, first))
        snapshot = source.clone()
        actual = MODULE.reorder_batch_mask(source, "smallest")
        self.assertEqual(actual.dtype, source.dtype)
        self.assertEqual(actual.device, source.device)
        self.assertEqual(tuple(actual.shape), tuple(source.shape))
        self.assertEqual(source_index(actual, source), [1, 0])
        self.assertTrue(torch.equal(source, snapshot))

    def test_ambiguous_same_position_fails_instead_of_inventing_order(self):
        source = torch.stack((rectangle_mask((4, 4, 12, 12)), rectangle_mask((5, 5, 11, 11))))
        with self.assertRaisesRegex(ValueError, "unique position lattice"):
            MODULE.reorder_batch_mask(source, "clockwise")

    def test_validation(self):
        with self.assertRaises(TypeError):
            MODULE.reorder_batch_mask([], "biggest")
        with self.assertRaises(ValueError):
            MODULE.reorder_batch_mask(torch.zeros((4, 5)), "biggest")
        with self.assertRaises(ValueError):
            MODULE.reorder_batch_mask(torch.zeros((0, 4, 5)), "biggest")
        with self.assertRaisesRegex(ValueError, "expected one of"):
            MODULE.reorder_batch_mask(torch.zeros((1, 4, 5)), "other")


if __name__ == "__main__":
    unittest.main()
