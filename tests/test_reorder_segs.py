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
    "utility_suite_reorder_segs_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.reorder_segs"]
FILTER = sys.modules[f"{SPEC.name}.filter_tile_segs"]
MASK_TILE = sys.modules[f"{SPEC.name}.mask_tile_segs"]
RESTORE = sys.modules[f"{SPEC.name}.restore_tile_segs_gaps"]

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)

SPIRAL_ORDERS = ["clockwise", "counter_clockwise", "clockwise_outward", "counter_clockwise_outward"]
SHAPES = [(1, 1), (1, 5), (5, 1), (2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4), (4, 5), (5, 4), (5, 5)]


def segment(row, column, label, row_span=None, column_span=None):
    y1, y2 = row_span if row_span is not None else (row * 7, row * 7 + 10)
    x1, x2 = column_span if column_span is not None else (column * 9, column * 9 + 12)
    mask = np.full((y2 - y1, x2 - x1), float(label), dtype=np.float32)
    return SEG(None, mask, 0.75, (x1, y1, x2, y2), (x1, y1, x2, y2), str(label), object())


def grid(rows, columns):
    return [segment(row, column, row * columns + column + 1) for row in range(rows) for column in range(columns)]


class GridTraversalTests(unittest.TestCase):
    def test_all_required_shapes_are_complete_and_unique(self):
        for rows, columns in SHAPES:
            expected = {(row, column) for row in range(rows) for column in range(columns)}
            for order in SPIRAL_ORDERS:
                with self.subTest(rows=rows, columns=columns, order=order):
                    result = MODULE.grid_coordinates(rows, columns, order)
                    self.assertEqual(len(result), rows * columns)
                    self.assertEqual(len(set(result)), rows * columns)
                    self.assertEqual(set(result), expected)

    def test_three_by_three_inward_directions(self):
        self.assertEqual(
            MODULE.grid_coordinates(3, 3, "clockwise"),
            [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (1, 1)],
        )
        self.assertEqual(
            MODULE.grid_coordinates(3, 3, "counter_clockwise"),
            [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1), (1, 1)],
        )

    def test_three_by_three_outward_directions(self):
        self.assertEqual(
            MODULE.grid_coordinates(3, 3, "clockwise_outward"),
            [(1, 1), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)],
        )
        self.assertEqual(
            MODULE.grid_coordinates(3, 3, "counter_clockwise_outward"),
            [(1, 1), (0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1)],
        )

    def test_even_center_block_direction(self):
        clockwise = MODULE.grid_coordinates(4, 4, "clockwise_outward")
        counter = MODULE.grid_coordinates(4, 4, "counter_clockwise_outward")
        self.assertEqual(clockwise[:4], [(1, 1), (1, 2), (2, 2), (2, 1)])
        self.assertEqual(counter[:4], [(1, 1), (2, 1), (2, 2), (1, 2)])
        self.assertEqual(clockwise[4], (0, 0))
        self.assertEqual(counter[4], (0, 0))

    def test_rectangular_center_blocks_follow_parity_rule(self):
        self.assertEqual(MODULE.grid_coordinates(4, 5, "clockwise_outward")[:2], [(1, 2), (2, 2)])
        self.assertEqual(MODULE.grid_coordinates(5, 4, "clockwise_outward")[:2], [(2, 1), (2, 2)])
        self.assertEqual(MODULE.grid_coordinates(3, 7, "clockwise_outward")[0], (1, 3))

    def test_degenerate_rows_and_columns(self):
        row = [(0, column) for column in range(5)]
        column = [(row, 0) for row in range(5)]
        for order in SPIRAL_ORDERS:
            with self.subTest(order=order):
                if "outward" not in order:
                    self.assertEqual(MODULE.grid_coordinates(1, 5, order), row)
                    self.assertEqual(MODULE.grid_coordinates(5, 1, order), column)
        self.assertEqual(MODULE.grid_coordinates(1, 5, "clockwise_outward")[:2], [(0, 2), (0, 1)])
        self.assertEqual(MODULE.grid_coordinates(5, 1, "clockwise_outward")[:2], [(2, 0), (1, 0)])


class ReorderSEGSTests(unittest.TestCase):
    def test_biggest_and_smallest_use_bbox_area_with_stable_ties(self):
        entries = [
            segment(0, 0, 1)._replace(bbox=(0, 0, 10, 10)),
            segment(0, 1, 2)._replace(bbox=(0, 0, 5, 10)),
            segment(0, 2, 3)._replace(bbox=(0, 0, 20, 5)),
            segment(0, 3, 4)._replace(bbox=(0.5, 0.5, 5.5, 10.5)),
        ]
        source_shape = (24, 50)
        biggest = MODULE.reorder_segs((source_shape, entries), "biggest")
        smallest = MODULE.reorder_segs((source_shape, entries), "smallest")
        self.assertEqual([entry.label for entry in biggest[1]], ["1", "3", "2", "4"])
        self.assertEqual([entry.label for entry in smallest[1]], ["2", "4", "1", "3"])

    def test_size_order_ignores_crop_size_and_mask_occupancy(self):
        thin = np.zeros((10, 10), dtype=np.float32)
        thin[0, :] = 1
        dense = np.ones((20, 20), dtype=np.float32)
        large_bbox = segment(0, 0, 1)._replace(
            cropped_mask=thin,
            crop_region=(0, 0, 10, 10),
            bbox=(0, 0, 40, 20),
        )
        large_crop = segment(0, 1, 2)._replace(
            cropped_mask=dense,
            crop_region=(20, 0, 80, 40),
            bbox=(20, 0, 30, 10),
        )
        result = MODULE.reorder_segs(((50, 100), [large_crop, large_bbox]), "biggest")
        self.assertEqual([entry.label for entry in result[1]], ["1", "2"])

    def test_size_order_preserves_header_entry_identity_and_cardinality(self):
        header = (24, 50)
        entries = [
            segment(0, 0, 1)._replace(bbox=(0, 0, 3, 4)),
            segment(0, 1, 2)._replace(bbox=(0, 0, 8, 9)),
            segment(0, 2, 3)._replace(bbox=(0, 0, 5, 5)),
        ]
        result = MODULE.reorder_segs((header, entries), "biggest")
        self.assertIs(result[0], header)
        self.assertEqual(len(result[1]), len(entries))
        self.assertEqual([id(entry) for entry in result[1]], [id(entries[1]), id(entries[2]), id(entries[0])])

    def test_empty_mask_segment_uses_existing_bbox_instead_of_zero_area(self):
        empty = segment(0, 0, 1)._replace(
            cropped_mask=np.zeros((10, 12), dtype=np.float32),
            bbox=(0, 0, 12, 10),
        )
        nonempty = segment(0, 1, 2)._replace(
            cropped_mask=np.ones((10, 12), dtype=np.float32),
            bbox=(0, 0, 3, 4),
        )
        result = MODULE.reorder_segs(((24, 30), [nonempty, empty]), "biggest")
        self.assertEqual([entry.label for entry in result[1]], ["1", "2"])

    def test_size_order_validates_bbox_without_crop_fallback(self):
        entry = segment(0, 0, 1)
        for bbox in (None, (0, 0, 1), (0, 0, "x", 1), (0, 0, -1, 2), (0, 0, float("inf"), 2)):
            with self.subTest(bbox=bbox), self.assertRaises((TypeError, ValueError)):
                MODULE.reorder_segs(((20, 20), [entry._replace(bbox=bbox)]), "biggest")

    def test_all_orders_preserve_shape_identity_fields_and_cardinality(self):
        entries = grid(3, 4)
        snapshots = [tuple(entry) for entry in entries]
        shuffled = entries[5:] + entries[:5]
        source_shape = (24, 39)
        for order in MODULE.ORDERS:
            with self.subTest(order=order):
                result = MODULE.reorder_segs((source_shape, shuffled), order)
                self.assertIs(result[0], source_shape)
                self.assertEqual(len(result[1]), len(entries))
                self.assertEqual({id(entry) for entry in result[1]}, {id(entry) for entry in entries})
                self.assertEqual([tuple(entry) for entry in entries], snapshots)

    def test_exact_real_segs_label_orders(self):
        entries = grid(3, 3)
        expected = {
            "left_to_right_top_to_bottom": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "clockwise": [1, 2, 3, 6, 9, 8, 7, 4, 5],
            "counter_clockwise": [1, 4, 7, 8, 9, 6, 3, 2, 5],
            "clockwise_outward": [5, 1, 2, 3, 6, 9, 8, 7, 4],
            "counter_clockwise_outward": [5, 1, 4, 7, 8, 9, 6, 3, 2],
        }
        for order, labels in expected.items():
            with self.subTest(order=order):
                result = MODULE.reorder_segs(((24, 30), entries[::-1]), order)
                self.assertEqual([int(entry.label) for entry in result[1]], labels)

    def test_sparse_restored_topology_uses_compressed_columns(self):
        labels = [[1, 2, 5, 6], [7, 8, 11, 12], [13, 14, 17, 18]]
        entries = [segment(row, column, label) for row, values in enumerate(labels) for column, label in enumerate(values)]
        for order in MODULE.ORDERS:
            with self.subTest(order=order):
                result = MODULE.reorder_segs(((24, 39), entries[::-1]), order)
                self.assertEqual(len(result[1]), 12)
                self.assertEqual({id(entry) for entry in result[1]}, {id(entry) for entry in entries})
        row_major = MODULE.reorder_segs(((24, 39), entries[::-1]), "left_to_right_top_to_bottom")
        self.assertEqual([int(entry.label) for entry in row_major[1]], [1, 2, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18])

    def test_sparse_cells_are_skipped_without_recalculating_path(self):
        entries = grid(3, 3)
        sparse = [entry for index, entry in enumerate(entries) if index not in (1, 4)]
        result = MODULE.reorder_segs(((24, 30), sparse), "clockwise")
        self.assertEqual([int(entry.label) for entry in result[1]], [1, 3, 6, 9, 8, 7, 4])

    def test_empty_input(self):
        source_shape = (10, 20)
        for order in MODULE.ORDERS:
            result = MODULE.reorder_segs((source_shape, []), order)
            self.assertIs(result[0], source_shape)
            self.assertEqual(result[1], [])

    def test_nonuniform_grid_spans(self):
        row_spans = [(0, 8), (5, 13), (11, 19)]
        column_spans = [(0, 9), (4, 13), (10, 19), (15, 24)]
        entries = [
            segment(row, column, row * 4 + column + 1, row_spans[row], column_spans[column])
            for row in range(3)
            for column in range(4)
        ]
        result = MODULE.reorder_segs(((19, 24), entries[::-1]), "left_to_right_top_to_bottom")
        self.assertEqual([int(entry.label) for entry in result[1]], list(range(1, 13)))

    def test_validation_rejects_non_tile_ambiguity_and_duplicates(self):
        first = segment(0, 0, 1)
        duplicate = first._replace(label="duplicate")
        with self.assertRaisesRegex(ValueError, "duplicate crop_region"):
            MODULE.reorder_segs(((20, 20), [first, duplicate]), "clockwise")
        ambiguous = segment(0, 1, 2)._replace(crop_region=(9, 0, 20, 11))
        with self.assertRaisesRegex(ValueError, "ambiguous row spans"):
            MODULE.reorder_segs(((20, 20), [first, ambiguous]), "clockwise")
        with self.assertRaisesRegex(ValueError, "expected one of"):
            MODULE.reorder_segs(((20, 20), []), "other")

    def test_mask_to_tile_segs_interoperability(self):
        mask = torch.ones((1, 18, 24), dtype=torch.float32)
        segs, _ = MASK_TILE.mask_to_tile_segs(mask, "uniform_grid", 10, 10, 0, 0, 3, 3, 2, 2, 3, 2)
        result = MODULE.reorder_segs(segs, "clockwise")
        self.assertEqual(len(result[1]), len(segs[1]))
        self.assertEqual({id(entry) for entry in result[1]}, {id(entry) for entry in segs[1]})

    def test_mask_to_tile_segs_size_order_uses_generated_content_bbox(self):
        mask = torch.zeros((1, 10, 20), dtype=torch.float32)
        mask[:, 1:9, 1:8] = 1
        mask[:, 3:6, 13:16] = 1
        segs, _ = MASK_TILE.mask_to_tile_segs(mask, "uniform_grid", 10, 10, 0, 0, 2, 1, 0, 0, 2, 1)
        self.assertEqual([tuple(entry.crop_region) for entry in segs[1]], [(0, 0, 10, 10), (10, 0, 20, 10)])
        result = MODULE.reorder_segs((segs[0], list(reversed(segs[1]))), "biggest")
        self.assertIs(result[1][0], segs[1][0])
        self.assertIs(result[1][1], segs[1][1])

    def test_filter_restore_reorder_interoperability(self):
        mask = torch.zeros((1, 12, 30), dtype=torch.float32)
        mask[:, :, :8] = 1
        mask[:, :, 22:] = 1
        original, _ = MASK_TILE.mask_to_tile_segs(mask, "uniform_grid", 14, 12, 0, 0, 3, 1, 2, 0, 3, 1)
        kept, excluded = FILTER.partition_tile_segs(original, "mask")
        restored = RESTORE.restore_tile_segs_gaps(kept, excluded, "mask", 4, 0)
        result = MODULE.reorder_segs(restored, "counter_clockwise_outward")
        self.assertEqual(result[0], original[0])
        self.assertEqual({id(entry) for entry in result[1]}, {id(entry) for entry in restored[1]})

    def test_schema(self):
        schema = MODULE.ReorderSEGS.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteReorderSEGS")
        self.assertEqual(schema.display_name, "Reorder SEGS")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual([entry.io_type for entry in schema.inputs], ["SEGS", "COMBO"])
        self.assertEqual(schema.inputs[1].options, MODULE.ORDERS)
        self.assertEqual(MODULE.ORDERS[-2:], ["biggest", "smallest"])
        self.assertEqual([output.io_type for output in schema.outputs], ["SEGS"])
        self.assertFalse(schema.outputs[0].is_output_list)


if __name__ == "__main__":
    unittest.main()
