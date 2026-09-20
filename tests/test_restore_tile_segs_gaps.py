from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_restore_tile_segs_gaps_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.restore_tile_segs_gaps"]
FILTER = sys.modules[f"{SPEC.name}.filter_tile_segs"]
MASK_TILE = sys.modules[f"{SPEC.name}.mask_tile_segs"]

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)


def segment(crop, label, active=True, bbox=None):
    x1, y1, x2, y2 = crop
    mask = np.ones((y2 - y1, x2 - x1), dtype=np.float32) if active else np.zeros(
        (y2 - y1, x2 - x1), dtype=np.float32
    )
    return SEG(None, mask, 1.0, crop, crop if bbox is None else bbox, str(label), None)


def source_segments(source_mask, crops):
    return [
        SEG(
            None,
            source_mask[y1:y2, x1:x2].copy(),
            1.0,
            crop,
            crop,
            str(index),
            None,
        )
        for index, crop in enumerate(crops, 1)
        for x1, y1, x2, y2 in [crop]
    ]


def restore(shape, kept, excluded, mode="mask", min_x=1, min_y=1):
    return MODULE.restore_tile_segs_gaps((shape, kept), (shape, excluded), mode, min_x, min_y)


def acceptance_grid():
    crops = [(x, y, x + 15, y + 60) for y in (0, 40, 80) for x in (0, 10, 20, 30, 40, 50)]
    source_mask = np.zeros((140, 65), dtype=np.float32)
    source_mask[:, :18] = 1
    source_mask[:, 40:] = 1
    entries = source_segments(source_mask, crops)
    kept_labels = {1, 2, 5, 6, 13, 14, 17, 18}
    return entries, [entry for entry in entries if int(entry.label) in kept_labels], [
        entry for entry in entries if int(entry.label) not in kept_labels
    ]


class RestoreTileSEGSGapsTests(unittest.TestCase):
    def test_mandatory_eighteen_tile_acceptance_trace(self):
        entries, kept, excluded = acceptance_grid()
        seen_candidates = []

        def classify(candidate_segs, mode):
            seen_candidates.extend(entry.label for entry in candidate_segs[1])
            return FILTER.partition_tile_segs(candidate_segs, mode)

        with mock.patch.object(MODULE, "partition_tile_segs", side_effect=classify):
            result = restore((140, 65), kept, excluded, min_x=0, min_y=20)

        self.assertEqual([entry.label for entry in kept], ["1", "2", "5", "6", "13", "14", "17", "18"])
        self.assertEqual(seen_candidates, ["7", "8", "9", "10", "11", "12"])
        self.assertEqual(
            [entry.label for entry in result[1]],
            ["1", "2", "5", "6", "7", "8", "11", "12", "13", "14", "17", "18"],
        )
        self.assertTrue(all(any(item is original for original in entries) for item in result[1]))

    def test_vertical_missing_row_expands_complete_row(self):
        _, kept, excluded = acceptance_grid()
        result = restore((140, 65), kept, excluded, min_x=0, min_y=20)
        self.assertEqual([entry.label for entry in result[1] if 7 <= int(entry.label) <= 12], ["7", "8", "11", "12"])

    def test_horizontal_missing_column_expands_complete_column(self):
        crops = [(x, y, x + 60, y + 15) for y in (0, 10, 20) for x in (0, 40, 80)]
        entries = source_segments(np.ones((35, 140), dtype=np.float32), crops)
        kept = [entry for entry in entries if int(entry.label) in (1, 3, 7, 9)]
        kept_ids = {id(entry) for entry in kept}
        excluded = [entry for entry in entries if id(entry) not in kept_ids]
        result = restore((35, 140), kept, excluded, min_x=20, min_y=0)
        self.assertEqual([entry.label for entry in result[1]], ["1", "2", "3", "5", "7", "8", "9"])

    def test_positive_insufficient_overlap_restores(self):
        a = segment((0, 0, 20, 10), "a")
        b = segment((6, 0, 26, 10), "b")
        c = segment((12, 0, 32, 10), "c")
        result = restore((10, 32), [a, c], [b], min_x=9, min_y=0)
        self.assertEqual(result[1], [a, b, c])

    def test_boundary_contact_requires_positive_minimum(self):
        a = segment((0, 0, 10, 10), "a")
        b = segment((5, 0, 15, 10), "b")
        c = segment((10, 0, 20, 10), "c")
        self.assertEqual(restore((10, 20), [a, c], [b], min_x=1, min_y=0)[1], [a, b, c])
        self.assertEqual(restore((10, 20), [a, c], [b], min_x=0, min_y=0)[1], [a, c])

    def test_actual_gap_with_multiple_intermediates_restores(self):
        items = [segment((x, 0, x + 10, 10), str(index)) for index, x in enumerate((0, 6, 12, 18), 1)]
        result = restore((10, 28), [items[0], items[3]], items[1:3], min_x=1, min_y=0)
        self.assertEqual(result[1], items)

    def test_overlap_equal_or_greater_than_minimum_needs_no_repair(self):
        a = segment((0, 0, 20, 10), "a")
        b = segment((5, 0, 25, 10), "b")
        c = segment((10, 0, 30, 10), "c")
        self.assertEqual(restore((10, 30), [a, c], [b], min_x=10, min_y=0)[1], [a, c])
        self.assertEqual(restore((10, 30), [a, c], [b], min_x=8, min_y=0)[1], [a, c])

    def test_axis_minimums_are_independent(self):
        left = segment((0, 0, 10, 10), "left")
        middle = segment((5, 0, 15, 10), "middle")
        right = segment((10, 0, 20, 10), "right")
        self.assertEqual(restore((10, 20), [left, right], [middle], min_x=0, min_y=999)[1], [left, right])

    def test_candidate_discovered_repeatedly_is_deduplicated(self):
        center = segment((5, 5, 15, 15), "center")
        kept = [
            segment((0, 5, 10, 15), "left"),
            segment((10, 5, 20, 15), "right"),
            segment((5, 0, 15, 10), "top"),
            segment((5, 10, 15, 20), "bottom"),
        ]
        result = restore((20, 20), kept, [center], min_x=1, min_y=1)
        self.assertEqual(sum(entry is center for entry in result[1]), 1)

    def test_diagonal_rectangles_do_not_form_sequence(self):
        first = segment((0, 0, 10, 10), "first")
        diagonal = segment((5, 5, 15, 15), "diagonal")
        last = segment((10, 10, 20, 20), "last")
        self.assertEqual(restore((20, 20), [first, last], [diagonal], min_x=5, min_y=5)[1], [first, last])

    def test_mask_and_bbox_candidate_filtering_share_existing_semantics(self):
        source_mask = np.zeros((10, 20), dtype=np.float32)
        source_mask[:, 9:15] = 1
        a, b, c = source_segments(source_mask, [(0, 0, 10, 10), (5, 0, 15, 10), (10, 0, 20, 10)])
        mask_result = restore((10, 20), [a, c], [b], "mask", 1, 0)
        bbox_result = restore((10, 20), [a, c], [b], "bbox", 1, 0)
        self.assertTrue(any(entry is b for entry in mask_result[1]))
        self.assertTrue(any(entry is b for entry in bbox_result[1]))

    def test_candidates_can_all_fail_or_all_survive(self):
        crops = [(0, 0, 10, 10), (6, 0, 16, 10), (16, 0, 26, 10)]
        failed_entries = source_segments(np.zeros((10, 26), dtype=np.float32), crops)
        passed_entries = source_segments(np.ones((10, 26), dtype=np.float32), crops)
        self.assertEqual(
            restore((10, 26), [failed_entries[0], failed_entries[2]], [failed_entries[1]], min_x=1, min_y=0)[1],
            [failed_entries[0], failed_entries[2]],
        )
        self.assertEqual(
            restore((10, 26), [passed_entries[0], passed_entries[2]], [passed_entries[1]], min_x=1, min_y=0)[1],
            passed_entries,
        )

    def test_no_problem_empty_excluded_and_empty_kept(self):
        a = segment((0, 0, 10, 10), "a")
        c = segment((5, 0, 15, 10), "c")
        excluded = segment((2, 0, 12, 10), "b")
        self.assertEqual(restore((10, 15), [a, c], [excluded], min_x=5, min_y=0)[1], [a, c])
        self.assertEqual(restore((10, 15), [a, c], [], min_x=99, min_y=99)[1], [a, c])
        self.assertEqual(restore((10, 15), [], [excluded], min_x=1, min_y=1)[1], [])

    def test_nonuniform_starts_are_supported(self):
        items = [segment((x, 0, x + 12, 10), str(index)) for index, x in enumerate((0, 5, 11), 1)]
        result = restore((10, 23), [items[0], items[2]], [items[1]], min_x=2, min_y=0)
        self.assertEqual(result[1], items)

    def test_connection_regions_cover_overlap_contact_and_gap_deficits(self):
        self.assertEqual(MODULE._connection_region((0, 0, 10, 10), (8, 0, 18, 10), 0, 4, (10, 18)), (8, 0, 10, 10))
        self.assertEqual(MODULE._connection_region((0, 0, 10, 10), (10, 0, 20, 10), 0, 4, (10, 20)), (8, 0, 12, 10))
        self.assertEqual(MODULE._connection_region((0, 0, 10, 10), (16, 0, 26, 10), 0, 4, (10, 26)), (8, 0, 18, 10))

    def test_connection_gate_uses_exact_positive_semantics(self):
        region = (1, 1, 3, 3)
        for value in (np.nextafter(np.float32(0), np.float32(1)), 0.01, 0.5, 0.99, 1.0):
            with self.subTest(value=float(value)):
                mask = np.zeros((4, 4), dtype=np.float32)
                mask[1, 1] = value
                self.assertTrue(MODULE._connection_has_paintable_mask(mask, region))
        self.assertFalse(MODULE._connection_has_paintable_mask(np.zeros((4, 4), dtype=np.float32), region))

    def test_unrelated_zero_outside_connection_does_not_suppress_repair(self):
        source_mask = np.zeros((10, 20), dtype=np.float32)
        source_mask[:, 8:12] = 0.5
        entries = source_segments(source_mask, [(0, 0, 10, 10), (5, 0, 15, 10), (10, 0, 20, 10)])
        result = restore((10, 20), [entries[0], entries[2]], [entries[1]], min_x=4, min_y=0)
        self.assertEqual(result[1], entries)

    def test_all_zero_connection_suppresses_direct_and_expanded_candidates(self):
        source_mask = np.ones((30, 30), dtype=np.float32)
        source_mask[:, 8:22] = 0
        crops = [(0, y, 10, y + 10) for y in (0, 10, 20)] + [
            (10, y, 20, y + 10) for y in (0, 10, 20)
        ] + [(20, y, 30, y + 10) for y in (0, 10, 20)]
        entries = source_segments(source_mask, crops)
        kept = [entries[0], entries[2], entries[6], entries[8]]
        excluded = [entry for entry in entries if not any(entry is item for item in kept)]
        result = restore((30, 30), kept, excluded, min_x=4, min_y=0)
        self.assertEqual(result[1], [entries[0], entries[6], entries[2], entries[8]])

    def test_audited_eighteen_tile_mask_gate_produces_twelve(self):
        height, width = 1024, 2048
        source_mask = np.zeros((height, width), dtype=np.float32)
        source_mask[:, :696] = 1
        source_mask[:, 1352:] = 1
        crops = [
            (x, y, x + 512, y + 512)
            for y in (0, 256, 512)
            for x in (0, 307, 614, 922, 1229, 1536)
        ]
        entries = source_segments(source_mask, crops)
        kept_indices = {1, 2, 5, 6, 13, 14, 17, 18}
        kept = [entry for index, entry in enumerate(entries, 1) if index in kept_indices]
        excluded = [entry for index, entry in enumerate(entries, 1) if index not in kept_indices]
        result = restore((height, width), kept, excluded, min_x=204, min_y=204)
        self.assertEqual(
            [int(entry.label) for entry in result[1]],
            [1, 2, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18],
        )

    def test_conflicting_overlap_masks_fail_clearly(self):
        a = segment((0, 0, 10, 10), "a", active=True)
        b = segment((5, 0, 15, 10), "b", active=False)
        c = segment((10, 0, 20, 10), "c", active=True)
        with self.assertRaisesRegex(ValueError, "identical source-mask values"):
            restore((10, 20), [a, c], [b], min_x=1, min_y=0)

    def test_order_and_identity_are_geometry_reconstructed(self):
        a = segment((0, 0, 10, 10), "a")
        b = segment((5, 0, 15, 10), "b")
        c = segment((10, 0, 20, 10), "c")
        result = restore((10, 20), [c, a], [b], min_x=1, min_y=0)
        self.assertEqual(result[1], [a, b, c])
        self.assertIs(result[1][1], b)

    def test_validation(self):
        empty = ((10, 10), [])
        with self.assertRaisesRegex(ValueError, "expected 'mask' or 'bbox'"):
            MODULE.restore_tile_segs_gaps(empty, empty, "other", 0, 0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            MODULE.restore_tile_segs_gaps(empty, empty, "mask", -1, 0)
        with self.assertRaisesRegex(ValueError, "matching SEGS source shapes"):
            MODULE.restore_tile_segs_gaps(empty, ((11, 10), []), "mask", 0, 0)
        item = segment((0, 0, 10, 10), "x")
        with self.assertRaisesRegex(ValueError, "disjoint"):
            restore((10, 10), [item], [item])
        duplicate = segment((0, 0, 10, 10), "y")
        with self.assertRaisesRegex(ValueError, "duplicate crop_regions"):
            restore((10, 10), [item], [duplicate])

    def test_mask_to_tile_filter_restore_interoperability(self):
        mask = torch.zeros((1, 12, 30), dtype=torch.float32)
        mask[:, :, :8] = 1
        mask[:, :, 22:] = 1
        original, _ = MASK_TILE.mask_to_tile_segs(mask, "uniform_grid", 14, 12, 0, 0, 3, 1, 2, 0, 3, 1)
        kept, excluded = FILTER.partition_tile_segs(original, "mask")
        result = MODULE.restore_tile_segs_gaps(kept, excluded, "mask", 4, 0)
        self.assertEqual(result[0], original[0])
        self.assertTrue(all(any(entry is source for source in original[1]) for entry in result[1]))

    def test_schema(self):
        schema = MODULE.RestoreTileSEGSGaps.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteRestoreTileSEGSGaps")
        self.assertEqual(schema.display_name, "Restore Tile SEGS Gaps")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual([entry.io_type for entry in schema.inputs], ["SEGS", "SEGS", "COMBO", "INT", "INT"])
        self.assertEqual([output.io_type for output in schema.outputs], ["SEGS"])
        self.assertFalse(schema.outputs[0].is_output_list)


if __name__ == "__main__":
    unittest.main()
