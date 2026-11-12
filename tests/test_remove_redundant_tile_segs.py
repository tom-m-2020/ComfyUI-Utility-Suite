from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_remove_redundant_tile_segs_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.remove_redundant_tile_segs"]
FILTER = sys.modules[f"{SPEC.name}.filter_tile_segs"]
RESTORE = sys.modules[f"{SPEC.name}.restore_tile_segs_gaps"]

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)


def source_segments(source_mask, crops):
    return [
        SEG(None, source_mask[y1:y2, x1:x2].copy(), 1.0, crop, crop, str(index), None)
        for index, crop in enumerate(crops, 1)
        for x1, y1, x2, y2 in [crop]
    ]


def audited_grid():
    height, width = 1024, 2048
    source_mask = np.zeros((height, width), dtype=np.float32)
    source_mask[:, :696] = 1
    source_mask[:, 1352:] = 1
    crops = [
        (x, y, x + 512, y + 512)
        for y in (0, 256, 512)
        for x in (0, 307, 614, 922, 1229, 1536)
    ]
    return (height, width), source_segments(source_mask, crops)


class RemoveRedundantTileSEGSTests(unittest.TestCase):
    def test_matches_explicit_filter_then_restore_workflow(self):
        source_shape, entries = audited_grid()
        input_segs = (source_shape, entries)
        kept, excluded = FILTER.partition_tile_segs(input_segs, "mask")
        expected = RESTORE.restore_tile_segs_gaps(kept, excluded, "mask", 204, 204)

        actual = MODULE.remove_redundant_tile_segs(input_segs, "mask", 204, 204)

        self.assertEqual([int(entry.label) for entry in kept[1]], [1, 2, 5, 6, 13, 14, 17, 18])
        self.assertEqual([int(entry.label) for entry in actual[1]], [1, 2, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18])
        self.assertEqual(actual, expected)
        self.assertIs(actual[0], source_shape)
        self.assertTrue(all(any(entry is original for original in entries) for entry in actual[1]))

    def test_zero_minimums_return_filter_survivors(self):
        source_shape, entries = audited_grid()
        input_segs = (source_shape, entries)
        expected, _ = FILTER.partition_tile_segs(input_segs, "mask")
        actual = MODULE.remove_redundant_tile_segs(input_segs, "mask", 0, 0)
        self.assertEqual(actual, expected)

    def test_bbox_mode_matches_explicit_workflow(self):
        source_mask = np.ones((10, 20), dtype=np.float32)
        entries = source_segments(source_mask, [(0, 0, 10, 10), (5, 0, 15, 10), (10, 0, 20, 10)])
        input_segs = ((10, 20), entries)
        kept, excluded = FILTER.partition_tile_segs(input_segs, "bbox")
        expected = RESTORE.restore_tile_segs_gaps(kept, excluded, "bbox", 1, 0)
        actual = MODULE.remove_redundant_tile_segs(input_segs, "bbox", 1, 0)
        self.assertEqual(actual, expected)

    def test_empty_segs_preserves_header_identity(self):
        source_shape = (12, 30)
        result = MODULE.remove_redundant_tile_segs((source_shape, []), "mask", 128, 128)
        self.assertIs(result[0], source_shape)
        self.assertEqual(result[1], [])

    def test_input_entries_are_not_mutated(self):
        source_shape, entries = audited_grid()
        snapshots = [tuple(entry) for entry in entries]
        MODULE.remove_redundant_tile_segs((source_shape, entries), "mask", 204, 204)
        self.assertEqual([tuple(entry) for entry in entries], snapshots)

    def test_validation_is_inherited_from_existing_stages(self):
        with self.assertRaisesRegex(ValueError, "expected 'mask' or 'bbox'"):
            MODULE.remove_redundant_tile_segs(((10, 10), []), "other", 0, 0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            MODULE.remove_redundant_tile_segs(((10, 10), []), "mask", -1, 0)

    def test_schema_and_registration(self):
        schema = MODULE.RemoveRedundantTileSEGS.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteRemoveRedundantTileSEGS")
        self.assertEqual(schema.display_name, "Remove Redundant Tile SEGS")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual([entry.io_type for entry in schema.inputs], ["SEGS", "COMBO", "INT", "INT"])
        self.assertEqual([output.io_type for output in schema.outputs], ["SEGS"])
        self.assertFalse(schema.outputs[0].is_output_list)


if __name__ == "__main__":
    unittest.main()
