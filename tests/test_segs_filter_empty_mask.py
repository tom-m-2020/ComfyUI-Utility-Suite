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
    "utility_suite_segs_filter_empty_mask_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.segs_filter_empty_mask"]
INSERT = sys.modules[f"{SPEC.name}.segs_insert_by_index"]

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)


def segment(label: str, mask) -> SEG:
    return SEG(object(), mask, 0.75, (1, 2, 5, 6), (1, 2, 5, 6), label, object())


class SEGSFilterEmptyMaskTests(unittest.TestCase):
    def setUp(self):
        self.header = (20, 30)
        self.empty = segment("empty", torch.zeros(4, 4))
        self.faint = segment("faint", torch.full((4, 4), 0.001))
        self.visible = segment("visible", torch.eye(4))
        self.entries = [self.empty, self.faint, self.visible]
        self.segs = (self.header, self.entries)

    def test_strict_matches_easy_use_exact_zero_semantics(self):
        filtered, remained, indices = MODULE.filter_segs_empty_mask(self.segs, True, True)
        self.assertEqual(filtered[1], [self.empty])
        self.assertEqual(remained[1], [self.faint, self.visible])
        self.assertEqual(indices, [0])

    def test_tolerant_mode_uses_eight_bit_visibility_threshold(self):
        below = np.nextafter(np.float32(1.0 / 255.0), np.float32(0.0))
        exact = np.float32(1.0 / 255.0)
        below_seg = segment("below", np.full((2, 3), below, dtype=np.float32))
        exact_seg = segment("exact", np.full((2, 3), exact, dtype=np.float32))

        filtered, remained, indices = MODULE.filter_segs_empty_mask(
            (self.header, [below_seg, exact_seg]), True, False
        )

        self.assertEqual(filtered[1], [below_seg])
        self.assertEqual(remained[1], [exact_seg])
        self.assertEqual(indices, [0])

    def test_nonempty_mode_is_the_complement_and_preserves_order(self):
        filtered, remained, indices = MODULE.filter_segs_empty_mask(self.segs, False, False)
        self.assertEqual(filtered[1], [self.visible])
        self.assertEqual(remained[1], [self.empty, self.faint])
        self.assertEqual(indices, [2])

    def test_none_mask_is_empty_like_easy_use(self):
        none_mask = segment("none", None)
        filtered, remained, indices = MODULE.filter_segs_empty_mask(
            (self.header, [none_mask, self.visible]), True, True
        )
        self.assertEqual(filtered[1], [none_mask])
        self.assertEqual(remained[1], [self.visible])
        self.assertEqual(indices, [0])

    def test_empty_input_preserves_header(self):
        filtered, remained, indices = MODULE.filter_segs_empty_mask((self.header, []), True, False)
        self.assertIs(filtered[0], self.header)
        self.assertIs(remained[0], self.header)
        self.assertEqual(filtered[1], [])
        self.assertEqual(remained[1], [])
        self.assertEqual(indices, [])

    def test_identity_metadata_and_input_are_preserved(self):
        original_entries = list(self.entries)
        filtered, remained, _ = MODULE.filter_segs_empty_mask(self.segs, True, False)
        self.assertEqual(self.entries, original_entries)
        self.assertIs(filtered[0], self.header)
        self.assertIs(remained[0], self.header)
        self.assertIs(filtered[1][0], self.empty)
        self.assertIs(filtered[1][1], self.faint)
        self.assertIs(remained[1][0], self.visible)

    def test_filter_modify_insert_restores_original_positions(self):
        filtered, remained, indices = MODULE.filter_segs_empty_mask(self.segs, True, False)
        modified = [entry._replace(label=f"{entry.label} modified") for entry in filtered[1]]

        reconstructed = INSERT.insert_segs_by_index(remained, (self.header, modified), indices)

        self.assertEqual([entry.label for entry in reconstructed[1]], ["empty modified", "faint modified", "visible"])
        self.assertIs(reconstructed[1][0], modified[0])
        self.assertIs(reconstructed[1][1], modified[1])
        self.assertIs(reconstructed[1][2], self.visible)

    def test_invalid_cropped_mask_fails_clearly(self):
        no_mask_field = object()
        with self.assertRaisesRegex(TypeError, "has no cropped_mask"):
            MODULE.filter_segs_empty_mask((self.header, [no_mask_field]), True, True)
        invalid_mask = segment("invalid", object())
        with self.assertRaisesRegex(TypeError, "tensor-compatible"):
            MODULE.filter_segs_empty_mask((self.header, [invalid_mask]), True, True)

    def test_schema_declares_boolean_modes_and_single_value_indices(self):
        schema = MODULE.SEGSFilterEmptyMask.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteSEGSFilterEmptyMask")
        self.assertEqual(schema.display_name, "SEGS Filter (empty mask)")
        self.assertEqual([entry.io_type for entry in schema.inputs], ["SEGS", "BOOLEAN", "BOOLEAN"])
        self.assertEqual(schema.inputs[1].label_on, "empty")
        self.assertEqual(schema.inputs[1].label_off, "nonempty")
        self.assertTrue(schema.inputs[1].default)
        self.assertTrue(schema.inputs[2].default)
        self.assertEqual([output.io_type for output in schema.outputs], ["SEGS", "SEGS", "SEGS_INDICES"])
        self.assertTrue(all(not output.is_output_list for output in schema.outputs))


if __name__ == "__main__":
    unittest.main()
