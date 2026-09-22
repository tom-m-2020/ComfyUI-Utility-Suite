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
    "utility_suite_invert_mask_segs_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.invert_mask_segs"]

from comfy_extras.nodes_mask import InvertMask

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)


def segment(index, mask, image=None):
    x1 = index * 5
    return SEG(image, mask, 0.25 + index, (x1, 0, x1 + 5, 3), (x1 + 1, 0, x1 + 4, 3), f"seg-{index}", object())


class InvertMaskSEGSTests(unittest.TestCase):
    def test_single_binary_mask(self):
        item = segment(0, np.array([[0, 1, 0, 1, 1]] * 3, dtype=np.float32))
        _, entries = MODULE.invert_mask_segs(((3, 5), [item]))
        np.testing.assert_array_equal(entries[0].cropped_mask, 1.0 - item.cropped_mask)

    def test_multiple_masks_are_inverted_independently_in_order(self):
        entries = [
            segment(0, np.zeros((3, 5), dtype=np.float32)),
            segment(1, np.full((3, 5), 0.2, dtype=np.float32)),
            segment(2, np.ones((3, 5), dtype=np.float32)),
        ]
        header, result = MODULE.invert_mask_segs(((3, 15), entries))
        self.assertEqual(header, (3, 15))
        self.assertEqual([entry.label for entry in result], ["seg-0", "seg-1", "seg-2"])
        np.testing.assert_array_equal(result[0].cropped_mask, 1)
        np.testing.assert_allclose(result[1].cropped_mask, 0.8)
        np.testing.assert_array_equal(result[2].cropped_mask, 0)

    def test_grayscale_values(self):
        values = np.array([0.0, 0.2, 0.5, 0.8, 1.0], dtype=np.float32)
        item = segment(0, np.stack([values] * 3))
        result = MODULE.invert_mask_segs(((3, 5), [item]))[1][0].cropped_mask
        np.testing.assert_allclose(result[0], [1.0, 0.8, 0.5, 0.2, 0.0], atol=1e-7)

    def test_soft_gradient_and_input_are_not_mutated(self):
        mask = np.linspace(0, 1, 15, dtype=np.float32).reshape(3, 5)
        before = mask.copy()
        item = segment(0, mask)
        result = MODULE.invert_mask_segs(((3, 5), [item]))[1][0]
        np.testing.assert_allclose(result.cropped_mask, 1 - before)
        np.testing.assert_array_equal(mask, before)
        self.assertIs(item.cropped_mask, mask)
        self.assertIsNot(result.cropped_mask, mask)

    def test_zero_and_one_masks(self):
        zero = segment(0, np.zeros((3, 5), dtype=np.float32))
        one = segment(1, np.ones((3, 5), dtype=np.float32))
        result = MODULE.invert_mask_segs(((3, 10), [zero, one]))[1]
        np.testing.assert_array_equal(result[0].cropped_mask, 1)
        np.testing.assert_array_equal(result[1].cropped_mask, 0)

    def test_header_cardinality_order_and_fields_are_preserved(self):
        header = [100, 200]
        image = torch.rand((1, 3, 5, 3))
        entries = [segment(0, np.zeros((3, 5), dtype=np.float32), image), segment(1, torch.ones((3, 5)))]
        result_header, result = MODULE.invert_mask_segs((header, entries))
        self.assertIs(result_header, header)
        self.assertEqual(len(result), 2)
        for old, new in zip(entries, result):
            self.assertIsNot(old, new)
            for field in SEG._fields:
                if field != "cropped_mask":
                    self.assertIs(getattr(new, field), getattr(old, field))

    def test_singleton_tensor_mask_becomes_normal_2d_mask(self):
        mask = torch.tensor([[[0.0, 0.2, 0.5, 0.8, 1.0]] * 3])
        result = MODULE.invert_mask_segs(((3, 5), [segment(0, mask)]))[1][0].cropped_mask
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 5))
        np.testing.assert_allclose(result[0], [1.0, 0.8, 0.5, 0.2, 0.0], atol=1e-7)

    def test_empty_segs_preserves_header(self):
        header = [20, 30]
        result = MODULE.invert_mask_segs((header, []))
        self.assertIs(result[0], header)
        self.assertEqual(result[1], [])

    def test_missing_and_invalid_masks_fail_like_existing_segs_mask_nodes(self):
        base = segment(0, np.zeros((3, 5), dtype=np.float32))
        with self.assertRaisesRegex(TypeError, "no cropped_mask"):
            MODULE.invert_mask_segs(((3, 5), [base._replace(cropped_mask=None)]))
        with self.assertRaisesRegex(ValueError, "does not match crop_region"):
            MODULE.invert_mask_segs(((3, 5), [base._replace(cropped_mask=np.zeros((2, 5)))]))
        with self.assertRaisesRegex(ValueError, "rank-3 form"):
            MODULE.invert_mask_segs(((3, 5), [base._replace(cropped_mask=np.zeros((2, 3, 5)))]))
        with self.assertRaisesRegex(TypeError, "NumPy array or Torch tensor"):
            MODULE.invert_mask_segs(((3, 5), [base._replace(cropped_mask=[[0] * 5] * 3)]))

    def test_malformed_segs_and_crop_fail_strictly(self):
        with self.assertRaisesRegex(TypeError, "expects SEGS"):
            MODULE.invert_mask_segs([])
        item = segment(0, np.zeros((3, 5), dtype=np.float32))._replace(crop_region=(0, 0, 6, 3))
        with self.assertRaisesRegex(ValueError, "outside source canvas"):
            MODULE.invert_mask_segs(((3, 5), [item]))

    def test_core_invert_decompose_edit_assemble_equivalence(self):
        mask = np.linspace(0, 1, 15, dtype=np.float32).reshape(3, 5)
        item = segment(0, mask, image=None)
        header = (100, 200)
        # Impact decompose returns header and SEG_ELTs unchanged. Core Invert
        # Mask operates on the extracted MASK, and Impact Edit replaces only
        # cropped_mask before Assemble retains the original header.
        core_result = InvertMask.execute(torch.from_numpy(mask).unsqueeze(0)).result[0][0].numpy()
        reference = (header, [item._replace(cropped_mask=core_result)])
        actual = MODULE.invert_mask_segs((header, [item]))
        self.assertIs(actual[0], reference[0])
        self.assertEqual(len(actual[1]), 1)
        np.testing.assert_array_equal(actual[1][0].cropped_mask, reference[1][0].cropped_mask)
        for field in SEG._fields:
            if field != "cropped_mask":
                self.assertIs(getattr(actual[1][0], field), getattr(reference[1][0], field))

    def test_schema(self):
        schema = MODULE.InvertMaskSEGS.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteInvertMaskSEGS")
        self.assertEqual(schema.display_name, "Invert Mask (SEGS)")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual([entry.io_type for entry in schema.inputs], ["SEGS"])
        self.assertEqual([entry.io_type for entry in schema.outputs], ["SEGS"])


if __name__ == "__main__":
    unittest.main()
