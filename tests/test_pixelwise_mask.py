from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_pixelwise_mask_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.pixelwise_mask"]


class PixelwiseMaskTests(unittest.TestCase):
    def test_min_keeps_darker_values(self):
        mask1 = torch.tensor([[0.2, 0.8], [0.7, 0.1]])
        mask2 = torch.tensor([[0.6, 0.3], [0.4, 0.9]])
        result = MODULE.pixelwise_masks(mask1, mask2, "min")
        torch.testing.assert_close(result, torch.tensor([[[0.2, 0.3], [0.4, 0.1]]]))

    def test_max_keeps_brighter_values(self):
        mask1 = torch.tensor([[0.2, 0.8], [0.7, 0.1]])
        mask2 = torch.tensor([[0.6, 0.3], [0.4, 0.9]])
        result = MODULE.pixelwise_masks(mask1, mask2, "max")
        torch.testing.assert_close(result, torch.tensor([[[0.6, 0.8], [0.7, 0.9]]]))

    def test_smaller_mask2_changes_only_top_left_overlap(self):
        mask1 = torch.full((1, 3, 4), 0.6)
        mask2 = torch.tensor([[0.2, 0.9], [0.8, 0.1]])
        result = MODULE.pixelwise_masks(mask1, mask2, "min")
        expected = mask1.clone()
        expected[:, :2, :2] = torch.tensor([[[0.2, 0.6], [0.6, 0.1]]])
        torch.testing.assert_close(result, expected)

    def test_larger_mask2_is_cut_to_mask1_canvas(self):
        mask1 = torch.tensor([[0.7, 0.2], [0.3, 0.8]])
        mask2 = torch.full((1, 4, 5), 0.5)
        result = MODULE.pixelwise_masks(mask1, mask2, "max")
        self.assertEqual(tuple(result.shape), (1, 2, 2))
        torch.testing.assert_close(result, torch.tensor([[[0.7, 0.5], [0.5, 0.8]]]))

    def test_preserves_mask1_dtype_device_and_inputs(self):
        mask1 = torch.tensor([[0.2, 0.8]], dtype=torch.float64)
        mask2 = torch.tensor([[0.5, 0.1]], dtype=torch.float32)
        mask1_before = mask1.clone()
        mask2_before = mask2.clone()
        result = MODULE.pixelwise_masks(mask1, mask2, "max")
        self.assertEqual(result.dtype, mask1.dtype)
        self.assertEqual(result.device, mask1.device)
        torch.testing.assert_close(mask1, mask1_before)
        torch.testing.assert_close(mask2, mask2_before)

    def test_rejects_mask_batches(self):
        with self.assertRaisesRegex(ValueError, r"\[H,W\] or \[1,H,W\]"):
            MODULE.pixelwise_masks(torch.zeros((2, 3, 4)), torch.zeros((3, 4)), "min")
        with self.assertRaisesRegex(ValueError, r"\[H,W\] or \[1,H,W\]"):
            MODULE.pixelwise_masks(torch.zeros((3, 4)), torch.zeros((2, 3, 4)), "max")

    def test_schemas_and_registration(self):
        expected = [
            (
                MODULE.PixelwiseMinMaskAndMask,
                "UtilitySuitePixelwiseMinMaskAndMask",
                "Pixelwise Min (MASK & MASK)",
            ),
            (
                MODULE.PixelwiseMaxMaskAndMask,
                "UtilitySuitePixelwiseMaxMaskAndMask",
                "Pixelwise Max (MASK & MASK)",
            ),
        ]
        registered = asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list())
        for node_class, node_id, display_name in expected:
            schema = node_class.define_schema()
            self.assertEqual(schema.node_id, node_id)
            self.assertEqual(schema.display_name, display_name)
            self.assertEqual(schema.category, "Utility Suite/Mask")
            self.assertEqual([item.id for item in schema.inputs], ["mask1", "mask2"])
            self.assertEqual([item.io_type for item in schema.outputs], ["MASK"])
            self.assertIn(node_class, registered)


if __name__ == "__main__":
    unittest.main()
