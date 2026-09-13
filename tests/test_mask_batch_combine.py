import importlib.util
import sys
import unittest
from pathlib import Path

import torch
from comfy_extras.nodes_mask import MaskComposite

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_mask_batch_combine_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
COMBINE = sys.modules[f"{SPEC.name}.mask_batch_combine"]


def core_fold(masks: torch.Tensor) -> torch.Tensor:
    result = masks[0:1]
    for index in range(1, masks.shape[0]):
        result = MaskComposite.execute(result, masks[index : index + 1], 0, 0, "add").result[0]
    return result


class MaskBatchToMaskTests(unittest.TestCase):
    def test_single_mask_preserves_singleton_batch(self):
        masks = torch.rand(1, 9, 13)
        result = COMBINE.combine_mask_batch(masks)
        self.assertIs(result, masks)
        self.assertEqual(result.shape, (1, 9, 13))

    def test_rank_two_is_one_mask(self):
        mask = torch.rand(7, 11)
        result = COMBINE.combine_mask_batch(mask)
        self.assertEqual(result.shape, (1, 7, 11))
        self.assertTrue(torch.equal(result[0], mask))

    def test_three_disjoint_non_square_masks(self):
        masks = torch.zeros(3, 12, 19)
        masks[0, 1:4, 2:6] = 1
        masks[1, 5:8, 7:12] = 1
        masks[2, 8:12, 14:19] = 1
        result = COMBINE.combine_mask_batch(masks)
        self.assertEqual(result.shape, (1, 12, 19))
        self.assertTrue(torch.equal(result, masks.sum(dim=0, keepdim=True)))

    def test_overlapping_binary_masks_saturate(self):
        masks = torch.zeros(3, 8, 10)
        masks[:, 2:6, 3:8] = 1
        result = COMBINE.combine_mask_batch(masks)
        self.assertTrue(torch.equal(result, masks.sum(dim=0, keepdim=True).clamp(0, 1)))
        self.assertEqual(result[0, 3, 4].item(), 1)

    def test_overlapping_fractional_masks_match_repeated_core(self):
        masks = torch.zeros(3, 6, 9)
        masks[0, 1:5, 2:8] = 0.2
        masks[1, 1:5, 2:8] = 0.35
        masks[2, 1:5, 2:8] = 0.7
        result = COMBINE.combine_mask_batch(masks)
        expected = core_fold(masks)
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result[0, 2, 3].item(), 1)

    def test_intermediate_clamping_exactly_matches_core(self):
        masks = torch.tensor([[[0.8]], [[0.8]], [[-0.4]]], dtype=torch.float32)
        result = COMBINE.combine_mask_batch(masks)
        expected = core_fold(masks)
        self.assertTrue(torch.equal(result, expected))
        self.assertAlmostEqual(result.item(), 0.6)
        self.assertNotEqual(result.item(), masks.sum().clamp(0, 1).item())

    def test_dtype_and_device_are_preserved(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                masks = torch.rand(3, 5, 7).to(dtype)
                result = COMBINE.combine_mask_batch(masks)
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(result.device, masks.device)
                self.assertTrue(torch.equal(result, core_fold(masks)))

    def test_empty_batch_and_malformed_ranks_fail(self):
        with self.assertRaisesRegex(ValueError, "at least one mask"):
            COMBINE.combine_mask_batch(torch.empty(0, 5, 7))
        for masks in (torch.zeros(3), torch.zeros(1, 1, 3, 4)):
            with self.subTest(shape=tuple(masks.shape)), self.assertRaisesRegex(ValueError, "rank-2 or rank-3"):
                COMBINE.combine_mask_batch(masks)

    def test_schema_is_ordinary_mask_transport(self):
        schema = COMBINE.MaskBatchToMask.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteMaskBatchToMask")
        self.assertEqual(schema.display_name, "Mask Batch to Mask")
        self.assertEqual(schema.category, "Utility Suite/Mask")
        self.assertEqual([item.id for item in schema.inputs], ["masks"])
        self.assertEqual(schema.outputs[0].io_type, "MASK")
        self.assertFalse(schema.outputs[0].is_output_list)

    def test_no_custom_node_runtime_dependency(self):
        source = (PACKAGE_ROOT / "mask_batch_combine.py").read_text(encoding="utf-8").lower()
        for dependency in ("impact", "was", "essentials", "kjnodes"):
            self.assertNotIn(dependency, source)


if __name__ == "__main__":
    unittest.main()
