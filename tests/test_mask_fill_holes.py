import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_mask_fill_holes_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
FILL_HOLES = sys.modules[f"{SPEC.name}.mask_fill_holes"]


class MaskFillHolesTests(unittest.TestCase):
    def test_fills_enclosed_hole_and_preserves_exterior_background(self):
        mask = torch.zeros(1, 11, 15)
        mask[0, 2:9, 3:12] = 1
        mask[0, 4:7, 6:9] = 0

        result = FILL_HOLES.fill_mask_holes(mask)

        expected = mask.clone()
        expected[0, 4:7, 6:9] = 1
        self.assertTrue(torch.equal(result, expected))

    def test_uses_was_positive_value_threshold_and_binary_output(self):
        mask = torch.zeros(1, 9, 13)
        mask[0, 1:8, 2:11] = 0.001
        mask[0, 3:6, 5:8] = 0
        mask[0, 0, 0] = -1

        result = FILL_HOLES.fill_mask_holes(mask)

        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(result[0, 4, 6].item(), 1)
        self.assertEqual(result[0, 2, 3].item(), 1)
        self.assertEqual(result[0, 0, 0].item(), 0)

    def test_processes_each_batch_item_independently_and_returns_bhw(self):
        mask = torch.zeros(2, 10, 14)
        mask[0, 1:9, 1:7] = 1
        mask[0, 3:6, 3:5] = 0
        mask[1, 1:9, 7:13] = 1
        mask[1, 4:7, 9:11] = 0

        result = FILL_HOLES.fill_mask_holes(mask)

        self.assertEqual(result.shape, (2, 10, 14))
        self.assertEqual(result[0, 4, 4].item(), 1)
        self.assertEqual(result[1, 5, 10].item(), 1)
        self.assertEqual(result[0, 5, 10].item(), 0)
        self.assertEqual(result[1, 4, 4].item(), 0)

    def test_rank_two_is_normalized_to_single_item_batch(self):
        result = FILL_HOLES.fill_mask_holes(torch.zeros(7, 12))
        self.assertEqual(result.shape, (1, 7, 12))

    def test_border_connected_background_is_not_filled(self):
        mask = torch.ones(1, 9, 11)
        mask[0, 0:6, 5] = 0
        result = FILL_HOLES.fill_mask_holes(mask)
        self.assertTrue(torch.equal(result, mask))

    def test_rejects_was_rank_four_layout_and_empty_dimensions(self):
        with self.assertRaisesRegex(ValueError, r"\[H,W\] or \[B,H,W\]"):
            FILL_HOLES.fill_mask_holes(torch.zeros(1, 1, 7, 12))
        with self.assertRaisesRegex(ValueError, "nonempty dimensions"):
            FILL_HOLES.fill_mask_holes(torch.zeros(1, 0, 12))

    def test_schema_and_registration(self):
        schema = FILL_HOLES.MaskFillHoles.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteMaskFillHoles")
        self.assertEqual(schema.display_name, "Mask Fill Holes")
        self.assertEqual(schema.category, "Utility Suite/Mask")
        self.assertEqual([item.id for item in schema.inputs], ["mask"])
        self.assertEqual(schema.outputs[0].io_type, "MASK")
        self.assertFalse(schema.outputs[0].is_output_list)

        extension = PACKAGE.UtilitySuiteExtension()
        node_types = __import__("asyncio").run(extension.get_node_list())
        self.assertIn(FILL_HOLES.MaskFillHoles, node_types)

    def test_no_third_party_node_runtime_dependency(self):
        source = (PACKAGE_ROOT / "mask_fill_holes.py").read_text(encoding="utf-8").lower()
        for dependency in ("impact", "kjnodes", "essentials", "was-ns"):
            self.assertNotIn(dependency, source)


if __name__ == "__main__":
    unittest.main()
