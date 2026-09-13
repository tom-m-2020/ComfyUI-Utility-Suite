import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_mask_fill_combined_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
COMBINED = sys.modules[f"{SPEC.name}.mask_fill_combined"]


def fill(mask, bbox_fill=False, drop_size=10, contour_fill=False):
    return COMBINED.fill_combined_mask(mask, bbox_fill, drop_size, contour_fill)


class MaskFillCombinedTests(unittest.TestCase):
    def test_disconnected_regions_stay_in_one_mask(self):
        mask = torch.zeros(1, 30, 42)
        mask[0, 2:8, 3:10] = 0.25
        mask[0, 13:21, 17:26] = 0.5
        mask[0, 23:29, 34:41] = 0.75
        result = fill(mask)
        expected = (mask * 255).to(torch.uint8).to(torch.float32) / 255.0
        self.assertEqual(result.shape, (1, 30, 42))
        self.assertTrue(torch.equal(result, expected))

    def test_rank_two_and_non_square_canvas(self):
        mask = torch.zeros(17, 31)
        mask[2:15, 5:27] = 1
        result = fill(mask)
        self.assertEqual(result.shape, (1, 17, 31))
        self.assertTrue(torch.equal(result[0], mask))

    def test_normal_mask_batch_produces_one_combined_item_per_plane(self):
        mask = torch.zeros(2, 20, 24)
        mask[0, 2:10, 3:12] = 0.4
        mask[1, 11:19, 14:23] = 0.8
        result = fill(mask)
        self.assertEqual(result.shape, (2, 20, 24))
        self.assertGreater(result[0, 4, 5].item(), 0)
        self.assertEqual(result[0, 14, 17].item(), 0)
        self.assertGreater(result[1, 14, 17].item(), 0)

    def test_drop_size_and_contour_fill_are_ignored_in_combined_mode(self):
        mask = torch.zeros(1, 20, 25)
        mask[0, 4:14, 5:7] = 0.6
        mask[0, 12:14, 5:18] = 0.6
        baseline = fill(mask, drop_size=1, contour_fill=False)
        self.assertTrue(torch.equal(fill(mask, drop_size=16384, contour_fill=False), baseline))
        self.assertTrue(torch.equal(fill(mask, drop_size=1, contour_fill=True), baseline))

    def test_bbox_fill_uses_combined_reference_geometry(self):
        mask = torch.zeros(1, 30, 35)
        mask[0, 5:10, 6:9] = 1
        mask[0, 15:20, 20:24] = 1
        result = fill(mask, bbox_fill=True)
        self.assertEqual(result.shape, (1, 30, 35))
        self.assertGreaterEqual(result.count_nonzero().item(), mask.count_nonzero().item())

    def test_empty_and_single_pixel_follow_reference_empty_result(self):
        empty = fill(torch.zeros(1, 9, 13))
        single = torch.zeros(1, 9, 13)
        single[0, 4, 6] = 1
        single_result = fill(single)
        for result in (empty, single_result):
            self.assertEqual(result.shape, (1, 9, 13))
            self.assertEqual(result.dtype, torch.float32)
            self.assertEqual(result.device.type, "cpu")
            self.assertEqual(result.count_nonzero().item(), 0)

    def test_rank_validation(self):
        with self.assertRaisesRegex(ValueError, "rank-2 or rank-3"):
            fill(torch.zeros(1, 1, 1, 2, 2))
        with self.assertRaisesRegex(ValueError, "nonempty spatial"):
            fill(torch.zeros(1, 0, 3))

    def test_schema_matches_mask_to_mask_batch_widgets(self):
        schema = COMBINED.MaskFillCombined.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteMaskFillCombined")
        self.assertEqual(schema.display_name, "Mask Fill Combined")
        self.assertEqual(schema.category, "Utility Suite/Mask")
        self.assertEqual([item.id for item in schema.inputs], ["mask", "bbox_fill", "drop_size", "contour_fill"])
        self.assertEqual(schema.inputs[1].default, False)
        self.assertEqual(schema.inputs[2].default, 10)
        self.assertEqual(schema.inputs[2].min, 1)
        self.assertEqual(schema.inputs[3].default, False)
        self.assertEqual(schema.outputs[0].io_type, "MASK")
        self.assertFalse(schema.outputs[0].is_output_list)

    def test_no_custom_node_runtime_dependency(self):
        source = (PACKAGE_ROOT / "mask_fill_combined.py").read_text(encoding="utf-8").lower()
        for dependency in ("impact", "kjnodes", "essentials", "was-ns"):
            self.assertNotIn(dependency, source)


if __name__ == "__main__":
    unittest.main()
