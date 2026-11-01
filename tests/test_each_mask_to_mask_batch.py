import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_each_mask_batch_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.each_mask_to_mask_batch"]


class EachMaskToMaskBatchTests(unittest.TestCase):
    def test_schema_uses_required_autogrowing_masks_and_matching_widgets(self):
        schema = MODULE.EachMaskToMaskBatch.define_schema()

        self.assertEqual(schema.node_id, "UtilitySuiteEachMaskToMaskBatch")
        self.assertEqual(schema.display_name, "Each Mask to Mask Batch")
        self.assertEqual(schema.category, "Utility Suite/Mask")
        self.assertEqual(
            [item.id for item in schema.inputs],
            ["masks", "bbox_fill", "drop_size", "contour_fill"],
        )
        autogrow = schema.inputs[0]
        self.assertEqual(autogrow.template.prefix, "mask")
        self.assertEqual(autogrow.template.min, 1)
        self.assertEqual(autogrow.template.max, 100)
        self.assertEqual(autogrow.template.get_all()[0].id, "mask0")
        self.assertEqual(schema.inputs[1].default, False)
        self.assertEqual(schema.inputs[2].default, 10)
        self.assertEqual(schema.inputs[2].min, 1)
        self.assertEqual(schema.inputs[3].default, False)
        self.assertEqual(schema.outputs[0].io_type, "MASK")
        self.assertFalse(schema.outputs[0].is_output_list)
        nodes = asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list())
        self.assertIn(MODULE.EachMaskToMaskBatch, nodes)

    def test_each_input_is_separated_and_batched_in_slot_order(self):
        mask0 = torch.zeros(1, 30, 40)
        mask0[0, 2:10, 3:12] = 0.5
        mask0[0, 16:27, 25:37] = 0.75
        mask1 = torch.zeros(1, 30, 40)
        mask1[0, 8:22, 14:24] = 1

        result = MODULE.each_mask_to_mask_batch(
            {"mask1": mask1, "mask0": mask0}, False, 1, False
        )
        expected = torch.cat(
            [
                PACKAGE.mask_batch.mask_to_mask_batch(mask0, False, 1, False),
                PACKAGE.mask_batch.mask_to_mask_batch(mask1, False, 1, False),
            ]
        )

        self.assertTrue(torch.equal(result, expected))

    def test_empty_input_retains_its_own_zero_mask(self):
        empty = torch.zeros(1, 20, 24)
        populated = torch.zeros(1, 20, 24)
        populated[0, 4:16, 6:18] = 1

        result = MODULE.each_mask_to_mask_batch(
            {"mask0": empty, "mask1": populated}, False, 1, False
        )

        self.assertEqual(result.shape, (2, 20, 24))
        self.assertEqual(result[0].count_nonzero().item(), 0)
        self.assertTrue(torch.equal(result[1], populated[0]))

    def test_batched_slot_processes_each_plane_independently(self):
        masks = torch.zeros(2, 18, 22)
        masks[0, 2:10, 3:11] = 1
        masks[1, 8:16, 12:20] = 1

        result = MODULE.each_mask_to_mask_batch({"mask0": masks}, False, 1, False)

        self.assertEqual(result.shape, masks.shape)
        self.assertTrue(torch.equal(result, masks))

    def test_later_masks_resize_to_mask0_canvas_before_processing(self):
        from comfy_extras.nodes_post_processing import batch_masks

        mask0 = torch.zeros(1, 20, 30)
        mask0[0, 2:12, 3:15] = 1
        mask1 = torch.zeros(1, 10, 15)
        mask1[0, 3:9, 7:14] = 1

        result = MODULE.each_mask_to_mask_batch(
            {"mask0": mask0, "mask1": mask1}, False, 1, False
        )
        prototype_batch = batch_masks([mask0, mask1])
        expected = torch.cat(
            [
                PACKAGE.mask_batch.mask_to_mask_batch(
                    item.unsqueeze(0), False, 1, False
                )
                for item in prototype_batch
            ]
        )

        self.assertEqual(result.shape[1:], (20, 30))
        self.assertEqual(result.shape[0], 2)
        self.assertTrue(torch.equal(result, expected))

    def test_widgets_are_forwarded_to_every_input(self):
        masks = torch.zeros(2, 24, 28)
        masks[0, 3:18, 4:7] = 0.5
        masks[0, 15:18, 4:16] = 0.5
        masks[1, 4:20, 9:25] = 0.75
        masks[1, 9:14, 14:20] = 0

        actual = MODULE.each_mask_to_mask_batch(
            {"mask0": masks[0], "mask1": masks[1]}, True, 1, True
        )
        expected = torch.cat(
            [
                PACKAGE.mask_batch.mask_to_mask_batch(masks[index], True, 1, True)
                for index in range(2)
            ]
        )

        self.assertTrue(torch.equal(actual, expected))

    def test_requires_mask0_and_valid_mask_rank(self):
        with self.assertRaisesRegex(ValueError, "requires mask0"):
            MODULE.each_mask_to_mask_batch({}, False, 1, False)
        with self.assertRaisesRegex(ValueError, r"\[B,H,W\]"):
            MODULE.each_mask_to_mask_batch(
                {"mask0": torch.zeros(1, 1, 4, 5)}, False, 1, False
            )


if __name__ == "__main__":
    unittest.main()
