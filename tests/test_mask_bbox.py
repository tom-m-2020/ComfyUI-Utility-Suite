import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_mask_bbox_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MASK_BBOX = sys.modules[f"{SPEC.name}.mask_bbox"]


class MaskToBoundingBoxTests(unittest.TestCase):
    def test_exact_bounds_and_bbox_formats(self):
        mask = torch.zeros(1, 10, 12)
        mask[:, 3:8, 2:9] = 1
        cropped_mask, cropped_image, x, y, width, height, bbox, bounding_box = MASK_BBOX.mask_bounding_box(
            mask, 0
        )
        self.assertEqual((x, y, width, height), (2, 3, 7, 5))
        self.assertEqual(cropped_mask.shape, (1, 5, 7))
        self.assertEqual(cropped_image.shape, (1, 5, 7, 3))
        self.assertEqual(bbox, [(2, 3, 7, 5)])
        self.assertEqual(bounding_box, [{"x": 2, "y": 3, "width": 7, "height": 5}])

    def test_padding_is_clipped_to_canvas(self):
        mask = torch.zeros(1, 8, 9)
        mask[:, 1:4, 6:9] = 1
        result = MASK_BBOX.mask_bounding_box(mask, 3)
        self.assertEqual(result[2:6], (3, 0, 6, 7))
        self.assertEqual(result[0].shape, (1, 7, 6))

    def test_rank_two_mask_becomes_single_item_batch(self):
        mask = torch.zeros(7, 8)
        mask[2:5, 3:7] = 1
        result = MASK_BBOX.mask_bounding_box(mask, 0)
        self.assertEqual(result[0].shape, (1, 3, 4))
        self.assertEqual(result[1].shape, (1, 3, 4, 3))

    def test_batch_uses_union_bounds(self):
        mask = torch.zeros(2, 10, 12)
        mask[0, 1:3, 2:4] = 1
        mask[1, 6:9, 8:11] = 1
        result = MASK_BBOX.mask_bounding_box(mask, 0)
        self.assertEqual(result[2:6], (2, 1, 9, 8))
        self.assertEqual(result[0].shape, (2, 8, 9))

    def test_optional_image_is_cropped_and_batch_repeated(self):
        mask = torch.zeros(3, 8, 10)
        mask[:, 2:7, 4:9] = 1
        image = torch.rand(1, 8, 10, 4)
        result = MASK_BBOX.mask_bounding_box(mask, 0, image)
        self.assertEqual(result[1].shape, (3, 5, 5, 4))
        self.assertTrue(torch.equal(result[1][0], result[1][1]))
        self.assertTrue(torch.equal(result[1][1], result[1][2]))

    def test_optional_image_is_truncated_to_mask_batch(self):
        mask = torch.ones(1, 4, 5)
        image = torch.rand(3, 4, 5, 3)
        result = MASK_BBOX.mask_bounding_box(mask, 0, image)
        self.assertEqual(result[1].shape, (1, 4, 5, 3))
        self.assertTrue(torch.equal(result[1][0], image[0]))

    def test_optional_image_is_resized_to_mask_canvas(self):
        mask = torch.ones(1, 6, 8)
        image = torch.rand(1, 3, 4, 3)
        result = MASK_BBOX.mask_bounding_box(mask, 0, image)
        self.assertEqual(result[1].shape, (1, 6, 8, 3))

    def test_empty_mask_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "nonzero mask pixel"):
            MASK_BBOX.mask_bounding_box(torch.zeros(1, 8, 9), 0)

    def test_schema_has_no_blur_and_declares_bbox_lists(self):
        schema = MASK_BBOX.MaskToBoundingBox.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteMaskToBoundingBox")
        self.assertEqual(schema.display_name, "Mask to Bounding Box")
        self.assertEqual([item.id for item in schema.inputs], ["mask", "padding", "image_optional"])
        self.assertNotIn("blur", [item.id for item in schema.inputs])
        self.assertEqual(
            [output.io_type for output in schema.outputs],
            ["MASK", "IMAGE", "INT", "INT", "INT", "INT", "BBOX", "BOUNDING_BOX"],
        )
        self.assertFalse(schema.outputs[0].is_output_list)
        self.assertTrue(schema.outputs[6].is_output_list)
        self.assertTrue(schema.outputs[7].is_output_list)


if __name__ == "__main__":
    unittest.main()
