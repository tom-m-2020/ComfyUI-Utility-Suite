import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_batch_mask_segs_comparison_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.batch_mask_to_segs_comparison"]


def run(mask, mask_base=None, **overrides):
    values = {
        "crop_factor": 2.0,
        "bbox_fill": False,
        "contour_fill": False,
        "start_comparison": 0,
        "mask_base": mask_base,
    }
    values.update(overrides)
    return MODULE.batch_mask_to_segs_with_comparison(mask, **values)


class BatchMaskToSEGSWithComparisonTests(unittest.TestCase):
    def test_schema_and_registration(self):
        schema = MODULE.BatchMaskToSEGSWithComparison.define_schema()

        self.assertEqual(schema.node_id, "UtilitySuiteBatchMaskToSEGSWithComparison")
        self.assertEqual(schema.display_name, "Batch MASK to SEGS with Comparison")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual(
            [item.id for item in schema.inputs],
            [
                "mask",
                "mask_base",
                "crop_factor",
                "bbox_fill",
                "contour_fill",
                "start_comparison",
            ],
        )
        self.assertTrue(schema.inputs[1].optional)
        self.assertEqual(schema.inputs[5].min, -99)
        self.assertEqual(schema.inputs[5].max, 99)
        nodes = asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list())
        self.assertIn(MODULE.BatchMaskToSEGSWithComparison, nodes)

    def test_without_base_applies_crop_factor_to_target(self):
        mask = torch.zeros(1, 40, 60)
        mask[0, 14:20, 20:30] = 1

        header, segments = run(mask)

        self.assertEqual(header, (40, 60))
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].bbox, (20, 14, 30, 20))
        self.assertEqual(segments[0].crop_region, [15, 11, 35, 23])
        self.assertEqual(segments[0].cropped_mask.shape, (12, 20))

    def test_comparison_takes_per_axis_maximum(self):
        mask = torch.zeros(1, 50, 70)
        mask[0, 20:25, 20:40] = 1
        base = torch.zeros(1, 50, 70)
        base[0, 3:9, 4:8] = 1

        _, segments = run(mask, base)

        self.assertEqual(segments[0].crop_region, [20, 16, 40, 28])
        self.assertEqual(segments[0].cropped_mask.shape, (12, 20))

    def test_positive_and_negative_comparison_offsets(self):
        masks = torch.zeros(3, 80, 100)
        for index in range(3):
            masks[index, 30:40, 40:50] = 1
        bases = torch.zeros(4, 80, 100)
        bases[0, 2:8, 2:8] = 1
        bases[1, 2:12, 2:12] = 1
        bases[2, 2:16, 2:16] = 1
        bases[3, 2:20, 2:20] = 1

        _, positive = run(masks, bases[:3], start_comparison=1)
        _, negative = run(masks, bases, start_comparison=-1)

        self.assertEqual(
            [(s.crop_region[2] - s.crop_region[0]) for s in positive],
            [10, 12, 20],
        )
        self.assertEqual(
            [(s.crop_region[2] - s.crop_region[0]) for s in negative],
            [20, 28, 10],
        )

    def test_extra_bases_are_ignored_and_missing_base_uses_tight_target(self):
        masks = torch.zeros(2, 50, 60)
        masks[:, 20:26, 20:28] = 1
        bases = torch.zeros(3, 50, 60)
        bases[:, 2:12, 2:12] = 1

        _, segments = run(masks, bases, start_comparison=-1)

        self.assertEqual(
            [(s.crop_region[2] - s.crop_region[0]) for s in segments],
            [20, 8],
        )
        _, missing = run(masks, bases[:1], start_comparison=-1)
        self.assertEqual(
            [(s.crop_region[2] - s.crop_region[0]) for s in missing],
            [8, 8],
        )

    def test_crop_shifts_at_canvas_boundary_like_impact(self):
        mask = torch.zeros(1, 30, 40)
        mask[0, 0:10, 0:10] = 1

        _, segments = run(mask, crop_factor=3.0)

        self.assertEqual(segments[0].crop_region, [0, 0, 30, 30])

    def test_one_seg_per_batch_item_including_disconnected_and_empty(self):
        masks = torch.zeros(2, 24, 32)
        masks[0, 2:7, 3:8] = 1
        masks[0, 15:21, 22:29] = 1

        header, segments = run(masks)

        self.assertEqual(header, (24, 32))
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].bbox, (3, 2, 29, 21))
        self.assertEqual(segments[1].bbox, (0, 0, 0, 0))
        self.assertEqual(segments[1].crop_region, [0, 0, 1, 1])
        self.assertEqual(segments[1].cropped_mask.shape, (1, 1))

    def test_bbox_and_contour_fill_modify_cropped_mask_only(self):
        mask = torch.zeros(1, 24, 30)
        mask[0, 3:20, 4:24] = 0.5
        mask[0, 8:15, 10:18] = 0

        _, preserved = run(mask)
        _, contour = run(mask, contour_fill=True)
        _, bbox = run(mask, bbox_fill=True)

        target_y = 10 - preserved[0].crop_region[1]
        target_x = 12 - preserved[0].crop_region[0]
        self.assertEqual(preserved[0].cropped_mask[target_y, target_x], 0)
        self.assertEqual(contour[0].cropped_mask[target_y, target_x], 1)
        self.assertEqual(bbox[0].cropped_mask[target_y, target_x], 1)
        self.assertAlmostEqual(float(preserved[0].cropped_mask.max()), 0.5)

    def test_rejects_incompatible_spatial_dimensions(self):
        with self.assertRaisesRegex(ValueError, "share spatial dimensions"):
            run(torch.zeros(1, 20, 30), torch.zeros(1, 10, 15))

    def test_seg_is_impact_compatible_shape(self):
        mask = torch.zeros(1, 12, 16)
        mask[0, 2:10, 4:12] = 1

        _, segments = run(mask)

        self.assertEqual(
            segments[0]._fields,
            (
                "cropped_image",
                "cropped_mask",
                "confidence",
                "crop_region",
                "bbox",
                "label",
                "control_net_wrapper",
            ),
        )
        self.assertIsInstance(segments[0].cropped_mask, np.ndarray)


if __name__ == "__main__":
    unittest.main()
