import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_nested_inpainting_simple_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.nested_inpainting_preprocessor_simple"]


def run(mask: torch.Tensor, **overrides):
    values = {
        "reorder_first": False,
        "tapered_corners": True,
        "max_mask_size_a": 256,
        "max_mask_size_b": 512,
        "blur_a": 0.0,
        "blur_b": 0.0,
        "blur_c": 0.0,
    }
    values.update(overrides)
    return MODULE.nested_inpainting_preprocess_simple(mask, **values)


def reference_run(mask: torch.Tensor, **overrides):
    values = {
        "reorder_first": False,
        "tapered_corners": True,
        "max_mask_size_a": 256,
        "max_mask_size_b": 512,
        "blur_a": 0.0,
        "blur_b": 0.0,
        "blur_c": 0.0,
    }
    values.update(overrides)
    source = MODULE._validate_mask_batch(mask)
    if values["reorder_first"]:
        source = PACKAGE.reorder_batch_mask.reorder_batch_mask(source, "biggest")
    source = (source > 0.0).to(torch.float32)

    overlap_removed = []
    for index in range(source.shape[0] - 1):
        later = PACKAGE.mask_batch_combine.combine_mask_batch(source[index + 1 :])
        overlap_removed.append(torch.clamp(source[index : index + 1] - later, 0.0, 1.0))
    overlap_removed.append(source[-1:])
    source = PACKAGE.reorder_batch_mask.reorder_batch_mask(
        torch.cat(overlap_removed), "biggest"
    )

    union_init = PACKAGE.mask_batch_combine.combine_mask_batch(source)
    bg_init = 1.0 - union_init
    blurred = torch.cat(
        [
            PACKAGE.blur_mask_outward.blur_mask_outward(
                item.unsqueeze(0),
                MODULE._blur_for_area(
                    item,
                    values["max_mask_size_a"],
                    values["max_mask_size_b"],
                    values["blur_a"],
                    values["blur_b"],
                    values["blur_c"],
                ),
                values["tapered_corners"],
                True,
            )
            for item in source
        ]
    )
    values_batch = torch.cat((bg_init, blurred))
    thresholded = values_batch > 0.0
    outputs = []
    for index in range(values_batch.shape[0] - 1):
        matches = [
            later
            for later in range(index + 1, values_batch.shape[0])
            if bool(torch.any(thresholded[index] & thresholded[later]).item())
        ]
        if not matches:
            outputs.append(values_batch[index : index + 1])
            continue
        combined = values_batch[matches[0] : matches[0] + 1]
        for later in matches[1:]:
            item = values_batch[later : later + 1]
            combined = torch.clamp(
                torch.clamp(combined - item, 0.0, 1.0) + item, 0.0, 1.0
            )
        outputs.append(
            torch.clamp(values_batch[index : index + 1] - combined, 0.0, 1.0)
        )
    outputs.append(values_batch[-1:])
    result = torch.cat(outputs)
    return result, result[1:], union_init, bg_init, result[0:1]


class NestedInpaintingPreprocessorSimpleTests(unittest.TestCase):
    def test_schema_and_registration(self):
        schema = MODULE.NestedInpaintingPreprocessorSimple.define_schema()
        self.assertEqual(
            schema.node_id, "UtilitySuiteNestedInpaintingPreprocessorSimple"
        )
        self.assertEqual(schema.display_name, "Nested Inpainting Preprocessor Simple")
        self.assertEqual(schema.category, "Utility Suite/Mask")
        self.assertEqual(
            [item.id for item in schema.inputs],
            [
                "mask",
                "reorder_first",
                "tapered_corners",
                "max_mask_size_a",
                "max_mask_size_b",
                "blur_a",
                "blur_b",
                "blur_c",
            ],
        )
        self.assertEqual(
            [item.id for item in schema.outputs],
            [
                "mask",
                "mask_no_bg",
                "union_init",
                "bg_init",
                "bg_blurred",
                "all_covered",
            ],
        )
        nodes = asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list())
        self.assertIn(MODULE.NestedInpaintingPreprocessorSimple, nodes)

    def test_outputs_have_expected_shapes_and_views(self):
        masks = torch.zeros(3, 24, 32)
        masks[0, 3:18, 2:15] = 1
        masks[1, 7:21, 12:25] = 1
        masks[2, 10:20, 23:30] = 1

        result, mask_no_bg, union_init, bg_init, bg_blurred, all_covered = run(
            masks, blur_a=2.0, blur_b=2.0, blur_c=2.0
        )

        self.assertEqual(result.shape, (4, 24, 32))
        self.assertEqual(mask_no_bg.shape, masks.shape)
        self.assertEqual(union_init.shape, (1, 24, 32))
        self.assertEqual(bg_init.shape, (1, 24, 32))
        self.assertEqual(bg_blurred.shape, (1, 24, 32))
        self.assertEqual(
            result.untyped_storage().data_ptr(), mask_no_bg.untyped_storage().data_ptr()
        )
        self.assertEqual(
            result.untyped_storage().data_ptr(), bg_blurred.untyped_storage().data_ptr()
        )
        self.assertTrue(torch.equal(bg_init, 1.0 - union_init))
        self.assertIsInstance(all_covered, bool)

    def test_optimized_path_matches_literal_topology_reference(self):
        masks = torch.zeros(5, 38, 46)
        masks[0, 3:31, 2:20] = 1
        masks[1, 7:34, 15:31] = 1
        masks[2, 11:29, 27:43] = 1
        masks[3, 2:13, 32:44] = 1
        masks[4, 25:36, 7:24] = 1

        settings = {
            "reorder_first": True,
            "max_mask_size_a": 12,
            "max_mask_size_b": 20,
            "blur_a": 1.0,
            "blur_b": 2.0,
            "blur_c": 3.0,
        }
        actual = run(masks, **settings)
        expected = reference_run(masks, **settings)

        for index in range(5):
            with self.subTest(output=index):
                self.assertTrue(
                    torch.allclose(actual[index], expected[index], atol=1e-7, rtol=0.0)
                )
        self.assertTrue(actual[5])

    def test_coverage_tolerates_only_float_noise(self):
        bg_init = torch.zeros(1, 3, 4)
        near_full = torch.full((1, 3, 4), 1.0 - 2.0**-24)
        visible_deficit = torch.full((1, 3, 4), 1.0 - 1e-4)

        self.assertTrue(MODULE._coverage_is_complete(bg_init, near_full))
        self.assertFalse(MODULE._coverage_is_complete(bg_init, visible_deficit))

    def test_blur_radius_uses_previous_node_bbox_area_criteria(self):
        small = torch.zeros(20, 20)
        medium = torch.zeros(20, 20)
        large = torch.zeros(20, 20)
        small[1:5, 1:5] = 1
        medium[1:8, 1:8] = 1
        large[1:13, 1:13] = 1

        settings = (5, 10, 1.0, 2.0, 3.0)
        self.assertEqual(MODULE._blur_for_area(small, *settings), 1.0)
        self.assertEqual(MODULE._blur_for_area(medium, *settings), 2.0)
        self.assertEqual(MODULE._blur_for_area(large, *settings), 3.0)

    def test_grouped_adaptive_blur_matches_individual_blurs(self):
        masks = torch.zeros(3, 32, 40)
        masks[0, 2:7, 2:7] = 1
        masks[1, 2:14, 10:22] = 1
        masks[2, 2:28, 14:38] = 1

        actual = MODULE._blur_by_area(masks, 8, 20, 1.0, 2.0, 3.0, True)
        expected = torch.cat(
            [
                PACKAGE.blur_mask_outward.blur_mask_outward(
                    masks[index : index + 1], radius, True, True
                )
                for index, radius in enumerate((1.0, 2.0, 3.0))
            ]
        )

        self.assertTrue(torch.equal(actual, expected))

    def test_singleton_and_zero_blur_are_supported(self):
        mask = torch.zeros(1, 20, 28)
        mask[0, 4:16, 7:21] = 1

        result = run(mask)

        self.assertEqual(result[0].shape, (2, 20, 28))
        self.assertEqual(result[1].shape, mask.shape)
        self.assertTrue(torch.equal(result[2], mask))
        self.assertTrue(torch.equal(result[3], 1.0 - mask))
        self.assertTrue(result[5])

    def test_reorder_first_changes_overlap_ownership(self):
        masks = torch.zeros(2, 28, 36)
        masks[0, 10:18, 13:23] = 1
        masks[1, 4:24, 6:30] = 1

        without_reorder = run(masks, reorder_first=False)[1]
        with_reorder = run(masks, reorder_first=True)[1]

        self.assertFalse(torch.equal(without_reorder, with_reorder))

    def test_rejects_nonbatch_and_empty_masks(self):
        with self.assertRaisesRegex(ValueError, r"\[B,H,W\]"):
            run(torch.zeros(12, 16))
        with self.assertRaisesRegex(ValueError, "nonempty dimensions"):
            run(torch.zeros(0, 12, 16))

    def test_has_no_runtime_dependency_on_prototype_custom_nodes(self):
        source = (
            (PACKAGE_ROOT / "nested_inpainting_preprocessor_simple.py")
            .read_text(encoding="utf-8")
            .lower()
        )
        for dependency in ("easyuse", "impact", "kjnodes", "was-ns"):
            self.assertNotIn(dependency, source)


if __name__ == "__main__":
    unittest.main()
