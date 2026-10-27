import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_nested_inpainting_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.nested_inpainting_preprocessor"]


def run(mask: torch.Tensor, **overrides):
    values = {
        "reorder_first": False,
        "tapered_corners": True,
        "counter_threshold": 0.0,
        "blur_a_max_res": 256,
        "blur_b_max_res": 512,
        "dummy_blur_a": 0.0,
        "dummy_blur_b": 0.0,
        "dummy_blur_c": 0.0,
        "counter_blur_a": 0.0,
        "counter_blur_b": 0.0,
        "counter_blur_c": 0.0,
        "bg_blur_overlap_resistance": 0.0,
    }
    values.update(overrides)
    return MODULE.nested_inpainting_preprocess(mask, **values)


def reference_run(mask: torch.Tensor, **overrides):
    values = {
        "reorder_first": False,
        "tapered_corners": True,
        "counter_threshold": 0.0,
        "blur_a_max_res": 256,
        "blur_b_max_res": 512,
        "dummy_blur_a": 0.0,
        "dummy_blur_b": 0.0,
        "dummy_blur_c": 0.0,
        "counter_blur_a": 0.0,
        "counter_blur_b": 0.0,
        "counter_blur_c": 0.0,
        "bg_blur_overlap_resistance": 0.0,
    }
    values.update(overrides)
    source = MODULE._validate_mask_batch(mask)
    if values["reorder_first"]:
        source = PACKAGE.reorder_batch_mask.reorder_batch_mask(source, "biggest")
    source = MODULE._threshold(source, 0.0)
    overlap_removed = []
    for index in range(source.shape[0] - 1):
        later = PACKAGE.mask_batch_combine.combine_mask_batch(source[index + 1 :])
        overlap_removed.append(MODULE._subtract(source[index : index + 1], later))
    overlap_removed.append(source[-1:])
    source = PACKAGE.reorder_batch_mask.reorder_batch_mask(torch.cat(overlap_removed), "biggest")

    union_init = PACKAGE.mask_batch_combine.combine_mask_batch(source)
    bg_init = 1.0 - union_init
    grown_background = MODULE._grow_one_tapered(bg_init)
    selected = [
        index
        for index in range(source.shape[0])
        if not MODULE._is_empty(MODULE._pixelwise_and(source[index : index + 1], grown_background))
    ]
    counters = [item.unsqueeze(0) for item in source]
    if selected:
        counter_radius = MODULE._blur_for_area(
            bg_init,
            values["counter_blur_a"],
            values["counter_blur_b"],
            values["counter_blur_c"],
            values["blur_a_max_res"],
            values["blur_b_max_res"],
        )
        for index in selected:
            counter = PACKAGE.blur_mask_outward.blur_mask_outward(
                source[index : index + 1], counter_radius, values["tapered_corners"], True
            )
            if index > 0:
                earlier = PACKAGE.mask_fill_holes.fill_mask_holes(source[:index])
                earlier = PACKAGE.mask_batch_combine.combine_mask_batch(earlier)
                resistance = MODULE._inward_blurred_prefix(
                    earlier,
                    values["bg_blur_overlap_resistance"],
                    values["tapered_corners"],
                )
                counter = MODULE._subtract(counter, resistance)
            if index + 1 < source.shape[0]:
                later = PACKAGE.mask_batch_combine.combine_mask_batch(source[index + 1 :])
                counter = MODULE._subtract(counter, later)
            counters[index] = counter
    counters = torch.cat(counters)
    bg_blurred = 1.0 - PACKAGE.mask_batch_combine.combine_mask_batch(counters)

    height, width = source.shape[1:]
    empty = torch.empty((0, height, width), dtype=torch.float32)
    diagnostics = [empty, counters[0:1], empty, [], 0, empty]
    carried = [item.unsqueeze(0) for item in counters]
    outputs = []
    for outer_index in range(source.shape[0] - 1):
        current = carried[0]
        future = carried[1:]
        dummy_radius = MODULE._blur_for_area(
            current,
            values["dummy_blur_a"],
            values["dummy_blur_b"],
            values["dummy_blur_c"],
            values["blur_a_max_res"],
            values["blur_b_max_res"],
        )
        counter_radius = MODULE._blur_for_area(
            current,
            values["counter_blur_a"],
            values["counter_blur_b"],
            values["counter_blur_c"],
            values["blur_a_max_res"],
            values["blur_b_max_res"],
        )
        future_source = source[outer_index + 1 :]
        dummy_batch = PACKAGE.blur_mask_outward.blur_mask_outward(
            future_source, dummy_radius, values["tapered_corners"], True
        )
        thresholded = MODULE._threshold(dummy_batch, 0.0)
        inner = MODULE._subtract(current, PACKAGE.mask_batch_combine.combine_mask_batch(dummy_batch))
        minus_init = MODULE._subtract(current, PACKAGE.mask_batch_combine.combine_mask_batch(future_source))
        probe = MODULE._grow_one_tapered(MODULE._threshold(minus_init, 0.0))
        pixelwise = torch.cat(
            [MODULE._pixelwise_and(thresholded[index : index + 1], probe) for index in range(len(future))]
        )
        matches = [index for index in range(len(future)) if not MODULE._is_empty(pixelwise[index : index + 1])]
        for relative_index in matches:
            future[relative_index] = MODULE._update_future_counter(
                future[relative_index],
                future_source[relative_index : relative_index + 1],
                pixelwise[relative_index : relative_index + 1],
                thresholded[relative_index : relative_index + 1],
                dummy_batch[relative_index : relative_index + 1],
                inner,
                dummy_radius,
                counter_radius,
                values["counter_threshold"],
                values["tapered_corners"],
            )
        if outer_index == 0:
            diagnostics = [dummy_batch, minus_init, pixelwise, matches, len(matches), torch.cat(future)]
        outputs.append(inner)
        carried = future
    outputs.extend(carried)
    result = torch.cat(outputs)
    mask_and_bg = torch.cat((bg_blurred, result))
    all_covered = MODULE._is_empty(
        1.0 - MODULE._add(bg_init, PACKAGE.mask_batch_combine.combine_mask_batch(result))
    )
    return (result, mask_and_bg, union_init, bg_init, bg_blurred, all_covered, *diagnostics)


class NestedInpaintingPreprocessorTests(unittest.TestCase):
    def test_schema_and_registration(self):
        schema = MODULE.NestedInpaintingPreprocessor.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteNestedInpaintingPreprocessor")
        self.assertEqual(schema.display_name, "Nested Inpainting Preprocessor")
        self.assertEqual(schema.category, "Utility Suite/Mask")
        self.assertEqual(
            [item.id for item in schema.inputs],
            [
                "mask",
                "reorder_first",
                "tapered_corners",
                "counter_threshold",
                "blur_a_max_res",
                "blur_b_max_res",
                "dummy_blur_a",
                "dummy_blur_b",
                "dummy_blur_c",
                "counter_blur_a",
                "counter_blur_b",
                "counter_blur_c",
                "bg_blur_overlap_resistance",
            ],
        )
        self.assertEqual(
            [item.id for item in schema.outputs],
            [
                "mask",
                "mask_and_bg",
                "union_init",
                "bg_init",
                "bg_blurred",
                "all_covered",
                "zeroth_dummy",
                "zeroth_minus_init",
                "zeroth_pixelwise",
                "zeroth_match_index",
                "zeroth_match_count",
                "zeroth_counter_mask",
            ],
        )
        self.assertTrue(schema.outputs[9].is_output_list)
        nodes = asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list())
        self.assertIn(MODULE.NestedInpaintingPreprocessor, nodes)

    def test_single_mask_has_explicit_empty_zeroth_diagnostics(self):
        mask = torch.zeros(1, 24, 32)
        mask[0, 6:18, 8:24] = 1

        result = run(mask)

        self.assertEqual(result[0].shape, (1, 24, 32))
        self.assertEqual(result[1].shape, (2, 24, 32))
        self.assertTrue(torch.equal(result[2], mask))
        self.assertTrue(torch.equal(result[3], 1.0 - mask))
        self.assertTrue(result[5])
        self.assertEqual(result[6].shape, (0, 24, 32))
        self.assertTrue(torch.equal(result[7], result[0]))
        self.assertEqual(result[8].shape, (0, 24, 32))
        self.assertEqual(result[9], [])
        self.assertEqual(result[10], 0)
        self.assertEqual(result[11].shape, (0, 24, 32))

    def test_zeroth_outputs_match_first_parent_iteration_boundaries(self):
        masks = torch.zeros(3, 32, 40)
        masks[0, 4:28, 3:19] = 1
        masks[1, 8:25, 15:29] = 1
        masks[2, 12:23, 27:37] = 1

        result = run(masks, dummy_blur_a=1.5, counter_blur_a=1.0)

        final, _, union_init, bg_init, bg_blurred, all_covered = result[:6]
        zeroth_dummy, zeroth_minus, zeroth_pixelwise, indices, count, zeroth_counter = result[6:]
        self.assertEqual(final.shape, masks.shape)
        self.assertEqual(zeroth_dummy.shape, (2, 32, 40))
        self.assertEqual(zeroth_minus.shape, (1, 32, 40))
        self.assertEqual(zeroth_pixelwise.shape, (2, 32, 40))
        self.assertEqual(zeroth_counter.shape, (2, 32, 40))
        self.assertEqual(count, len(indices))
        self.assertEqual(indices, sorted(set(indices)))
        self.assertTrue(torch.equal(bg_init, 1.0 - union_init))
        self.assertTrue(torch.equal(bg_blurred, result[1][0:1]))
        self.assertIsInstance(all_covered, bool)

    def test_reorder_first_changes_overlap_ownership(self):
        masks = torch.zeros(2, 28, 36)
        masks[0, 10:18, 13:23] = 1
        masks[1, 4:24, 6:30] = 1

        without_reorder = run(masks, reorder_first=False)[0]
        with_reorder = run(masks, reorder_first=True)[0]

        self.assertFalse(torch.equal(without_reorder, with_reorder))
        self.assertEqual(without_reorder.shape, with_reorder.shape)

    def test_blur_zero_is_supported_for_every_blur_widget(self):
        masks = torch.zeros(3, 20, 28)
        masks[0, 2:17, 2:12] = 1
        masks[1, 5:18, 10:20] = 1
        masks[2, 8:16, 18:26] = 1

        result = run(masks)

        for output_index in (0, 1, 2, 3, 4, 6, 7, 8, 11):
            self.assertEqual(result[output_index].dtype, torch.float32)
            self.assertEqual(result[output_index].device.type, "cpu")
        self.assertEqual(result[0].shape, masks.shape)

    def test_static_blur_uint8_cache_round_trip_is_exact(self):
        masks = torch.zeros(2, 31, 37)
        masks[0, 4:25, 5:20] = 1
        masks[1, 9:28, 17:33] = 1
        cache = {}

        for tapered in (False, True):
            for radius in (0.0, 1.5, 4.0):
                with self.subTest(tapered=tapered, radius=radius):
                    expected = PACKAGE.blur_mask_outward.blur_mask_outward(masks[1:2], radius, tapered, True)
                    actual = MODULE._cached_dummy_blur(masks, 1, radius, tapered, cache)
                    self.assertTrue(torch.equal(actual, expected))

    def test_optimized_path_matches_literal_topology_reference(self):
        masks = torch.zeros(5, 38, 46)
        masks[0, 3:31, 2:20] = 1
        masks[1, 7:34, 15:31] = 1
        masks[2, 11:29, 27:43] = 1
        masks[3, 2:13, 32:44] = 1
        masks[4, 25:36, 7:24] = 1
        settings = {
            "reorder_first": True,
            "counter_threshold": 0.2,
            "blur_a_max_res": 20,
            "blur_b_max_res": 32,
            "dummy_blur_a": 1.5,
            "dummy_blur_b": 2.0,
            "dummy_blur_c": 2.5,
            "counter_blur_a": 1.0,
            "counter_blur_b": 1.5,
            "counter_blur_c": 2.0,
            "bg_blur_overlap_resistance": 1.25,
        }

        optimized = run(masks, **settings)
        reference = reference_run(masks, **settings)

        for index, (actual, expected) in enumerate(zip(optimized, reference)):
            with self.subTest(output=index):
                if isinstance(actual, torch.Tensor):
                    self.assertTrue(torch.equal(actual, expected))
                else:
                    self.assertEqual(actual, expected)

    def test_rejects_nonbatch_and_empty_masks(self):
        with self.assertRaisesRegex(ValueError, r"\[B,H,W\]"):
            run(torch.zeros(12, 16))
        with self.assertRaisesRegex(ValueError, "nonempty dimensions"):
            run(torch.zeros(0, 12, 16))

    def test_has_no_runtime_dependency_on_prototype_custom_nodes(self):
        source = (PACKAGE_ROOT / "nested_inpainting_preprocessor.py").read_text(encoding="utf-8").lower()
        for dependency in ("easyuse", "impact", "kjnodes", "was-ns"):
            self.assertNotIn(dependency, source)


if __name__ == "__main__":
    unittest.main()
