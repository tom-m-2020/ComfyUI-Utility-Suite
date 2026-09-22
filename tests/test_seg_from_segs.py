from __future__ import annotations

import ast
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
    "utility_suite_seg_from_segs_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.seg_from_segs"]
MASK_TILE = sys.modules[f"{SPEC.name}.mask_tile_segs"]
IMPACT_ROOT = COMFY_ROOT / "custom_nodes" / "ComfyUI-Impact-Pack" / "modules" / "impact"


def impact_class(filename, name):
    source_path = IMPACT_ROOT / filename
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    namespace = {"any_typ": "*"}
    # Execute only the installed Impact node class, avoiding its package-wide optional imports.
    exec(compile(ast.Module(body=[class_node], type_ignores=[]), str(source_path), "exec"), namespace)  # noqa: S102
    return namespace[name]

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)


def segment(index, image=...):
    x1 = index * 10
    cropped_image = object() if image is ... else image
    return SEG(
        cropped_image,
        np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10),
        0.25 + index,
        (x1, 0, x1 + 10, 10),
        (x1 + 1, 1, x1 + 9, 9),
        f"label-{index}",
        object(),
    )


class SEGFromSEGSTests(unittest.TestCase):
    def setUp(self):
        self.header = (10, 50)
        self.entries = [segment(index) for index in range(5)]
        self.segs = (self.header, self.entries)

    def test_basic_selection_and_zero_based_index(self):
        result = MODULE.seg_from_segs(self.segs, 1, 2)
        self.assertEqual(len(result[1]), 2)
        self.assertIs(result[1][0], self.entries[1])
        self.assertIs(result[1][1], self.entries[2])
        first = MODULE.seg_from_segs(self.segs, 0, 1)
        self.assertIs(first[1][0], self.entries[0])

    def test_final_item_and_overlong_length(self):
        final = MODULE.seg_from_segs(self.segs, 4, 1)
        self.assertEqual(len(final[1]), 1)
        self.assertIs(final[1][0], self.entries[4])
        overlong = MODULE.seg_from_segs(self.segs, 3, 99)
        self.assertEqual(len(overlong[1]), 2)
        self.assertIs(overlong[1][0], self.entries[3])
        self.assertIs(overlong[1][1], self.entries[4])

    def test_index_past_end_clamps_to_final_item_like_essentials(self):
        result = MODULE.seg_from_segs(self.segs, 999, 3)
        self.assertEqual(len(result[1]), 1)
        self.assertIs(result[1][0], self.entries[-1])

    def test_impact_decompose_select_assemble_equivalence(self):
        decompose = impact_class("segs_nodes.py", "DecomposeSEGS")()
        select = impact_class("util_nodes.py", "NthItemOfAnyList")()
        assemble = impact_class("segs_nodes.py", "AssembleSEGS")()
        image_b = torch.rand((1, 10, 10, 3))
        image_c = torch.rand((1, 10, 10, 3))
        mask_a = np.eye(10, dtype=np.float32)
        mask_b = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
        mask_c = np.linspace(1, 0, 100, dtype=np.float32).reshape(10, 10)
        entries = [
            SEG(None, mask_a, 0.1, (0, 0, 10, 10), (2, 2, 8, 8), "A", object()),
            SEG(image_b, mask_b, 0.5, (10, 0, 20, 10), (11, 1, 19, 9), "B", object()),
            SEG(image_c, mask_c, 0.9, (20, 0, 30, 10), (21, 1, 29, 9), "C", object()),
        ]
        header = (1024, 2048)
        original = (header, entries)
        impact_header, impact_entries = decompose.doit(original)
        for index in range(3):
            selected = select.doit(impact_entries, [index])[0]
            reference = assemble.doit([impact_header], [selected])[0]
            actual = MODULE.seg_from_segs(original, index, 1)
            self.assertIs(actual[0], reference[0])
            self.assertEqual(len(actual[1]), 1)
            self.assertIs(actual[1][0], reference[1][0])
            for field in SEG._fields:
                self.assertIs(getattr(actual[1][0], field), getattr(reference[1][0], field))
        self.assertIsNone(MODULE.seg_from_segs(original, 0, 1)[1][0].cropped_image)
        self.assertIs(MODULE.seg_from_segs(original, 1, 1)[1][0].cropped_image, image_b)
        self.assertIs(MODULE.seg_from_segs(original, 1, 1)[1][0].cropped_mask, mask_b)
        extended = MODULE.seg_from_segs(original, 1, 2)
        self.assertIs(extended[1][0], entries[1])
        self.assertIs(extended[1][1], entries[2])

    def test_impact_out_of_range_selects_last_and_empty_reference_errors(self):
        select = impact_class("util_nodes.py", "NthItemOfAnyList")()
        self.assertIs(select.doit(self.entries, [999])[0], self.entries[-1])
        self.assertIs(MODULE.seg_from_segs(self.segs, 999, 1)[1][0], self.entries[-1])
        with self.assertRaises(IndexError):
            select.doit([], [0])

    def test_header_and_selected_object_identity_are_preserved(self):
        result = MODULE.seg_from_segs(self.segs, 2, 2)
        self.assertIs(result[0], self.header)
        self.assertIs(result[1][0], self.entries[2])
        self.assertIs(result[1][1], self.entries[3])
        self.assertIsNot(result[1], self.entries)

    def test_cropped_image_none_and_object_references_are_preserved(self):
        image_object = object()
        entries = [segment(0, image=image_object), segment(1, image=None)]
        result = MODULE.seg_from_segs(((10, 20), entries), 0, 2)
        self.assertIs(result[1][0].cropped_image, image_object)
        self.assertIsNone(result[1][1].cropped_image)

    def test_mask_and_all_other_fields_are_untouched(self):
        selected = self.entries[2]
        original_mask = selected.cropped_mask.copy()
        result = MODULE.seg_from_segs(self.segs, 2, 1)[1][0]
        self.assertIs(result, selected)
        self.assertIs(result.cropped_mask, selected.cropped_mask)
        np.testing.assert_array_equal(result.cropped_mask, original_mask)
        self.assertEqual(result.confidence, selected.confidence)
        self.assertIs(result.crop_region, selected.crop_region)
        self.assertIs(result.bbox, selected.bbox)
        self.assertIs(result.label, selected.label)
        self.assertIs(result.control_net_wrapper, selected.control_net_wrapper)

    def test_current_nonspatial_order_is_selected_without_sorting(self):
        reordered = [self.entries[0], self.entries[3], self.entries[2], self.entries[1]]
        result = MODULE.seg_from_segs((self.header, reordered), 1, 2)
        self.assertIs(result[1][0], self.entries[3])
        self.assertIs(result[1][1], self.entries[2])

    def test_input_container_and_entries_are_not_mutated(self):
        before = list(self.entries)
        result = MODULE.seg_from_segs(self.segs, 1, 3)
        self.assertEqual(len(result[1]), 3)
        self.assertEqual(len(self.entries), 5)
        self.assertTrue(all(current is original for current, original in zip(self.entries, before)))

    def test_empty_collection_preserves_header_and_returns_empty_list(self):
        header = [20, 30]
        result = MODULE.seg_from_segs((header, []), 100, 5)
        self.assertIs(result[0], header)
        self.assertEqual(result[1], [])

    def test_unselected_malformed_geometry_is_not_inspected(self):
        malformed = SEG(None, np.zeros((1, 1)), 0.2, None, None, "unselected", None)
        entries = [self.entries[0], malformed]
        result = MODULE.seg_from_segs((self.header, entries), 0, 1)
        self.assertIs(result[1][0], self.entries[0])
        self.assertIsNone(entries[1].crop_region)

    def test_invalid_range_and_structure_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            MODULE.seg_from_segs(self.segs, -1, 1)
        with self.assertRaisesRegex(ValueError, "at least one"):
            MODULE.seg_from_segs(self.segs, 0, 0)
        with self.assertRaisesRegex(TypeError, "expects SEGS"):
            MODULE.seg_from_segs([], 0, 1)

    def test_mask_to_tile_segs_interoperability_preserves_exact_middle_range(self):
        mask = torch.linspace(0, 1, 12 * 30, dtype=torch.float32).reshape(1, 12, 30)
        original, _ = MASK_TILE.mask_to_tile_segs(mask, "uniform_grid", 14, 12, 0, 0, 3, 1, 2, 0, 3, 1)
        result = MODULE.seg_from_segs(original, 1, 2)
        self.assertIs(result[0], original[0])
        self.assertEqual(len(result[1]), 2)
        self.assertIs(result[1][0], original[1][1])
        self.assertIs(result[1][1], original[1][2])
        self.assertIs(result[1][0].cropped_mask, original[1][1].cropped_mask)

    def test_schema_is_one_ordinary_segs_output(self):
        schema = MODULE.SEGFromSEGS.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteSEGFromSEGS")
        self.assertEqual(schema.display_name, "SEG From SEGS")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual([entry.io_type for entry in schema.inputs], ["SEGS", "INT", "INT"])
        self.assertEqual([output.io_type for output in schema.outputs], ["SEGS"])
        self.assertFalse(schema.outputs[0].is_output_list)


if __name__ == "__main__":
    unittest.main()
