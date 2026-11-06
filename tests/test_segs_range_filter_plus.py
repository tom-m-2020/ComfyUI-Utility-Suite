from __future__ import annotations

import ast
import importlib.util
import logging
import sys
import unittest
from collections import namedtuple
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_segs_range_filter_plus_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.segs_range_filter_plus"]

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)

TARGET_RANGES = {
    "area(=w*h)": (20, 35),
    "width": (3, 5),
    "height": (4, 6),
    "x1": (3, 8),
    "y1": (2, 9),
    "x2": (8, 14),
    "y2": (7, 12),
    "length_percent": (120, 250),
    "confidence(0-100)": (20, 80),
}


def segment(crop_region, confidence, label):
    return SEG(object(), object(), confidence, crop_region, crop_region, label, object())


def load_installed_impact_filter():
    source_path = COMFY_ROOT / "custom_nodes" / "ComfyUI-Impact-Pack" / "modules" / "impact" / "segs_nodes.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SEGSRangeFilter")
    namespace = {"logging": logging, "sys": sys}
    exec(  # noqa: S102
        compile(ast.Module(body=[class_node], type_ignores=[]), str(source_path), "exec"), namespace
    )
    return namespace["SEGSRangeFilter"]()


class SEGSRangeFilterPlusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = load_installed_impact_filter()

    def setUp(self):
        self.header = (64, 96)
        self.entries = [
            segment((0, 1, 2, 9), 0.1, "a"),
            segment((4, 3, 10, 7), 0.5, "b"),
            segment((8, 10, 13, 15), 0.9, "c"),
            segment((15, 2, 24, 5), 0.3, "d"),
        ]
        self.segs = (self.header, self.entries)

    def test_all_targets_and_modes_match_installed_impact_filter(self):
        for target, (minimum, maximum) in TARGET_RANGES.items():
            for mode in (True, False):
                with self.subTest(target=target, mode=mode):
                    expected_filtered, expected_remained = self.upstream.doit(
                        self.segs, target, mode, minimum, maximum
                    )
                    filtered, remained, indices = MODULE.filter_segs_range_plus(
                        self.segs, target, mode, minimum, maximum
                    )
                    self.assertEqual(filtered, expected_filtered)
                    self.assertEqual(remained, expected_remained)
                    self.assertEqual([self.entries[index] for index in indices], filtered[1])

    def test_noncontiguous_indices_correspond_to_filtered_order(self):
        filtered, remained, indices = MODULE.filter_segs_range_plus(
            self.segs, "confidence(0-100)", True, 20, 80
        )
        self.assertEqual(indices, [1, 3])
        self.assertEqual(filtered[1], [self.entries[1], self.entries[3]])
        self.assertEqual(remained[1], [self.entries[0], self.entries[2]])

    def test_all_selected_none_selected_and_empty_input(self):
        filtered, remained, indices = MODULE.filter_segs_range_plus(
            self.segs, "x1", True, 0, sys.maxsize
        )
        self.assertEqual(filtered[1], self.entries)
        self.assertEqual(remained[1], [])
        self.assertEqual(indices, [0, 1, 2, 3])

        filtered, remained, indices = MODULE.filter_segs_range_plus(self.segs, "x1", True, 100, 200)
        self.assertEqual(filtered[1], [])
        self.assertEqual(remained[1], self.entries)
        self.assertEqual(indices, [])

        empty = (self.header, [])
        self.assertEqual(
            MODULE.filter_segs_range_plus(empty, "width", True, 0, 10),
            ((self.header, []), (self.header, []), []),
        )

    def test_headers_order_and_segment_identity_are_preserved(self):
        filtered, remained, _ = MODULE.filter_segs_range_plus(
            self.segs, "confidence(0-100)", True, 20, 80
        )
        self.assertIs(filtered[0], self.header)
        self.assertIs(remained[0], self.header)
        self.assertIs(filtered[1][0], self.entries[1])
        self.assertIs(filtered[1][1], self.entries[3])
        self.assertIs(remained[1][0], self.entries[0])
        self.assertIs(remained[1][1], self.entries[2])

    def test_schema_matches_upstream_widgets_and_index_is_one_value(self):
        schema = MODULE.SEGSRangeFilterPlus.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteSEGSRangeFilterPlus")
        self.assertEqual(schema.display_name, "SEGS Filter (range) Plus")
        self.assertEqual([entry.id for entry in schema.inputs], ["segs", "target", "mode", "min_value", "max_value"])
        self.assertEqual(schema.inputs[1].options, MODULE.TARGETS)
        self.assertTrue(schema.inputs[2].default)
        self.assertEqual(schema.inputs[2].label_on, "inside")
        self.assertEqual(schema.inputs[2].label_off, "outside")
        self.assertEqual((schema.inputs[3].default, schema.inputs[3].min, schema.inputs[3].max, schema.inputs[3].step), (0, 0, sys.maxsize, 1))
        self.assertEqual((schema.inputs[4].default, schema.inputs[4].min, schema.inputs[4].max, schema.inputs[4].step), (67108864, 0, sys.maxsize, 1))
        self.assertEqual([output.id for output in schema.outputs], ["filtered_SEGS", "remained_SEGS", "index"])
        self.assertEqual([output.io_type for output in schema.outputs], ["SEGS", "SEGS", "SEGS_INDICES"])
        self.assertTrue(all(not output.is_output_list for output in schema.outputs))


if __name__ == "__main__":
    unittest.main()
