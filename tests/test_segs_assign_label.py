from __future__ import annotations

import ast
import asyncio
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
    "utility_suite_segs_assign_label_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.segs_assign_label"]
IMPACT_SOURCE = (
    COMFY_ROOT / "custom_nodes" / "ComfyUI-Impact-Pack" / "modules" / "impact" / "segs_nodes.py"
)

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
)


def impact_assign_class():
    tree = ast.parse(IMPACT_SOURCE.read_text(encoding="utf-8"), filename=str(IMPACT_SOURCE))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SEGSLabelAssign")
    namespace = {"logging": logging}
    exec(compile(ast.Module(body=[class_node], type_ignores=[]), str(IMPACT_SOURCE), "exec"), namespace)  # noqa: S102
    return namespace["SEGSLabelAssign"]


def segment(label: str) -> SEG:
    return SEG(
        object(),
        object(),
        0.75,
        (1, 2, 7, 9),
        (2, 3, 6, 8),
        label,
        object(),
    )


class SEGSAssignLabelTests(unittest.TestCase):
    def setUp(self):
        self.header = (100, 200)
        self.entries = [segment(label) for label in "ABCDE"]
        self.segs = (self.header, self.entries)

    def execute(self, labels, index=None, index_text=""):
        return MODULE.SEGSAssignLabel.execute(
            self.segs,
            labels,
            index=index,
            index_text=index_text,
        ).result[0]

    def test_no_index_source_matches_impact_all_segs_behavior(self):
        labels = " first,second , third,fourth,fifth "
        expected = impact_assign_class()().doit(self.segs, labels)[0]
        actual = self.execute(labels)
        self.assertIs(actual[0], expected[0])
        self.assertEqual([entry.label for entry in actual[1]], [entry.label for entry in expected[1]])
        for actual_entry, expected_entry in zip(actual[1], expected[1], strict=True):
            for field in SEG._fields:
                if field != "label":
                    self.assertIs(getattr(actual_entry, field), getattr(expected_entry, field))

    def test_connected_indices_assign_in_supplied_order(self):
        result = self.execute("D replacement,B replacement", index=[3, 1])
        self.assertEqual(
            [entry.label for entry in result[1]],
            ["A", "B replacement", "C", "D replacement", "E"],
        )
        self.assertIs(result[1][0], self.entries[0])
        self.assertIs(result[1][2], self.entries[2])
        self.assertIs(result[1][4], self.entries[4])

    def test_nonempty_manual_indices_override_connected_input(self):
        result = self.execute("B replacement,D replacement", index=[0, 2], index_text=" 1, 3 ")
        self.assertEqual(
            [entry.label for entry in result[1]],
            ["A", "B replacement", "C", "D replacement", "E"],
        )

    def test_empty_or_whitespace_manual_indices_use_connected_input(self):
        for text in ("", "   "):
            with self.subTest(text=text):
                result = self.execute("B replacement", index=[1], index_text=text)
                self.assertEqual([entry.label for entry in result[1]], ["A", "B replacement", "C", "D", "E"])

    def test_invalid_nonempty_manual_indices_never_fall_back(self):
        for text, message in (
            ("1,x", "integers"),
            ("1,,3", "empty element"),
            (",1", "empty element"),
            ("1,", "empty element"),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                self.execute("valid connected", index=[0], index_text=text)

    def test_invalid_manual_index_values_never_fall_back(self):
        for text, message in (("-1", "outside"), ("5", "outside"), ("1,1", "duplicate")):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                self.execute("valid connected", index=[0], index_text=text)

    def test_indices_are_validated(self):
        cases = (
            ([-1], ValueError, "outside"),
            ([5], ValueError, "outside"),
            ([1, 1], ValueError, "duplicate"),
            ([True], TypeError, "integer"),
            ([1.0], TypeError, "integer"),
            ((1,), TypeError, "list"),
        )
        for index, error_type, message in cases:
            with self.subTest(index=index), self.assertRaisesRegex(error_type, message):
                self.execute("label", index=index)

    def test_fewer_labels_leave_remaining_selected_segs_unchanged(self):
        with self.assertLogs(level="WARNING") as messages:
            result = self.execute("only first", index=[1, 3])
        self.assertIn("length of labels (1) != length of selected segs (2)", messages.output[0])
        self.assertEqual([entry.label for entry in result[1]], ["A", "only first", "C", "D", "E"])
        self.assertIs(result[1][3], self.entries[3])

    def test_extra_labels_are_ignored_like_impact(self):
        with self.assertLogs(level="WARNING"):
            result = self.execute("replacement,ignored", index=[2])
        self.assertEqual([entry.label for entry in result[1]], ["A", "B", "replacement", "D", "E"])

    def test_selected_replacement_preserves_all_nonlabel_metadata(self):
        result = self.execute("replacement", index=[2])
        replacement = result[1][2]
        self.assertIsNot(replacement, self.entries[2])
        self.assertEqual(replacement.label, "replacement")
        for field in SEG._fields:
            if field != "label":
                self.assertIs(getattr(replacement, field), getattr(self.entries[2], field))
        self.assertEqual([entry.label for entry in self.entries], list("ABCDE"))

    def test_empty_segs_without_indices_returns_valid_empty_segs(self):
        empty = (self.header, [])
        with self.assertLogs(level="WARNING"):
            result = MODULE.SEGSAssignLabel.execute(empty, "unused", index=None, index_text="").result[0]
        self.assertIs(result[0], self.header)
        self.assertEqual(result[1], [])

    def test_schema_and_registration(self):
        schema = MODULE.SEGSAssignLabel.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteSEGSAssignLabel")
        self.assertEqual(schema.display_name, "SEGS Assign (label)")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual([entry.io_type for entry in schema.inputs], ["SEGS", "STRING", "SEGS_INDICES", "STRING"])
        self.assertEqual([entry.id for entry in schema.inputs], ["segs", "labels", "index", "index_text"])
        self.assertTrue(schema.inputs[1].multiline)
        self.assertTrue(schema.inputs[2].optional)
        self.assertEqual(schema.inputs[2].display_name, "index")
        self.assertEqual(schema.inputs[3].display_name, "index")
        self.assertEqual(schema.inputs[3].default, "")
        self.assertFalse(schema.inputs[3].multiline)
        self.assertFalse(schema.is_input_list)
        self.assertEqual([output.io_type for output in schema.outputs], ["SEGS"])
        self.assertIn(MODULE.SEGSAssignLabel, asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list()))


if __name__ == "__main__":
    unittest.main()
