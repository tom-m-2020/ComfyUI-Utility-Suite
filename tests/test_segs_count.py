from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_segs_count_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.segs_count"]


class SEGSCountTests(unittest.TestCase):
    def test_schema_and_registration(self):
        schema = MODULE.SEGSCount.define_schema()
        self.assertEqual(schema.display_name, "SEGS Count")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual([item.id for item in schema.inputs], ["segs"])
        self.assertEqual([item.id for item in schema.outputs], ["count"])
        nodes = asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list())
        self.assertIn(MODULE.SEGSCount, nodes)

    def test_counts_entries(self):
        entries = [object(), object(), object()]
        self.assertEqual(MODULE.segs_count(((64, 96), entries)), 3)

    def test_empty_segs_returns_zero(self):
        self.assertEqual(MODULE.segs_count(((64, 96), [])), 0)

    def test_does_not_modify_segs_or_entries(self):
        entries = [object(), object()]
        segs = ((64, 96), entries)
        original = list(entries)

        self.assertEqual(MODULE.segs_count(segs), 2)
        self.assertIs(segs[1], entries)
        self.assertEqual(entries, original)

    def test_rejects_invalid_segs_structure(self):
        for value in (None, (), ((64, 96),), ((64, 96), [], "extra")):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "Impact-compatible"):
                MODULE.segs_count(value)
        with self.assertRaisesRegex(TypeError, "entries to be a list"):
            MODULE.segs_count(((64, 96), (object(),)))


if __name__ == "__main__":
    unittest.main()
