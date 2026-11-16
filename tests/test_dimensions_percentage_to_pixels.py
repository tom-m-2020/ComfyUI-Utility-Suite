from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_dimensions_percentage_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.dimensions_percentage_to_pixels"]


class DimensionsPercentageToPixelsTests(unittest.TestCase):
    def test_schema_and_registration(self):
        schema = MODULE.DimensionsPercentageToPixels.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteDimensionsPercentageToPixels")
        self.assertEqual(schema.display_name, "Dimensions Percentage to Pixels")
        self.assertEqual([item.id for item in schema.inputs], ["width", "height", "percent", "divisible_by"])
        self.assertEqual([item.min for item in schema.inputs], [0, 0, 0, 1])
        self.assertEqual([item.max for item in schema.inputs], [None, None, None, None])
        self.assertEqual(
            [item.id for item in schema.outputs],
            ["old_width", "old_height", "new_width", "new_height"],
        )
        self.assertIn(
            MODULE.DimensionsPercentageToPixels,
            asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list()),
        )

    def test_matches_prototype_expression(self):
        self.assertEqual(
            MODULE.dimensions_percentage_to_pixels(1024, 768, 20, 8),
            (1024, 768, 208, 152),
        )

    def test_percent_can_exceed_one_hundred(self):
        self.assertEqual(
            MODULE.dimensions_percentage_to_pixels(100, 50, 250, 8),
            (100, 50, 248, 128),
        )

    def test_zero_dimensions_and_percent_return_zero_new_dimensions(self):
        self.assertEqual(
            MODULE.dimensions_percentage_to_pixels(0, 0, 100, 64),
            (0, 0, 0, 0),
        )
        self.assertEqual(
            MODULE.dimensions_percentage_to_pixels(123, 456, 0, 7),
            (123, 456, 0, 0),
        )

    def test_preserves_prototype_half_even_rounding(self):
        self.assertEqual(
            MODULE.dimensions_percentage_to_pixels(4, 12, 100, 8),
            (4, 12, 0, 16),
        )

    def test_rejects_invalid_direct_calls(self):
        invalid = [
            ((True, 1, 100, 1), TypeError, "integers"),
            ((-1, 1, 100, 1), ValueError, "width and height"),
            ((1, -1, 100, 1), ValueError, "width and height"),
            ((1, 1, -1, 1), ValueError, "percent"),
            ((1, 1, 100, 0), ValueError, "divisible_by"),
        ]
        for values, error, message in invalid:
            with self.subTest(values=values), self.assertRaisesRegex(error, message):
                MODULE.dimensions_percentage_to_pixels(*values)


if __name__ == "__main__":
    unittest.main()
