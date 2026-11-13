from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_coordinate_percentage_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.get_coordinate_by_percentage"]


class GetCoordinateByPercentageTests(unittest.TestCase):
    def test_leading_center_and_trailing_positions(self):
        function = MODULE.get_coordinate_by_percentage
        self.assertEqual(function(1000, 800, 200, 100, 0, 0), (0, 0))
        self.assertEqual(function(1000, 800, 200, 100, 50, 50), (400, 350))
        self.assertEqual(function(1000, 800, 200, 100, 100, 100), (800, 700))

    def test_source_larger_than_destination_produces_negative_available_space(self):
        self.assertEqual(
            MODULE.get_coordinate_by_percentage(800, 600, 1000, 900, 25, 75),
            (-50, -225),
        )

    def test_rounding_matches_math_expression_node(self):
        self.assertEqual(MODULE.get_coordinate_by_percentage(6, 6, 1, 1, 50, 50), (2, 2))
        self.assertEqual(MODULE.get_coordinate_by_percentage(1, 1, 6, 6, 50, 50), (-2, -2))

    def test_execute_returns_x_then_y(self):
        result = MODULE.GetCoordinateByPercentage.execute(1920, 1080, 640, 480, 25, 75)
        self.assertEqual(result.result, (320, 450))

    def test_schema_and_registration(self):
        schema = MODULE.GetCoordinateByPercentage.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteGetCoordinateByPercentage")
        self.assertEqual(schema.display_name, "Get Coordinate by Percentage")
        self.assertEqual(schema.category, "Utility Suite/Image")
        self.assertEqual(
            [item.id for item in schema.inputs],
            [
                "destination_width",
                "destination_height",
                "source_width",
                "source_height",
                "x_percent",
                "y_percent",
            ],
        )
        self.assertEqual([item.id for item in schema.outputs], ["x", "y"])
        self.assertTrue(all(item.io_type == "INT" for item in (*schema.inputs, *schema.outputs)))
        self.assertEqual((schema.inputs[4].min, schema.inputs[4].max), (0, 100))
        self.assertEqual((schema.inputs[5].min, schema.inputs[5].max), (0, 100))
        self.assertIn(
            MODULE.GetCoordinateByPercentage,
            asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list()),
        )


if __name__ == "__main__":
    unittest.main()
