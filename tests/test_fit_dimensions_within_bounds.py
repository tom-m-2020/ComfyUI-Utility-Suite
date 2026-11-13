from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_fit_dimensions_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.fit_dimensions_within_bounds"]


class FitDimensionsWithinBoundsTests(unittest.TestCase):
    def test_landscape_source_is_limited_by_destination_width(self):
        self.assertEqual(MODULE.fit_dimensions_within_bounds(1024, 1024, 1920, 1080), (1024, 576))

    def test_portrait_source_is_limited_by_destination_height(self):
        self.assertEqual(MODULE.fit_dimensions_within_bounds(1024, 1024, 1080, 1920), (576, 1024))

    def test_source_can_scale_up(self):
        self.assertEqual(MODULE.fit_dimensions_within_bounds(1920, 1080, 640, 480), (1440, 1080))

    def test_matching_aspect_ratio_fills_both_bounds(self):
        self.assertEqual(MODULE.fit_dimensions_within_bounds(1280, 720, 1920, 1080), (1280, 720))

    def test_rounding_matches_math_expression_node(self):
        self.assertEqual(MODULE.fit_dimensions_within_bounds(5, 5, 2, 3), (3, 5))

    def test_execute_returns_width_then_height(self):
        result = MODULE.FitDimensionsWithinBounds.execute(1024, 768, 1920, 1080)
        self.assertEqual(result.result, (1024, 576))

    def test_schema_and_registration(self):
        schema = MODULE.FitDimensionsWithinBounds.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteFitDimensionsWithinBounds")
        self.assertEqual(schema.display_name, "Fit Dimensions Within Bounds")
        self.assertEqual(schema.category, "Utility Suite/Image")
        self.assertEqual(
            [item.id for item in schema.inputs],
            ["destination_width", "destination_height", "source_width", "source_height"],
        )
        self.assertTrue(all(item.io_type == "INT" for item in schema.inputs))
        self.assertEqual([item.id for item in schema.outputs], ["width", "height"])
        self.assertTrue(all(item.io_type == "INT" for item in schema.outputs))
        self.assertIn(
            MODULE.FitDimensionsWithinBounds,
            asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list()),
        )


if __name__ == "__main__":
    unittest.main()
