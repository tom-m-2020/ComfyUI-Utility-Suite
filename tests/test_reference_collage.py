from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_reference_collage_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.reference_collage"]


def source(index: int, width: int, height: int, weight: float = 1.0):
    return MODULE.CollageSource(index, width, height, weight)


class ReferenceCollageTests(unittest.TestCase):
    def test_schema_and_registration(self):
        schema = MODULE.ReferenceCollage.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteReferenceCollage")
        self.assertEqual(schema.display_name, "Reference Collage")
        self.assertEqual(
            [item.id for item in schema.inputs],
            ["images", "layout", "columns", "weights", "gap", "resize_mode", "alignment", "background", "frame_gap"],
        )
        self.assertFalse(schema.inputs[-1].default)
        self.assertEqual(schema.inputs[0].template.prefix, "image")
        self.assertEqual(schema.inputs[0].template.min, 1)
        self.assertEqual([item.id for item in schema.outputs], ["collage"])
        self.assertIn(MODULE.ReferenceCollage, asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list()))

    def test_collection_orders_dynamic_inputs_and_requires_singletons(self):
        image0 = torch.zeros(1, 4, 5, 3)
        image2 = torch.ones(1, 6, 7, 3)
        ordered, sources = MODULE.collect_images({"image2": image2, "image0": image0}, "2")
        self.assertIs(ordered[0], image0)
        self.assertIs(ordered[1], image2)
        self.assertEqual([item.index for item in sources], [0, 2])
        self.assertEqual([item.weight for item in sources], [2.0, 1.0])
        with self.assertRaisesRegex(ValueError, "one IMAGE"):
            MODULE.collect_images({"image0": torch.zeros(2, 4, 5, 3)}, "1")

    def test_weight_rules(self):
        self.assertEqual(MODULE.parse_weights("2", 3), (2.0, 1.0, 1.0))
        self.assertEqual(MODULE.parse_weights("", 2), (1.0, 1.0))
        for value in ("0", "-1", "nan", "x"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "weight"):
                MODULE.parse_weights(value, 1)
        with self.assertRaisesRegex(ValueError, "3 weights for 2 images"):
            MODULE.parse_weights("1,2,3", 2)

    def test_grid_none_preserves_sizes_gap_alignment_and_incomplete_row(self):
        sources = (source(0, 100, 50), source(1, 40, 80), source(2, 30, 20))
        layout = MODULE.calculate_grid_layout(sources, 2, 10, "none", "end")
        self.assertEqual((layout.width, layout.height), (150, 110))
        placements = layout.placements
        self.assertEqual((placements[0].x, placements[0].y, placements[0].width, placements[0].height), (0, 30, 100, 50))
        self.assertEqual((placements[1].x, placements[1].y), (110, 0))
        self.assertEqual((placements[2].x, placements[2].y), (70, 90))

    def test_grid_auto_preserves_aspect_ratio_and_uses_weights(self):
        sources = (source(0, 100, 50, 1), source(1, 60, 120, 4))
        layout = MODULE.calculate_grid_layout(sources, 2, 0, "auto", "start")
        first, second = layout.placements
        self.assertEqual((first.width, first.height), (120, 60))
        self.assertEqual((second.width, second.height), (60, 120))
        self.assertEqual(first.width * 50, first.height * 100)
        self.assertEqual(second.width * 120, second.height * 60)
        self.assertEqual(first.scale, (1.2, 1.2))

    def test_columns_are_a_maximum_without_placeholder_cells(self):
        layout = MODULE.calculate_grid_layout((source(0, 37, 23),), 8, 12, "none", "center")
        self.assertEqual((layout.width, layout.height), (37, 23))
        self.assertEqual((layout.placements[0].cell_width, layout.placements[0].cell_height), (37, 23))

    def test_mosaic_critical_example_none_and_auto(self):
        sources = tuple(source(index, 512, 512) for index in range(3))
        none = MODULE.calculate_mosaic_layout(sources, 16, "none", "start")
        auto = MODULE.calculate_mosaic_layout(sources, 16, "auto", "start")
        self.assertEqual((none.width, none.height), (1040, 1040))
        self.assertEqual((auto.width, auto.height), (1568, 1040))
        self.assertEqual((auto.placements[2].width, auto.placements[2].height), (1040, 1040))

    def test_mosaic_weights_change_deterministic_geometry(self):
        equal = tuple(source(index, 100, 100) for index in range(4))
        weighted = tuple(source(index, 100, 100, weight) for index, weight in enumerate((4, 1, 1, 1)))
        equal_layout = MODULE.calculate_mosaic_layout(equal, 0, "auto", "center")
        weighted_layout = MODULE.calculate_mosaic_layout(weighted, 0, "auto", "center")
        self.assertEqual(equal_layout, MODULE.calculate_mosaic_layout(equal, 0, "auto", "center"))
        self.assertNotEqual(equal_layout, weighted_layout)
        self.assertGreater(weighted_layout.placements[0].width, weighted_layout.placements[1].width)

    def test_render_is_deterministic_and_uses_background(self):
        red = torch.zeros(1, 2, 2, 3)
        red[..., 0] = 1
        green = torch.zeros(1, 1, 1, 3)
        green[..., 1] = 1
        result1, layout1 = MODULE.reference_collage(
            {"image0": red, "image1": green}, "grid", 2, "1,1", 1, "none", "start", "#0000ff"
        )
        result2, layout2 = MODULE.reference_collage(
            {"image0": red, "image1": green}, "grid", 2, "1,1", 1, "none", "start", "#0000ff"
        )
        self.assertEqual(layout1, layout2)
        self.assertTrue(torch.equal(result1, result2))
        self.assertEqual(tuple(result1.shape), (1, 2, 4, 3))
        self.assertTrue(torch.equal(result1[0, 1, 3], torch.tensor([0.0, 0.0, 1.0])))

    def test_frame_gap_expands_canvas_and_translates_grid_and_mosaic(self):
        sources = (source(0, 20, 10), source(1, 10, 20), source(2, 8, 8))
        for layout_name in ("grid", "mosaic"):
            with self.subTest(layout=layout_name):
                without_frame = MODULE.calculate_layout(sources, layout_name, 2, 3, "none", "center")
                with_frame = MODULE.calculate_layout(sources, layout_name, 2, 3, "none", "center", True)
                self.assertEqual((with_frame.width, with_frame.height), (without_frame.width + 6, without_frame.height + 6))
                for original, framed in zip(without_frame.placements, with_frame.placements):
                    self.assertEqual((framed.x, framed.y), (original.x + 3, original.y + 3))
                    self.assertEqual((framed.cell_x, framed.cell_y), (original.cell_x + 3, original.cell_y + 3))
                    self.assertEqual((framed.width, framed.height), (original.width, original.height))
                    self.assertEqual((framed.cell_width, framed.cell_height), (original.cell_width, original.cell_height))

    def test_frame_gap_renders_background_on_every_edge(self):
        image = torch.ones(1, 2, 3, 3)
        result, layout = MODULE.reference_collage(
            {"image0": image}, "grid", 1, "1", 2, "none", "start", "#000000", True
        )
        self.assertEqual((layout.width, layout.height), (7, 6))
        self.assertTrue(torch.equal(result[:, :2], torch.zeros(1, 2, 7, 3)))
        self.assertTrue(torch.equal(result[:, -2:], torch.zeros(1, 2, 7, 3)))
        self.assertTrue(torch.equal(result[:, 2:4, :2], torch.zeros(1, 2, 2, 3)))
        self.assertTrue(torch.equal(result[:, 2:4, -2:], torch.zeros(1, 2, 2, 3)))
        self.assertTrue(torch.equal(result[:, 2:4, 2:5], image))


if __name__ == "__main__":
    unittest.main()
