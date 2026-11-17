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
            [
                "images",
                "layout",
                "columns",
                "weights",
                "gap",
                "resize_mode",
                "alignment",
                "background",
                "frame_gap",
                "normalize_mode",
                "normalize_size",
                "allow_upscale",
                "weight_mode",
            ],
        )
        self.assertFalse(schema.inputs[8].default)
        self.assertEqual(schema.inputs[9].default, "none")
        self.assertEqual(schema.inputs[10].default, 512)
        self.assertTrue(schema.inputs[11].default)
        self.assertEqual(schema.inputs[12].default, "area")
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
            {"image0": red, "image1": green}, "grid", 2, "1,1", 1, "none", "start", "#0000ff", False, "none"
        )
        result2, layout2 = MODULE.reference_collage(
            {"image0": red, "image1": green}, "grid", 2, "1,1", 1, "none", "start", "#0000ff", False, "none"
        )
        self.assertEqual(layout1, layout2)
        self.assertTrue(torch.equal(result1, result2))
        self.assertEqual(tuple(result1.shape), (1, 1, 3, 3))
        self.assertTrue(torch.equal(result1[0, 0, 1], torch.tensor([0.0, 0.0, 1.0])))

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
            {"image0": image}, "grid", 1, "1", 2, "none", "start", "#000000", True, "none"
        )
        self.assertEqual((layout.width, layout.height), (7, 6))
        self.assertTrue(torch.equal(result[:, :2], torch.zeros(1, 2, 7, 3)))
        self.assertTrue(torch.equal(result[:, -2:], torch.zeros(1, 2, 7, 3)))
        self.assertTrue(torch.equal(result[:, 2:4, :2], torch.zeros(1, 2, 2, 3)))
        self.assertTrue(torch.equal(result[:, 2:4, -2:], torch.zeros(1, 2, 2, 3)))
        self.assertTrue(torch.equal(result[:, 2:4, 2:5], image))

    def test_normalization_modes_preserve_aspect_ratio(self):
        sources = (source(0, 1000, 500),)
        self.assertEqual(MODULE.normalize_sources(sources, "none", 512, True), sources)
        self.assertEqual((MODULE.normalize_sources(sources, "longest_side", 512, True)[0].width,
                          MODULE.normalize_sources(sources, "longest_side", 512, True)[0].height), (512, 256))
        self.assertEqual((MODULE.normalize_sources(sources, "shortest_side", 512, True)[0].width,
                          MODULE.normalize_sources(sources, "shortest_side", 512, True)[0].height), (1024, 512))
        area = MODULE.normalize_sources(sources, "area", 512, True)[0]
        self.assertAlmostEqual(area.width * area.height, 512 * 512, delta=512)
        self.assertEqual(area.width, area.height * 2)

    def test_normalization_allow_upscale(self):
        sources = (source(0, 100, 50),)
        allowed = MODULE.normalize_sources(sources, "longest_side", 512, True)[0]
        blocked = MODULE.normalize_sources(sources, "longest_side", 512, False)[0]
        self.assertEqual((allowed.width, allowed.height), (512, 256))
        self.assertEqual((blocked.width, blocked.height), (100, 50))

    def test_normalized_resolution_does_not_change_allocated_geometry(self):
        low = (source(0, 512, 512, 2), source(1, 512, 512), source(2, 512, 512))
        high = (source(0, 2048, 2048, 2), source(1, 512, 512), source(2, 512, 512))
        low_weighted = MODULE.apply_weight_sizing(MODULE.normalize_sources(low, "longest_side", 512, True), "area")
        high_weighted = MODULE.apply_weight_sizing(MODULE.normalize_sources(high, "longest_side", 512, True), "area")
        self.assertEqual(low_weighted, high_weighted)
        self.assertEqual(
            MODULE.calculate_layout(low_weighted, "grid", 3, 0, "auto", "start", False, True),
            MODULE.calculate_layout(high_weighted, "grid", 3, 0, "auto", "start", False, True),
        )

    def test_area_weights_control_visible_area_across_aspect_ratios(self):
        normalized = MODULE.normalize_sources(
            (source(0, 800, 400, 2), source(1, 400, 800), source(2, 500, 500)),
            "longest_side",
            512,
            True,
        )
        weighted = MODULE.apply_weight_sizing(normalized, "area")
        areas = [item.width * item.height for item in weighted]
        self.assertAlmostEqual(areas[0] / areas[1], 2.0, delta=0.02)
        self.assertAlmostEqual(areas[0] / areas[2], 2.0, delta=0.02)
        self.assertAlmostEqual(areas[1] / areas[2], 1.0, delta=0.02)

    def test_equal_weights_control_visible_area_across_aspect_ratios(self):
        normalized = MODULE.normalize_sources(
            (source(0, 1200, 300), source(1, 300, 1200), source(2, 640, 640)), "longest_side", 512, True
        )
        weighted = MODULE.apply_weight_sizing(normalized, "area")
        areas = [item.width * item.height for item in weighted]
        self.assertLessEqual(max(areas) / min(areas), 1.01)

    def test_linear_weights_control_linear_scale(self):
        normalized = MODULE.normalize_sources(
            (source(0, 800, 400, 2), source(1, 400, 800), source(2, 500, 500)), "area", 512, True
        )
        weighted = MODULE.apply_weight_sizing(normalized, "linear")
        areas = [item.width * item.height for item in weighted]
        self.assertAlmostEqual(areas[0] / areas[1], 4.0, delta=0.04)
        self.assertAlmostEqual(areas[0] / areas[2], 4.0, delta=0.04)

    def test_normalized_layout_preserves_weighted_sizes_in_all_modes(self):
        normalized = MODULE.normalize_sources(
            (source(0, 300, 600, 2), source(1, 900, 300), source(2, 400, 400)), "shortest_side", 256, True
        )
        weighted = MODULE.apply_weight_sizing(normalized, "area")
        expected = [(item.width, item.height) for item in weighted]
        for layout_name in ("grid", "mosaic"):
            for resize_mode in ("none", "auto"):
                with self.subTest(layout=layout_name, resize=resize_mode):
                    layout = MODULE.calculate_layout(
                        weighted, layout_name, 2, 7, resize_mode, "center", False, True
                    )
                    self.assertEqual([(item.width, item.height) for item in layout.placements], expected)

    def test_none_normalization_preserves_baseline_before_weight_sizing(self):
        images = {
            "image0": torch.zeros(1, 50, 100, 3),
            "image2": torch.zeros(1, 80, 40, 3),
        }
        _, layout = MODULE.reference_collage(
            images, "grid", 2, "1,4", 0, "auto", "start", "#000000", False, "none", 512, True, "area"
        )
        baseline = (source(0, 100, 50, 1), source(2, 40, 80, 4))
        self.assertEqual(MODULE.normalize_sources(baseline, "none", 512, True), baseline)
        areas = [item.width * item.height for item in layout.placements]
        self.assertAlmostEqual(areas[1] / areas[0], 4.0, delta=0.07)

    def test_resize_none_bypasses_every_normalization_setting(self):
        images = {
            "image0": torch.zeros(1, 30, 90, 3),
            "image1": torch.zeros(1, 80, 40, 3),
            "image2": torch.zeros(1, 50, 50, 3),
        }
        _, expected = MODULE.reference_collage(
            images, "grid", 3, "2,1,1", 3, "none", "center", "#000000", False, "none", 64, True, "area"
        )
        for normalize_mode in ("none", "longest_side", "shortest_side", "area"):
            for normalize_size, allow_upscale in ((16, False), (256, True)):
                with self.subTest(mode=normalize_mode, size=normalize_size, upscale=allow_upscale):
                    _, actual = MODULE.reference_collage(
                        images,
                        "grid",
                        3,
                        "2,1,1",
                        3,
                        "none",
                        "center",
                        "#000000",
                        False,
                        normalize_mode,
                        normalize_size,
                        allow_upscale,
                        "area",
                    )
                    self.assertEqual(actual, expected)

    def test_none_equals_auto_with_normalization_none(self):
        images = {
            "image0": torch.zeros(1, 30, 90, 3),
            "image1": torch.zeros(1, 80, 40, 3),
            "image2": torch.zeros(1, 50, 50, 3),
        }
        _, none_layout = MODULE.reference_collage(
            images, "mosaic", 2, "2,1,1", 5, "none", "end", "#000000", True, "area", 32, False, "area"
        )
        _, auto_layout = MODULE.reference_collage(
            images, "mosaic", 2, "2,1,1", 5, "auto", "end", "#000000", True, "none", 512, True, "area"
        )
        self.assertEqual(none_layout, auto_layout)

    def test_weights_resize_images_in_both_resize_modes(self):
        images = {
            "image0": torch.zeros(1, 30, 90, 3),
            "image1": torch.zeros(1, 80, 40, 3),
            "image2": torch.zeros(1, 50, 50, 3),
        }
        for resize_mode in ("none", "auto"):
            with self.subTest(resize=resize_mode):
                _, layout = MODULE.reference_collage(
                    images,
                    "grid",
                    3,
                    "2,1,1",
                    0,
                    resize_mode,
                    "start",
                    "#000000",
                    False,
                    "area",
                    64,
                    True,
                    "area",
                )
                areas = [item.width * item.height for item in layout.placements]
                self.assertAlmostEqual(areas[0] / areas[1], 2.0, delta=0.05)
                self.assertAlmostEqual(areas[0] / areas[2], 2.0, delta=0.05)
                for image, placement in zip(images.values(), layout.placements):
                    source_ratio = image.shape[2] / image.shape[1]
                    self.assertAlmostEqual(placement.width / placement.height, source_ratio, delta=0.03)

    def test_auto_area_normalizes_before_weight_sizing(self):
        images = {
            "image0": torch.zeros(1, 30, 90, 3),
            "image1": torch.zeros(1, 80, 40, 3),
            "image2": torch.zeros(1, 50, 50, 3),
        }
        _, actual = MODULE.reference_collage(
            images, "grid", 3, "2,1,1", 0, "auto", "start", "#000000", False, "area", 64, True, "area"
        )
        sources = (source(0, 90, 30, 2), source(1, 40, 80), source(2, 50, 50))
        expected_sources = MODULE.apply_weight_sizing(MODULE.normalize_sources(sources, "area", 64, True), "area")
        expected = MODULE.calculate_layout(expected_sources, "grid", 3, 0, "auto", "start", False, True)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
