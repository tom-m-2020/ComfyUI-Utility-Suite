import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_mask_draw_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MASK_DRAW = sys.modules[f"{SPEC.name}.mask_draw"]


def draw(image, mask, line_width=1, color="red", shape="rectangle", fill=False, opacity=1.0):
    return MASK_DRAW.draw_mask(image, mask, line_width, color, shape, fill, opacity)


class MaskDrawTests(unittest.TestCase):
    def test_rectangle_matches_kj_pixel_convention(self):
        image = torch.zeros(1, 12, 16, 3)
        mask = torch.zeros(12, 16)
        mask[3:8, 4:10] = 1
        result = draw(image, mask)
        expected = torch.zeros_like(image)
        expected[:, 3, 4:10, 0] = 1
        expected[:, 8, 4:10, 0] = 1
        expected[:, 3:8, 4, 0] = 1
        expected[:, 3:8, 10, 0] = 1
        self.assertTrue(torch.equal(result, expected))

    def test_line_width_grows_edges_inward_like_kj(self):
        image = torch.zeros(1, 14, 18, 3)
        mask = torch.zeros(1, 14, 18)
        mask[:, 3:11, 4:14] = 1
        thin = draw(image, mask, line_width=1)
        thick = draw(image, mask, line_width=3)
        self.assertGreater(thick.count_nonzero().item(), thin.count_nonzero().item())
        self.assertEqual(thick[0, 5, 8, 0].item(), 1)
        self.assertEqual(thick[0, 7, 8, 0].item(), 0)

    def test_multiple_components_and_mask_batch_draw_in_one_call(self):
        image = torch.zeros(1, 20, 25, 3)
        mask = torch.zeros(2, 20, 25)
        mask[0, 2:7, 3:9] = 1
        mask[0, 11:17, 14:21] = 1
        mask[1, 5:10, 18:24] = 1
        result = draw(image, mask)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result[0, 2, 4, 0].item(), 1)
        self.assertEqual(result[0, 11, 15, 0].item(), 1)
        self.assertEqual(result[0, 5, 19, 0].item(), 1)

    def test_oval_and_contour_are_distinct_shapes(self):
        image = torch.zeros(1, 20, 24, 3)
        mask = torch.zeros(20, 24)
        mask[3:16, 4:7] = 1
        mask[13:16, 4:19] = 1
        oval = draw(image, mask, shape="oval")
        contour = draw(image, mask, shape="contour")
        rectangle = draw(image, mask, shape="rectangle")
        self.assertFalse(torch.equal(oval, contour))
        self.assertFalse(torch.equal(contour, rectangle))
        self.assertGreater(oval.count_nonzero().item(), 0)
        self.assertGreater(contour.count_nonzero().item(), 0)

    def test_fill_on_and_off(self):
        image = torch.zeros(1, 12, 16, 3)
        mask = torch.zeros(12, 16)
        mask[2:9, 3:12] = 1
        outline = draw(image, mask, fill=False)
        filled = draw(image, mask, fill=True)
        self.assertEqual(outline[0, 5, 7, 0].item(), 0)
        self.assertEqual(filled[0, 5, 7, 0].item(), 1)

    def test_all_predefined_colors(self):
        expected = {
            "red": (1, 0, 0),
            "green": (0, 1, 0),
            "blue": (0, 0, 1),
            "yellow": (1, 1, 0),
            "black": (0, 0, 0),
            "white": (1, 1, 1),
        }
        mask = torch.zeros(8, 9)
        mask[2:7, 2:8] = 1
        for color, rgb in expected.items():
            with self.subTest(color=color):
                image = torch.full((1, 8, 9, 3), 0.25)
                result = draw(image, mask, color=color, fill=True)
                self.assertEqual(tuple(result[0, 4, 4].tolist()), rgb)

    def test_opacity_zero_partial_and_one(self):
        image = torch.full((1, 9, 11, 3), 0.2)
        mask = torch.zeros(9, 11)
        mask[2:8, 3:10] = 1
        zero = draw(image, mask, fill=True, opacity=0)
        partial = draw(image, mask, fill=True, opacity=0.5)
        full = draw(image, mask, fill=True, opacity=1)
        self.assertTrue(torch.equal(zero, image))
        self.assertTrue(torch.allclose(partial[0, 4, 5], torch.tensor([0.6, 0.1, 0.1])))
        self.assertTrue(torch.equal(full[0, 4, 5], torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(partial[0, 0, 0], image[0, 0, 0]))

    def test_border_touching_non_square_and_image_batch(self):
        image = torch.zeros(2, 11, 19, 4)
        image[..., 3] = 0.7
        mask = torch.zeros(1, 11, 19)
        mask[:, 0:6, 0:8] = 1
        result = draw(image, mask, color="white")
        self.assertEqual(result.shape, (2, 11, 19, 4))
        self.assertTrue(torch.equal(result[0], result[1]))
        self.assertTrue(torch.equal(result[..., 3], image[..., 3]))

    def test_validation(self):
        image = torch.zeros(1, 8, 9, 3)
        mask = torch.zeros(8, 9)
        with self.assertRaisesRegex(ValueError, "matching image/mask"):
            draw(image, torch.zeros(7, 9))
        with self.assertRaisesRegex(ValueError, "rank-2 or rank-3"):
            draw(image, torch.zeros(1, 1, 8, 9))
        with self.assertRaisesRegex(ValueError, "at least three channels"):
            draw(torch.zeros(1, 8, 9, 1), mask)

    def test_schema_and_no_list_transport(self):
        schema = MASK_DRAW.MaskDraw.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteMaskDraw")
        self.assertEqual(schema.display_name, "Mask Draw")
        self.assertEqual(schema.category, "Utility Suite/Mask")
        self.assertEqual(
            [item.id for item in schema.inputs],
            ["image", "mask", "line_width", "color", "shape", "fill", "opacity"],
        )
        self.assertEqual(schema.inputs[3].options, list(MASK_DRAW.COLORS))
        self.assertEqual(schema.inputs[4].options, list(MASK_DRAW.SHAPES))
        self.assertEqual(schema.outputs[0].io_type, "IMAGE")
        self.assertFalse(schema.outputs[0].is_output_list)

    def test_no_custom_node_dependency_or_loop(self):
        source = (PACKAGE_ROOT / "mask_draw.py").read_text(encoding="utf-8").lower()
        for dependency in ("kjnodes", "impact", "easy-use", "for loop"):
            self.assertNotIn(dependency, source)


if __name__ == "__main__":
    unittest.main()
