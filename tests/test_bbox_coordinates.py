import importlib.util
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_bbox_coordinates_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
BBOX_COORDINATES = sys.modules[f"{SPEC.name}.bbox_coordinates"]


class BBOXCoordinatesTests(unittest.TestCase):
    def test_xywh_to_xyxy(self):
        self.assertEqual(
            BBOX_COORDINATES.convert_bbox((731, 116, 188, 298), "xywh to xyxy"),
            (731, 116, 919, 414),
        )

    def test_xyxy_to_xywh(self):
        self.assertEqual(
            BBOX_COORDINATES.convert_bbox((731, 116, 919, 414), "xyxy to xywh"),
            (731, 116, 188, 298),
        )

    def test_round_trip_both_directions(self):
        xywh = (13, -7, 101, 29)
        xyxy = BBOX_COORDINATES.convert_bbox(xywh, "xywh to xyxy")
        self.assertEqual(BBOX_COORDINATES.convert_bbox(xyxy, "xyxy to xywh"), xywh)
        self.assertEqual(
            BBOX_COORDINATES.convert_bbox(
                BBOX_COORDINATES.convert_bbox(xyxy, "xyxy to xywh"),
                "xywh to xyxy",
            ),
            xyxy,
        )

    def test_zero_extent_and_list_input(self):
        self.assertEqual(BBOX_COORDINATES.convert_bbox([5, 8, 0, 0], "xywh to xyxy"), (5, 8, 5, 8))
        self.assertEqual(BBOX_COORDINATES.convert_bbox([5, 8, 5, 8], "xyxy to xywh"), (5, 8, 0, 0))

    def test_non_square_float_box_preserves_numeric_kind(self):
        result = BBOX_COORDINATES.convert_bbox((1.5, 2.5, 7.25, 3.5), "xywh to xyxy")
        self.assertEqual(result, (1.5, 2.5, 8.75, 6.0))
        self.assertTrue(all(isinstance(value, float) for value in result))

    def test_integer_input_stays_integer(self):
        result = BBOX_COORDINATES.convert_bbox((1, 2, 3, 4), "xywh to xyxy")
        self.assertTrue(all(type(value) is int for value in result))

    def test_malformed_length_and_non_numeric_values(self):
        for bbox in ((1, 2, 3), (1, 2, 3, 4, 5), (1, 2, "3", 4), (1, 2, True, 4)):
            with self.subTest(bbox=bbox), self.assertRaisesRegex(TypeError, "exactly four numeric"):
                BBOX_COORDINATES.convert_bbox(bbox, "xywh to xyxy")

    def test_reversed_xyxy_is_rejected(self):
        for bbox in ((5, 2, 4, 8), (2, 5, 8, 4)):
            with self.subTest(bbox=bbox), self.assertRaisesRegex(ValueError, "must not be reversed"):
                BBOX_COORDINATES.convert_bbox(bbox, "xyxy to xywh")

    def test_negative_extent_is_rejected(self):
        for bbox in ((1, 2, -1, 4), (1, 2, 3, -1)):
            with self.subTest(bbox=bbox), self.assertRaisesRegex(ValueError, "must be nonnegative"):
                BBOX_COORDINATES.convert_bbox(bbox, "xywh to xyxy")

    def test_schema_is_scalar_bbox_to_scalar_bbox(self):
        schema = BBOX_COORDINATES.BBOXCoordinates.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteBBOXCoordinates")
        self.assertEqual(schema.display_name, "BBOX xywh to xyxy")
        self.assertEqual(schema.category, "Utility Suite/BBOX")
        self.assertEqual([item.id for item in schema.inputs], ["bbox", "conversion"])
        self.assertEqual(schema.inputs[1].options, ["xywh to xyxy", "xyxy to xywh"])
        self.assertFalse(schema.is_input_list)
        self.assertEqual(schema.outputs[0].io_type, "BBOX")
        self.assertFalse(schema.outputs[0].is_output_list)

    def test_execute_returns_tuple_bbox(self):
        result = BBOX_COORDINATES.BBOXCoordinates.execute([10, 20, 30, 40], "xywh to xyxy").result[0]
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, (10, 20, 40, 60))


if __name__ == "__main__":
    unittest.main()
