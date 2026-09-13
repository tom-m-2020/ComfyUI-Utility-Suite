import importlib.util
import sys
import unittest
from collections import namedtuple
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_segs_bbox_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
SEGS_BBOX = sys.modules[f"{SPEC.name}.segs_bbox"]
SEG = namedtuple("SEG", ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "wrapper"])


def segment(bbox):
    return SEG(None, None, 1.0, None, bbox, "test", None)


class BBoxOnlySegment:
    bbox = (3, 4, 8, 10)

    @property
    def cropped_image(self):
        raise AssertionError("cropped_image must not be accessed")


class SEGSToBBOXTests(unittest.TestCase):
    def test_one_segment_exact_outputs(self):
        result = SEGS_BBOX.segs_to_bboxes(((1024, 1024), [segment((731, 116, 919, 414))]))
        self.assertEqual(result[0], [(731, 116, 919, 414)])
        self.assertEqual(result[1], [(731, 116, 188, 298)])
        self.assertEqual(result[2], [{"x": 731, "y": 116, "width": 188, "height": 298}])

    def test_multiple_segments_preserve_order(self):
        boxes = [(428, 729, 618, 924), (85, 387, 254, 564), (731, 116, 919, 414)]
        xyxy, xywh, bounding_boxes = SEGS_BBOX.segs_to_bboxes(((1024, 1024), [segment(box) for box in boxes]))
        self.assertEqual(xyxy, boxes)
        self.assertEqual(xywh, [(428, 729, 190, 195), (85, 387, 169, 177), (731, 116, 188, 298)])
        self.assertEqual(
            bounding_boxes,
            [
                {"x": 428, "y": 729, "width": 190, "height": 195},
                {"x": 85, "y": 387, "width": 169, "height": 177},
                {"x": 731, "y": 116, "width": 188, "height": 298},
            ],
        )

    def test_empty_segs_returns_three_empty_lists(self):
        self.assertEqual(SEGS_BBOX.segs_to_bboxes(((640, 960), [])), ([], [], []))

    def test_non_square_and_zero_extent_boxes(self):
        result = SEGS_BBOX.segs_to_bboxes(((100, 100), [segment((2, 3, 9, 20)), segment((5, 6, 5, 6))]))
        self.assertEqual(result[1], [(2, 3, 7, 17), (5, 6, 0, 0)])

    def test_does_not_access_cropped_image(self):
        result = SEGS_BBOX.segs_to_bboxes(((20, 20), [BBoxOnlySegment()]))
        self.assertEqual(result[0], [(3, 4, 8, 10)])

    def test_reversed_coordinates_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "reversed coordinates"):
            SEGS_BBOX.segs_to_bboxes(((20, 20), [segment((8, 4, 3, 10))]))

    def test_malformed_segs_and_bbox_fail_clearly(self):
        with self.assertRaisesRegex(TypeError, "source_size"):
            SEGS_BBOX.segs_to_bboxes([])
        with self.assertRaisesRegex(TypeError, "no bbox"):
            SEGS_BBOX.segs_to_bboxes(((20, 20), [object()]))
        with self.assertRaisesRegex(TypeError, "four coordinates"):
            SEGS_BBOX.segs_to_bboxes(((20, 20), [segment((1, 2, 3))]))

    def test_nodes_2_schema_declares_three_list_outputs(self):
        schema = SEGS_BBOX.SEGSToBBOX.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteSEGSToBBOX")
        self.assertEqual(schema.display_name, "SEGS to BBOX")
        self.assertEqual(schema.category, "Utility Suite/SEGS")
        self.assertEqual(schema.inputs[0].io_type, "SEGS")
        self.assertEqual([output.io_type for output in schema.outputs], ["BBOX", "BBOX", "BOUNDING_BOX"])
        self.assertTrue(all(output.is_output_list for output in schema.outputs))


if __name__ == "__main__":
    unittest.main()
