import importlib.util
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_bbox_transport_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
BBOX_TRANSPORT = sys.modules[f"{SPEC.name}.bbox_transport"]


class BBOXTransportTests(unittest.TestCase):
    def test_pack_empty_single_and_multiple(self):
        bbox0 = (0, 0, 768, 1024)
        bbox1 = [640, 0, 768, 1024]
        self.assertEqual(BBOX_TRANSPORT.bbox_list_to_collection([]), [])
        self.assertEqual(BBOX_TRANSPORT.bbox_list_to_collection([bbox0]), [bbox0])
        self.assertEqual(BBOX_TRANSPORT.bbox_list_to_collection([bbox0, bbox1]), [bbox0, bbox1])

    def test_pack_preserves_item_identity_order_and_outer_input(self):
        boxes = [(0, 0, 1, 2), [3, 4, 5, 6], object()]
        packed = BBOX_TRANSPORT.bbox_list_to_collection(boxes)
        self.assertIsNot(packed, boxes)
        self.assertTrue(all(actual is expected for actual, expected in zip(packed, boxes, strict=True)))
        self.assertEqual(len(boxes), 3)

    def test_unpack_tuple_single_bbox_as_one_item(self):
        bbox = (0, 0, 768, 1024)
        result = BBOX_TRANSPORT.bbox_collection_to_list(bbox)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], bbox)

    def test_unpack_list_single_bbox_as_one_item(self):
        bbox = [0, 0, 768, 1024]
        result = BBOX_TRANSPORT.bbox_collection_to_list(bbox)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], bbox)

    def test_unpack_nested_collection_preserves_items(self):
        boxes = [(0, 0, 768, 1024), [640, 0, 768, 1024], (1280, 0, 768, 1024)]
        result = BBOX_TRANSPORT.bbox_collection_to_list(boxes)
        self.assertTrue(all(actual is expected for actual, expected in zip(result, boxes, strict=True)))

    def test_empty_round_trip_remains_empty(self):
        packed = BBOX_TRANSPORT.bbox_list_to_collection([])
        self.assertEqual(BBOX_TRANSPORT.bbox_collection_to_list(packed), [])

    def test_multi_box_round_trip_preserves_values_and_order(self):
        boxes = [(0, 0, 768, 1024), [640, 0, 768, 1024], (1280, 0, 768, 1024)]
        packed = BBOX_TRANSPORT.BBOXListToCollection.execute(boxes).result[0]
        unpacked = BBOX_TRANSPORT.BBOXCollectionToList.execute(packed).result[0]
        self.assertTrue(all(actual is expected for actual, expected in zip(unpacked, boxes, strict=True)))

    def test_invalid_collection_fails_clearly(self):
        with self.assertRaisesRegex(TypeError, "list or tuple"):
            BBOX_TRANSPORT.bbox_collection_to_list({"x": 1})

    def test_nodes_2_schema_declares_static_transport_boundaries(self):
        pack = BBOX_TRANSPORT.BBOXListToCollection.define_schema()
        unpack = BBOX_TRANSPORT.BBOXCollectionToList.define_schema()
        self.assertTrue(pack.is_input_list)
        self.assertFalse(pack.outputs[0].is_output_list)
        self.assertFalse(unpack.is_input_list)
        self.assertTrue(unpack.outputs[0].is_output_list)
        self.assertEqual(pack.inputs[0].io_type, "BBOX")
        self.assertEqual(pack.outputs[0].io_type, "BBOX")
        self.assertEqual(unpack.inputs[0].io_type, "BBOX")
        self.assertEqual(unpack.outputs[0].io_type, "BBOX")


if __name__ == "__main__":
    unittest.main()
