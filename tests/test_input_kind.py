import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_input_kind_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
INPUT_KIND = sys.modules[f"{SPEC.name}.input_kind"]


class ListBatchInspectorTests(unittest.TestCase):
    def test_multi_item_and_empty_comfy_lists(self):
        self.assertEqual(INPUT_KIND.classify_list_or_batch([]), ("list", 0))
        self.assertEqual(INPUT_KIND.classify_list_or_batch(["a", "b", "c"]), ("list", 3))

    def test_single_python_list_payload(self):
        self.assertEqual(INPUT_KIND.classify_list_or_batch([[1, 2, 3, 4]]), ("list", 4))

    def test_image_and_mask_batches(self):
        self.assertEqual(INPUT_KIND.classify_list_or_batch([torch.zeros(3, 8, 9, 3)]), ("batch", 3))
        self.assertEqual(INPUT_KIND.classify_list_or_batch([torch.zeros(5, 8, 9)]), ("batch", 5))
        self.assertEqual(INPUT_KIND.classify_list_or_batch([torch.zeros(1, 8, 9, 3)]), ("batch", 1))

    def test_unknown_values(self):
        for value in (42, "text", {"samples": torch.zeros(2, 4, 8, 8)}, torch.zeros(8, 9)):
            with self.subTest(value_type=type(value).__name__):
                self.assertEqual(INPUT_KIND.classify_list_or_batch([value]), ("unknown", 0))

    def test_schema_and_execution(self):
        schema = INPUT_KIND.ListBatchInspector.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteListBatchInspector")
        self.assertTrue(schema.is_input_list)
        self.assertEqual(schema.inputs[0].io_type, "*")
        self.assertEqual([output.io_type for output in schema.outputs], ["STRING", "INT"])
        self.assertEqual(INPUT_KIND.ListBatchInspector.execute([torch.zeros(2, 4, 5, 3)]).result, ("batch", 2))


if __name__ == "__main__":
    unittest.main()
