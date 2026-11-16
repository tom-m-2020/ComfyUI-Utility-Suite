from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_seed_generator_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.seed_generator"]


class SeedGeneratorTests(unittest.TestCase):
    def test_schema_and_registration(self):
        schema = MODULE.SeedGenerator.define_schema()
        self.assertEqual(schema.node_id, "UtilitySuiteSeedGenerator")
        self.assertEqual(schema.display_name, "Seed Generator")
        self.assertEqual([item.id for item in schema.inputs], ["state"])
        self.assertTrue(schema.inputs[0].extra_dict["seed_generator"])
        self.assertIn(MODULE.SeedGenerator, asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list()))

    def test_fixed_seed_is_preserved(self):
        state = '{"seed":42,"randomize":false}'
        self.assertEqual(MODULE.generate_seed(state), 42)
        self.assertEqual(MODULE.SeedGenerator.fingerprint_inputs(state), 42)

    def test_random_seed_and_fingerprint_change(self):
        state = '{"seed":0,"randomize":true}'
        with patch.object(MODULE.secrets, "randbelow", return_value=123):
            self.assertEqual(MODULE.generate_seed(state), 123)
        with patch.object(MODULE.secrets, "token_hex", side_effect=["a", "b"]):
            self.assertNotEqual(
                MODULE.SeedGenerator.fingerprint_inputs(state),
                MODULE.SeedGenerator.fingerprint_inputs(state),
            )

    def test_execute_reports_actual_seed(self):
        with patch.object(MODULE, "generate_seed", return_value=456):
            output = MODULE.SeedGenerator.execute('{"seed":0,"randomize":true}')
        self.assertEqual(output[0], 456)
        self.assertEqual(output.ui, {"seed": [456]})

    def test_rejects_invalid_state(self):
        invalid = [
            ("[]", "JSON object"),
            ('{"seed":true}', "integer"),
            ('{"seed":-1}', "within"),
            ('{"seed":1125899906842625}', "within"),
            ('{"randomize":1}', "boolean"),
        ]
        for state, message in invalid:
            with self.subTest(state=state), self.assertRaisesRegex((TypeError, ValueError), message):
                MODULE.parse_seed_state(state)


if __name__ == "__main__":
    unittest.main()
