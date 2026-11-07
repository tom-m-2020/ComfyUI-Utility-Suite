from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

import comfy.samplers
import torch
from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_segs_sampler_unsample_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.segs_sampler_unsample"]
CUSTOM = sys.modules[f"{SPEC.name}.segs_sampler_custom"]
LOOP = sys.modules[f"{SPEC.name}.segs_sampler_loop"]

SEG = namedtuple(
    "SEG", "cropped_image cropped_mask confidence crop_region bbox label control_net_wrapper"
)


class FakeModel:
    def __init__(self):
        self.model_options = {}
        self.sampling = type("Sampling", (), {"sigmas": torch.linspace(0.01, 1.0, 100)})()

    def get_model_object(self, name):
        if name != "model_sampling":
            raise KeyError(name)
        return self.sampling

    def is_dynamic(self):
        return False


class FakeVAE:
    def spacial_compression_encode(self):
        return 1

    def encode(self, image):
        return image.clone()

    def decode(self, samples):
        return samples


def options(**overrides):
    values = {
        "model": FakeModel(),
        "positive": [[torch.ones(1), {}]],
        "negative": [[torch.zeros(1), {}]],
        "sampler_name": "euler",
        "scheduler": "simple",
        "steps": 4,
        "steps_to_run": -1,
    }
    values.update(overrides)
    return MODULE.SEGSUnsampleOptions(**values)


def linear_sigmas(_model_sampling, _scheduler, steps):
    return torch.linspace(1.0, 0.0, steps + 1)


class SEGSSamplerUnsampleTests(unittest.TestCase):
    def test_schema_defaults_catalogs_required_inputs_and_registration(self):
        schema = MODULE.SEGSSamplerUnsample.define_schema()
        self.assertEqual(schema.display_name, "SEGS Sampler Unsample")
        self.assertEqual(
            [item.id for item in schema.inputs],
            ["model", "positive", "negative", "sampler_name", "scheduler", "steps", "steps_to_run"],
        )
        self.assertTrue(all(not item.optional for item in schema.inputs[:3]))
        self.assertEqual(schema.inputs[3].options, comfy.samplers.SAMPLER_NAMES)
        self.assertEqual(schema.inputs[4].options, comfy.samplers.SCHEDULER_NAMES)
        self.assertEqual([item.default for item in schema.inputs[3:]], ["euler", "simple", 20, -1])
        self.assertEqual(schema.outputs[0].io_type, "SEGS_UNSAMPLE")
        self.assertFalse(schema.outputs[0].is_output_list)
        nodes = asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list())
        self.assertIn(MODULE.SEGSSamplerUnsample, nodes)

    def test_full_zero_terminated_schedule_becomes_positive_with_requested_intervals(self):
        with patch.object(comfy.samplers, "calculate_sigmas", side_effect=linear_sigmas) as calculate:
            sigmas = MODULE.effective_sigmas(options())
        self.assertTrue(torch.allclose(sigmas, torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2])))
        self.assertEqual([call.args[2] for call in calculate.call_args_list], [4, 5])

    def test_partial_schedule_selects_consecutive_low_noise_tail(self):
        with patch.object(comfy.samplers, "calculate_sigmas", side_effect=linear_sigmas):
            sigmas = MODULE.effective_sigmas(options(steps_to_run=2))
        self.assertTrue(torch.allclose(sigmas, torch.tensor([0.6, 0.4, 0.2])))
        self.assertEqual(sigmas.numel() - 1, 2)

    def test_nonzero_terminal_schedule_is_used_without_regeneration(self):
        expected = torch.tensor([1.0, 0.7, 0.4, 0.1])
        with patch.object(comfy.samplers, "calculate_sigmas", return_value=expected) as calculate:
            actual = MODULE.effective_sigmas(options(steps=3))
        self.assertTrue(torch.equal(actual, expected))
        calculate.assert_called_once()

    def test_pair_reverses_then_forwards_and_hands_off_without_fresh_noise(self):
        option = options(steps_to_run=2)
        main_noise = object()
        main_guider = object()
        main_sampler = object()
        main_sigmas = torch.tensor([99.0, 0.0])
        latent = {"samples": torch.zeros(1, 1, 2, 2), "noise_mask": torch.ones(1, 2, 2)}
        calls = []
        outputs = []

        def sample_once(noise, guider, sampler, sigmas, current):
            calls.append((noise, guider, sampler, sigmas.clone(), current))
            result = current.copy()
            result["samples"] = current["samples"] + 1
            outputs.append(result)
            return result

        reverse_sampler = object()
        with patch.object(comfy.samplers, "calculate_sigmas", side_effect=linear_sigmas), patch.object(
            comfy.samplers, "sampler_object", return_value=reverse_sampler
        ):
            result = MODULE.sample_with_optional_unsample(
                main_noise, main_guider, main_sampler, main_sigmas, latent, option, sample_once
            )

        self.assertEqual(len(calls), 2)
        self.assertTrue(all(isinstance(call[0], Noise_EmptyNoise) for call in calls))
        self.assertIs(calls[0][2], reverse_sampler)
        self.assertTrue(torch.allclose(calls[0][3], torch.tensor([0.2, 0.4, 0.6])))
        self.assertIs(calls[1][1], main_guider)
        self.assertIs(calls[1][2], main_sampler)
        self.assertTrue(torch.allclose(calls[1][3], torch.tensor([0.6, 0.4, 0.2])))
        self.assertIs(calls[1][4], outputs[0])
        self.assertTrue(torch.equal(calls[1][4]["samples"], torch.ones_like(latent["samples"])))
        self.assertIs(calls[1][4]["noise_mask"], latent["noise_mask"])
        self.assertTrue(torch.equal(result["samples"], torch.full_like(latent["samples"], 2)))
        self.assertFalse(any(torch.equal(call[3], main_sigmas) for call in calls))

    def test_disconnected_path_uses_main_noise_and_sigmas_once(self):
        values = [object(), object(), object(), torch.tensor([2.0, 0.0]), {"samples": torch.zeros(1)}]
        calls = []

        def sample_once(*args):
            calls.append(args)
            return args[-1]

        result = MODULE.sample_with_optional_unsample(*values, None, sample_once)
        self.assertIs(result, values[-1])
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], values[0])
        self.assertIs(calls[0][3], values[3])

    def test_unsample_guider_uses_option_model_conditioning_and_cfg_one(self):
        option = options()
        guider = MODULE.unsample_guider(option)
        self.assertIs(guider.model_patcher, option.model)
        self.assertEqual(guider.cfg, 1.0)
        self.assertIs(guider.original_conds["positive"][0]["cross_attn"], option.positive[0][0])
        self.assertIs(guider.original_conds["negative"][0]["cross_attn"], option.negative[0][0])

    def test_invalid_steps_sampler_scheduler_and_schedule_fail(self):
        for option in (
            options(steps=0),
            options(steps_to_run=0),
            options(steps_to_run=-2),
            options(steps_to_run=5),
            options(sampler_name="euler_ancestral"),
            options(sampler_name="not-a-sampler"),
            options(scheduler="not-a-scheduler"),
        ):
            with self.subTest(option=option), self.assertRaises(ValueError):
                MODULE.validate_unsample_options(option)
        with patch.object(
            comfy.samplers, "calculate_sigmas", return_value=torch.tensor([1.0, 0.5, 0.5, 0.1, 0.0])
        ), self.assertRaisesRegex(ValueError, "positive-endpoint schedule"):
            MODULE.effective_sigmas(options())

    def test_both_consumers_use_same_optional_socket_and_cycles_are_removed(self):
        custom = CUSTOM.SEGSSamplerCustom.define_schema()
        loop_end = LOOP.SEGSSamplerLoopEnd.define_schema()
        custom_inputs = {item.id: item for item in custom.inputs}
        loop_inputs = {item.id: item for item in loop_end.inputs}
        self.assertNotIn("cycle", custom_inputs)
        self.assertEqual(custom_inputs["unsample"].io_type, "SEGS_UNSAMPLE")
        self.assertTrue(custom_inputs["unsample"].optional)
        self.assertEqual(loop_inputs["unsample"].io_type, "SEGS_UNSAMPLE")
        self.assertTrue(loop_inputs["unsample"].optional)

    def test_custom_sampler_passes_option_in_individual_and_sequential_modes(self):
        image = torch.zeros(1, 2, 2, 3)
        entry = SEG(None, torch.ones(2, 2), 1.0, (0, 0, 2, 2), (0, 0, 2, 2), "x", None)
        option = options()
        seen = []

        def paired(_noise, _guider, _sampler, _sigmas, latent, received, _sample_once):
            seen.append(received)
            result = latent.copy()
            result["samples"] = latent["samples"] + 1
            return result

        arguments = {
            "image": image,
            "segs": ((2, 2), [entry]),
            "noise": object(),
            "guider": object(),
            "sampler": object(),
            "sigmas": torch.tensor([9.0, 0.0]),
            "vae": FakeVAE(),
            "guide_size": 2,
            "guide_size_for": "crop_region",
            "scale_to_guide_size": "none",
            "max_size": 2,
            "force_inpaint": "enabled",
            "feather": 0,
            "noise_mask": "disabled",
            "noise_mask_feather": 0,
            "tiled_encode": "disabled",
            "tiled_decode": "disabled",
            "unsample": option,
        }
        with patch.object(CUSTOM, "sample_with_optional_unsample", side_effect=paired):
            for operation in ("individual", "sequential"):
                output, _ = CUSTOM.segs_sampler_custom(**arguments, operation=operation)
                self.assertTrue(torch.all(output == 1))
        self.assertEqual(seen, [option, option])

    def test_loop_iteration_passes_option_and_preserves_carried_state_rules(self):
        image = torch.zeros(1, 2, 2, 3)
        entry = SEG(None, torch.ones(2, 2), 1.0, (0, 0, 2, 2), (0, 0, 2, 2), "x", None)
        state = LOOP._initialize_state(image, ((2, 2), [entry]), 0, 1)
        state, *_ = LOOP._segment_outputs(state, 0)
        option = options()
        seen = []

        def paired(_noise, _guider, _sampler, _sigmas, latent, received, _sample_once):
            seen.append(received)
            result = latent.copy()
            result["samples"] = latent["samples"] + 1
            return result

        with patch.object(LOOP, "sample_with_optional_unsample", side_effect=paired):
            output, segs = LOOP._process_iteration(
                state,
                object(),
                object(),
                object(),
                torch.tensor([9.0, 0.0]),
                {"samples": torch.zeros(1, 2, 2, 3)},
                FakeVAE(),
                0,
                "disabled",
                "sequential",
                option,
            )
        self.assertEqual(seen, [option])
        self.assertTrue(torch.all(output == 1))
        self.assertEqual(segs[1][0].label, "x")

    def test_simple_scheduler_integration_uses_core_api(self):
        sigmas = MODULE.effective_sigmas(options(steps=3))
        self.assertEqual(sigmas.numel(), 4)
        self.assertTrue(torch.all(sigmas[:-1] > sigmas[1:]))
        self.assertGreater(float(sigmas[-1]), 0.0)


if __name__ == "__main__":
    unittest.main()
