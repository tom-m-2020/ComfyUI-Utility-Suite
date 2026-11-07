import asyncio
import copy
import importlib.util
import sys
import unittest
from collections import namedtuple
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import nodes
import numpy as np
import torch
from comfy_execution.validation import LoopValidationError, validate_loops
from execution import PromptExecutor

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_segs_sampler_loop_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
MODULE = sys.modules[f"{SPEC.name}.segs_sampler_loop"]
SAMPLER = sys.modules[f"{SPEC.name}.segs_sampler_custom"]

SEG = namedtuple(
    "SEG", "cropped_image cropped_mask confidence crop_region bbox label control_net_wrapper"
)


class FakeVAE:
    def __init__(self):
        self.calls = []

    def decode(self, samples):
        self.calls.append(("decode", samples))
        return samples

    def encode(self, image):
        self.calls.append(("encode", image))
        return image

    def spacial_compression_encode(self):
        return 1

    def spacial_compression_decode(self):
        return 1

    def decode_tiled(self, samples, **kwargs):
        self.calls.append(("decode_tiled", samples, kwargs))
        return samples


class GraphSource:
    image = None
    segs = None

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("IMAGE", "SEGS")
    FUNCTION = "execute"

    def execute(self):
        return self.image, self.segs


class GraphComponents:
    values = None
    calls = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("NOISE", "GUIDER", "SAMPLER", "SIGMAS", "VAE")
    FUNCTION = "execute"

    def execute(self):
        type(self).calls += 1
        return self.values


class GraphLatent:
    seen: ClassVar[list] = []

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "amount": ("FLOAT", {"default": 0.0})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"

    def execute(self, image, amount):
        self.seen.append(image.clone())
        return ({"samples": image + amount},)


class GraphCapture:
    values: ClassVar[list] = []

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "segs": ("SEGS",)}}

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self, image, segs):
        self.values.append((image, segs))
        return ()


class GraphTermination:
    values: ClassVar[list] = []

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(self, image):
        self.values.append(image.clone())
        return (image,)


class ExecutionServer:
    client_id = None

    def send_sync(self, *args, **kwargs):
        pass


class ProgressServer:
    def send_progress_text(self, *args, **kwargs):
        pass


def segment(region, value=-1, label="x", mask=None, wrapper=None):
    x1, y1, x2, y2 = region
    if mask is None:
        mask = torch.ones((y2 - y1, x2 - x1))
    crop = np.full((1, y2 - y1, x2 - x1, 3), value, dtype=np.float32)
    return SEG(crop, mask, 0.75, region, region, label, wrapper)


def source():
    image = torch.arange(4 * 6, dtype=torch.float32).reshape(1, 4, 6, 1).repeat(1, 1, 1, 3)
    entries = [
        segment((0, 0, 4, 4), label="first", wrapper=object()),
        segment((2, 0, 6, 4), label="second"),
        segment((0, 0, 2, 2), label="third"),
    ]
    return image, ((4, 6), entries)


def sample_add_one(noise, guider, sampler, sigmas, latent):
    result = latent.copy()
    result["samples"] = latent["samples"] + 1
    return result


def run_direct(operation, count=2, tiled_decode="disabled"):
    image = torch.zeros((1, 4, 6, 3))
    segs = ((4, 6), [segment((0, 0, 4, 4), value=0), segment((2, 0, 6, 4), value=0)])
    state = MODULE._initialize_state(image, segs, 0, count)
    vae = FakeVAE()
    for index in range(count):
        loop, *_ = MODULE._segment_outputs(state, index)
        crop = torch.as_tensor(loop.current_segs[1][index].cropped_image)
        with patch.object(MODULE, "_sample_once", sample_add_one):
            updated_image, updated_segs = MODULE._process_iteration(
                loop,
                object(),
                object(),
                object(),
                torch.tensor([1.0, 0.0]),
                {"samples": crop},
                vae,
                0,
                tiled_decode,
                operation,
            )
        state = MODULE.replace(
            loop,
            current_image=updated_image,
            current_segs=updated_segs,
            current_index=index + 1,
            processed_indices=(*loop.processed_indices, index),
        )
    return state, vae


class SEGSSamplerLoopTests(unittest.TestCase):
    def test_schema_defaults_outputs_and_registration(self):
        start = MODULE.SEGSSamplerLoopStart.define_schema()
        end = MODULE.SEGSSamplerLoopEnd.define_schema()
        self.assertEqual(start.display_name, "SEGS Sampler Loop Start")
        self.assertEqual(end.display_name, "SEGS Sampler Loop End")
        self.assertEqual(start.loop_boundary, "start")
        self.assertEqual(end.loop_boundary, "end")
        self.assertEqual([output.id for output in start.outputs], [
            "loop", "segs_index", "segs", "image", "seg_elt", "cropped_image", "cropped_mask",
            "crop_region", "bbox", "confidence", "label",
        ])
        defaults = {item.id: getattr(item, "default", None) for item in start.inputs}
        self.assertEqual((defaults["start_index"], defaults["max_iteration"]), (0, -1))
        self.assertIn("terminations", [item.id for item in end.inputs])
        self.assertNotEqual(MODULE.SEGSSamplerLoopEnd.fingerprint_inputs(), MODULE.SEGSSamplerLoopEnd.fingerprint_inputs())
        nodes = asyncio.run(PACKAGE.UtilitySuiteExtension().get_node_list())
        self.assertIn(MODULE.SEGSSamplerLoopStart, nodes)
        self.assertIn(MODULE.SEGSSamplerLoopEnd, nodes)

    def test_initialization_refreshes_every_crop_and_preserves_metadata(self):
        image, segs = source()
        original_entries = list(segs[1])
        state = MODULE._initialize_state(image, segs, 1, 3)
        self.assertIs(state.original_image, image)
        self.assertIs(state.current_image, image)
        self.assertIsNot(state.current_segs, segs)
        for original, initialized in zip(original_entries, state.current_segs[1]):
            x1, y1, x2, y2 = original.crop_region
            self.assertTrue(np.array_equal(initialized.cropped_image, image[:, y1:y2, x1:x2].numpy()))
            self.assertEqual(initialized._replace(cropped_image=None), original._replace(cropped_image=None))
        self.assertTrue(all(entry.cropped_image.min() == -1 for entry in original_entries))

    def test_initialization_accepts_empty_mask_zero_bbox_sentinel(self):
        image = torch.zeros((1, 4, 4, 3))
        empty = segment((0, 0, 2, 2), mask=torch.zeros((2, 2)))._replace(bbox=(0, 0, 0, 0))
        state = MODULE._initialize_state(image, ((4, 4), [empty]), 0, 1)
        self.assertEqual(state.current_segs[1][0].bbox, (0, 0, 0, 0))
        self.assertTrue(np.array_equal(state.current_segs[1][0].cropped_image, image[:, :2, :2].numpy()))

    def test_iteration_bounds(self):
        self.assertEqual(MODULE._iteration_bounds(10, 0, -1), (0, 10))
        self.assertEqual(MODULE._iteration_bounds(10, 3, -1), (3, 10))
        self.assertEqual(MODULE._iteration_bounds(10, 3, 0), (3, 10))
        self.assertEqual(MODULE._iteration_bounds(10, 3, 4), (3, 7))
        self.assertEqual(MODULE._iteration_bounds(10, 8, 10), (8, 10))
        self.assertEqual(MODULE._iteration_bounds(10, 10, -1), (10, 10))

    def test_iteration_exposes_current_seg_fields(self):
        image, segs = source()
        state = MODULE._initialize_state(image, segs, 1, 3)
        outputs = MODULE._segment_outputs(state, 1)
        self.assertEqual(outputs[1], 1)
        self.assertIs(outputs[2], state.current_segs)
        self.assertIs(outputs[3], image)
        self.assertEqual(outputs[4].label, "second")
        self.assertTrue(torch.equal(outputs[5], image[:, :, 2:6]))
        self.assertEqual(tuple(outputs[6].shape), (1, 4, 4))
        self.assertEqual(outputs[7:11], ((2, 0, 6, 4), (2, 0, 6, 4), 0.75, "second"))

    def test_individual_preserves_canvas_until_final_and_composites_processed_only(self):
        state, _ = run_direct("individual", count=1)
        self.assertTrue(torch.all(state.current_image[:, :, :4] == 1))
        self.assertTrue(torch.all(state.current_image[:, :, 4:] == 0))
        self.assertTrue(np.all(state.current_segs[1][0].cropped_image == 1))
        self.assertTrue(np.all(state.current_segs[1][1].cropped_image == 0))

    def test_individual_intermediate_canvas_is_unchanged(self):
        image = torch.zeros((1, 4, 6, 3))
        segs = ((4, 6), [segment((0, 0, 4, 4), value=0), segment((2, 0, 6, 4), value=0)])
        state = MODULE._initialize_state(image, segs, 0, 2)
        loop, *_ = MODULE._segment_outputs(state, 0)
        with patch.object(MODULE, "_sample_once", sample_add_one):
            updated_image, updated_segs = MODULE._process_iteration(
                loop, object(), object(), object(), torch.tensor([1.0, 0.0]),
                {"samples": torch.as_tensor(loop.current_segs[1][0].cropped_image)}, FakeVAE(),
                0, "disabled", "individual",
            )
        self.assertIs(updated_image, image)
        self.assertTrue(np.all(updated_segs[1][0].cropped_image == 1))
        self.assertTrue(np.all(updated_segs[1][1].cropped_image == 0))

    def test_sequential_updates_canvas_and_refreshes_overlap(self):
        state, _ = run_direct("sequential")
        self.assertTrue(torch.all(state.current_image[:, :, :2] == 1))
        self.assertTrue(torch.all(state.current_image[:, :, 2:4] == 2))
        self.assertTrue(torch.all(state.current_image[:, :, 4:] == 1))
        self.assertTrue(np.all(state.current_segs[1][0].cropped_image[:, :, 2:] == 2))

    def test_overlap_differs_between_modes(self):
        individual, _ = run_direct("individual")
        sequential, _ = run_direct("sequential")
        self.assertTrue(torch.all(individual.current_image[:, :, 2:4] == 1))
        self.assertTrue(torch.all(sequential.current_image[:, :, 2:4] == 2))

    def test_final_results_match_standalone_sampler(self):
        for operation in ("individual", "sequential"):
            loop_state, _ = run_direct(operation)
            image = torch.zeros((1, 4, 6, 3))
            entries = [segment((0, 0, 4, 4), value=0), segment((2, 0, 6, 4), value=0)]
            with patch.object(SAMPLER, "_sample_once", sample_add_one):
                standalone_image, standalone_segs = SAMPLER.segs_sampler_custom(
                    image=image,
                    segs=((4, 6), entries),
                    noise=object(),
                    guider=object(),
                    sampler=object(),
                    sigmas=torch.tensor([1.0, 0.0]),
                    vae=FakeVAE(),
                    guide_size=1,
                    guide_size_for="crop_region",
                    scale_to_guide_size="none",
                    max_size=8,
                    force_inpaint="enabled",
                    feather=0,
                    noise_mask="disabled",
                    noise_mask_feather=0,
                    tiled_encode="disabled",
                    tiled_decode="disabled",
                    operation=operation,
                )
            self.assertTrue(torch.equal(loop_state.current_image, standalone_image))
            for actual, expected in zip(loop_state.current_segs[1], standalone_segs[1]):
                self.assertTrue(np.array_equal(actual.cropped_image, expected.cropped_image))

    def test_tiled_and_normal_decode(self):
        normal, normal_vae = run_direct("individual", count=1)
        tiled, tiled_vae = run_direct("individual", count=1, tiled_decode="enabled")
        self.assertEqual(normal_vae.calls[0][0], "decode")
        self.assertEqual(tiled_vae.calls[0][0], "decode_tiled")
        self.assertTrue(torch.equal(normal.current_image, tiled.current_image))

    def test_sampling_receives_exact_components_and_latent(self):
        image, segs = source()
        state = MODULE._initialize_state(image, segs, 0, 1)
        loop, *_ = MODULE._segment_outputs(state, 0)
        values = [object(), object(), object(), torch.tensor([1.0, 0.0])]
        latent = {"samples": torch.zeros((1, 4, 4, 3))}
        seen = []

        def sample(*args):
            seen.append(args)
            return args[-1]

        with patch.object(MODULE, "_sample_once", sample):
            MODULE._process_iteration(loop, *values, latent, FakeVAE(), 0, "disabled", "individual")
        self.assertEqual(seen[0][:-2], tuple(values[:3]))
        self.assertIs(seen[0][-2], values[3])
        self.assertIs(seen[0][-1], latent)

    def test_empty_range_returns_original_image_and_initialized_segs(self):
        image, segs = source()
        state = MODULE._initialize_state(image, segs, 3, 3)
        image_out, segs_out = MODULE.SEGSSamplerLoopFinalize.execute(state).result
        self.assertIs(image_out, image)
        self.assertEqual(len(segs_out[1]), 3)
        self.assertFalse(np.array_equal(segs_out[1][0].cropped_image, segs[1][0].cropped_image))
        empty = MODULE._initialize_state(image, ((4, 6), []), 0, 0)
        empty_image, empty_segs = MODULE.SEGSSamplerLoopFinalize.execute(empty).result
        self.assertIs(empty_image, image)
        self.assertEqual(empty_segs, ((4, 6), []))

    def test_private_api_incompatibility_is_explicit(self):
        with self.assertRaisesRegex(RuntimeError, "native loop execution API"):
            MODULE._require_native_loop_api(object(), object())

    def test_native_execution_runs_body_for_main_and_flattened_subgraph_ids(self):
        layouts = [
            ("start", "latent", "end", "capture"),
            ("10:start", "10:latent", "10:end", "10:capture"),
        ]
        registered = {
            cls.define_schema().node_id: cls
            for cls in MODULE.LOOP_NODE_CLASSES
        }
        registered.update({
            "LoopGraphSource": GraphSource,
            "LoopGraphComponents": GraphComponents,
            "LoopGraphLatent": GraphLatent,
            "LoopGraphCapture": GraphCapture,
            "LoopGraphTermination": GraphTermination,
        })
        import server

        with patch.dict(nodes.NODE_CLASS_MAPPINGS, registered), patch.object(
            server.PromptServer, "instance", ProgressServer(), create=True
        ), patch.object(MODULE, "_sample_once", sample_add_one), patch.object(
            MODULE, "_initialize_state", wraps=MODULE._initialize_state
        ) as initialize:
            for start, latent, end, capture in layouts:
                initialize.reset_mock()
                image = torch.zeros((1, 4, 6, 3))
                GraphSource.image = image
                GraphSource.segs = ((4, 6), [
                    segment((0, 0, 4, 4), value=-1),
                    segment((2, 0, 6, 4), value=-1),
                ])
                GraphComponents.values = (object(), object(), object(), torch.tensor([1.0, 0.0]), FakeVAE())
                GraphComponents.calls = 0
                GraphLatent.seen = []
                GraphCapture.values = []
                GraphTermination.values = []
                prompt = {
                    "source": {"class_type": "LoopGraphSource", "inputs": {}},
                    "components": {"class_type": "LoopGraphComponents", "inputs": {}},
                    start: {
                        "class_type": "UtilitySuiteSEGSSamplerLoopStart",
                        "inputs": {"image": ["source", 0], "segs": ["source", 1], "start_index": 0, "max_iteration": -1},
                    },
                    latent: {"class_type": "LoopGraphLatent", "inputs": {"image": [start, 5], "amount": 0.0}},
                    "termination": {"class_type": "LoopGraphTermination", "inputs": {"image": [start, 5]}},
                    end: {
                        "class_type": "UtilitySuiteSEGSSamplerLoopEnd",
                        "inputs": {
                            "loop": [start, 0], "noise": ["components", 0], "guider": ["components", 1],
                            "sampler": ["components", 2], "sigmas": ["components", 3], "latent": [latent, 0],
                            "vae": ["components", 4], "feather": 0, "tiled_decode": "disabled",
                            "operation": "sequential",
                            "termination0": ["termination", 0],
                        },
                    },
                    capture: {"class_type": "LoopGraphCapture", "inputs": {"image": [end, 0], "segs": [end, 1]}},
                }
                validate_loops(prompt, {capture}, prompt, {start}, {end})
                executor = PromptExecutor(ExecutionServer(), cache_type=False, cache_args={"ram": 0, "ram_inactive": 0})
                executor.execute(prompt, f"loop-{start}", {}, [capture])
                self.assertTrue(executor.success)
                self.assertEqual(len(GraphLatent.seen), 2)
                self.assertEqual(GraphComponents.calls, 1)
                self.assertEqual(initialize.call_count, 1)
                self.assertEqual(len(GraphTermination.values), 2)
                self.assertTrue(torch.all(GraphTermination.values[1][:, :, :2] == 1))
                self.assertTrue(torch.all(GraphLatent.seen[1][:, :, :2] == 1))
                self.assertTrue(torch.all(GraphCapture.values[0][0][:, :, 2:4] == 2))

    def test_native_validation_rejects_loop_escape(self):
        prompt = {
            "start": {"class_type": "UtilitySuiteSEGSSamplerLoopStart", "inputs": {}},
            "end": {"class_type": "UtilitySuiteSEGSSamplerLoopEnd", "inputs": {"loop": ["start", 0]}},
            "escape": {"class_type": "LoopGraphCapture", "inputs": {"image": ["start", 3]}},
        }
        with self.assertRaises(LoopValidationError):
            validate_loops(prompt, {"end", "escape"}, prompt, {"start"}, {"end"})

    def test_native_execution_invalidates_when_body_setting_changes(self):
        registered = {cls.define_schema().node_id: cls for cls in MODULE.LOOP_NODE_CLASSES}
        registered.update({
            "LoopGraphSource": GraphSource,
            "LoopGraphComponents": GraphComponents,
            "LoopGraphLatent": GraphLatent,
            "LoopGraphCapture": GraphCapture,
        })
        image = torch.zeros((1, 4, 4, 3))
        GraphSource.image = image
        GraphSource.segs = ((4, 4), [segment((0, 0, 4, 4), value=-1)])
        GraphComponents.values = (object(), object(), object(), torch.tensor([1.0, 0.0]), FakeVAE())
        GraphLatent.seen = []
        GraphCapture.values = []
        prompt = {
            "source": {"class_type": "LoopGraphSource", "inputs": {}},
            "components": {"class_type": "LoopGraphComponents", "inputs": {}},
            "start": {
                "class_type": "UtilitySuiteSEGSSamplerLoopStart",
                "inputs": {"image": ["source", 0], "segs": ["source", 1], "start_index": 0, "max_iteration": -1},
            },
            "latent": {"class_type": "LoopGraphLatent", "inputs": {"image": ["start", 5], "amount": 0.0}},
            "end": {
                "class_type": "UtilitySuiteSEGSSamplerLoopEnd",
                "inputs": {
                    "loop": ["start", 0], "noise": ["components", 0], "guider": ["components", 1],
                    "sampler": ["components", 2], "sigmas": ["components", 3], "latent": ["latent", 0],
                    "vae": ["components", 4], "feather": 0, "tiled_decode": "disabled", "operation": "sequential",
                },
            },
            "capture": {"class_type": "LoopGraphCapture", "inputs": {"image": ["end", 0], "segs": ["end", 1]}},
        }
        import server

        with patch.dict(nodes.NODE_CLASS_MAPPINGS, registered), patch.object(
            server.PromptServer, "instance", ProgressServer(), create=True
        ), patch.object(MODULE, "_sample_once", sample_add_one):
            executor = PromptExecutor(ExecutionServer(), cache_type=False, cache_args={"ram": 0, "ram_inactive": 0})
            for amount in (0.0, 2.0):
                current_prompt = copy.deepcopy(prompt)
                current_prompt["latent"]["inputs"]["amount"] = amount
                validate_loops(current_prompt, {"capture"}, current_prompt, {"start"}, {"end"})
                executor.execute(current_prompt, f"loop-cache-{amount}", {}, ["capture"])
                self.assertTrue(executor.success)

        self.assertEqual(len(GraphLatent.seen), 2)
        self.assertTrue(torch.all(GraphCapture.values[0][0] == 1))
        self.assertTrue(torch.all(GraphCapture.values[1][0] == 3))


if __name__ == "__main__":
    unittest.main()
