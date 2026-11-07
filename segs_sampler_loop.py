from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder, is_link

from .segs_sampler_custom import (
    _blur_mask,
    _crop,
    _decode,
    _image_resize,
    _mask_tensor,
    _paste,
    _replace_crop,
    _sample_once,
    _validate,
)
from .segs_sampler_unsample import (
    SEGS_UNSAMPLE,
    SEGSUnsampleOptions,
    sample_with_optional_unsample,
)

SEGS = io.Custom("SEGS")
SEG_ELT = io.Custom("SEG_ELT")
SEG_ELT_CROP_REGION = io.Custom("SEG_ELT_crop_region")
SEG_ELT_BBOX = io.Custom("SEG_ELT_bbox")
LOOP_STATE = io.Custom("UTILITY_SEGS_SAMPLER_LOOP")

SUPPORTED_COMFYUI_COMMIT = "2255709aa0be2deade91c7c80cda49d31b73906f"


@dataclass(frozen=True)
class SEGSSamplerLoopState:
    original_image: torch.Tensor
    current_image: torch.Tensor
    current_segs: Any
    current_index: int
    start_index: int
    stop_index: int
    processed_indices: tuple[int, ...] = ()


def _execution_list_hidden() -> Any:
    value = getattr(io.Hidden, "execution_list", None)
    if value is None:
        raise RuntimeError(
            "SEGS Sampler Loop requires ComfyUI's private EXECUTION_LIST hidden input as present at "
            f"commit {SUPPORTED_COMFYUI_COMMIT}."
        )
    return value


def _require_native_loop_api(dynprompt: Any, execution_list: Any) -> None:
    missing = []
    for name in ("get_node", "override_node"):
        if not callable(getattr(dynprompt, name, None)):
            missing.append(f"DYNPROMPT.{name}")
    for name in (
        "add_node",
        "add_external_block",
        "inhibit_nodes",
        "get_external_block_result",
        "release_external_block",
    ):
        if not callable(getattr(execution_list, name, None)):
            missing.append(f"EXECUTION_LIST.{name}")
    if missing:
        raise RuntimeError(
            "SEGS Sampler Loop requires ComfyUI's native loop execution API as present at "
            f"commit {SUPPORTED_COMFYUI_COMMIT}; missing {', '.join(missing)}."
        )


def _iteration_bounds(count: int, start_index: int, max_iteration: int) -> tuple[int, int]:
    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
        raise ValueError(f"SEGS Sampler Loop start_index must be a nonnegative integer, got {start_index!r}.")
    if isinstance(max_iteration, bool) or not isinstance(max_iteration, int) or max_iteration < -1:
        raise ValueError(f"SEGS Sampler Loop max_iteration must be -1 or greater, got {max_iteration!r}.")
    start = min(start_index, count)
    stop = count if max_iteration <= 0 else min(count, start + max_iteration)
    return start, stop


def _initialize_state(
    image: torch.Tensor,
    segs: Any,
    start_index: int,
    stop_index: int,
) -> SEGSSamplerLoopState:
    header, entries, regions = _validate(image, segs)
    initialized = [
        _replace_crop(segment, _crop(image, region).clone())
        for segment, region in zip(entries, regions)
    ]
    initialized_segs = (header, initialized)
    return SEGSSamplerLoopState(
        original_image=image,
        current_image=image,
        current_segs=initialized_segs,
        current_index=start_index,
        start_index=start_index,
        stop_index=stop_index,
    )


def _segment_outputs(state: SEGSSamplerLoopState, index: int) -> tuple[Any, ...]:
    if not state.start_index <= index < state.stop_index:
        raise IndexError(f"SEGS Sampler Loop index {index} is outside [{state.start_index}, {state.stop_index}).")
    segment = state.current_segs[1][index]
    image = torch.as_tensor(segment.cropped_image)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    mask = torch.as_tensor(segment.cropped_mask)
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(f"SEGS Sampler Loop expects current cropped_mask as one MASK, got {tuple(mask.shape)}.")
    return (
        replace(state, current_index=index),
        index,
        state.current_segs,
        state.current_image,
        segment,
        image,
        mask,
        segment.crop_region,
        segment.bbox,
        float(segment.confidence),
        str(segment.label),
    )


def _process_iteration(
    state: SEGSSamplerLoopState,
    noise: Any,
    guider: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    latent: dict[str, Any],
    vae: Any,
    feather: int,
    tiled_decode: str,
    operation: str,
    unsample: SEGSUnsampleOptions | None = None,
) -> tuple[torch.Tensor, Any]:
    if operation not in {"sequential", "individual"}:
        raise ValueError(f"Unsupported operation {operation!r}.")
    if tiled_decode not in {"disabled", "enabled"}:
        raise ValueError(f"Unsupported tiled_decode {tiled_decode!r}.")
    if isinstance(feather, bool) or not isinstance(feather, int) or not 0 <= feather <= 100:
        raise ValueError(f"SEGS Sampler Loop feather must be an integer in 0..100, got {feather!r}.")
    if not isinstance(latent, dict) or not isinstance(latent.get("samples"), torch.Tensor):
        raise TypeError("SEGS Sampler Loop requires LATENT containing a samples tensor.")
    if latent["samples"].shape[0] != 1:
        raise ValueError("SEGS Sampler Loop requires a single latent sample per SEG iteration.")

    index = state.current_index
    if not state.start_index <= index < state.stop_index:
        raise IndexError(f"SEGS Sampler Loop index {index} is outside the active range.")
    header, entries = state.current_segs
    entries = list(entries)
    segment = entries[index]
    region = tuple(int(value) for value in segment.crop_region)
    crop_height = region[3] - region[1]
    crop_width = region[2] - region[0]

    sampled = sample_with_optional_unsample(noise, guider, sampler, sigmas, latent, unsample, _sample_once)
    decoded = _decode(vae, sampled, tiled_decode == "enabled")
    if decoded.shape[0] != 1:
        raise ValueError("SEGS Sampler Loop VAE decode must return one IMAGE per iteration.")
    enhanced = _image_resize(decoded, crop_width, crop_height).cpu()
    entries[index] = _replace_crop(segment, enhanced)
    mask = _blur_mask(_mask_tensor(segment.cropped_mask, crop_height, crop_width), feather)
    is_last = index + 1 >= state.stop_index

    if operation == "individual":
        image = state.current_image
        if is_last:
            image = state.original_image.clone()
            for processed_index in (*state.processed_indices, index):
                processed = entries[processed_index]
                processed_region = tuple(int(value) for value in processed.crop_region)
                processed_crop = torch.as_tensor(processed.cropped_image, dtype=image.dtype)
                processed_mask = _blur_mask(
                    _mask_tensor(
                        processed.cropped_mask,
                        processed_region[3] - processed_region[1],
                        processed_region[2] - processed_region[0],
                    ),
                    feather,
                )
                _paste(image, processed_crop, processed_region, processed_mask)
    else:
        image = state.current_image.clone()
        _paste(image, enhanced, region, mask)
        if is_last:
            for processed_index in (*state.processed_indices, index):
                processed = entries[processed_index]
                processed_region = tuple(int(value) for value in processed.crop_region)
                entries[processed_index] = _replace_crop(processed, _crop(image, processed_region).clone())
        else:
            next_index = index + 1
            next_segment = entries[next_index]
            next_region = tuple(int(value) for value in next_segment.crop_region)
            entries[next_index] = _replace_crop(next_segment, _crop(image, next_region).clone())

    return image, (header, entries)


def _copy_link(source: Any, opener_id: str, iteration: Any, copies: dict[str, Any]) -> Any:
    if not is_link(source):
        return source
    if source[0] == opener_id:
        return iteration.out(source[1])
    if source[0] in copies:
        return copies[source[0]].out(source[1])
    return source


def _expand_loop(
    dynprompt: Any,
    opener_id: str,
    body: set[str],
    close_id: str,
    indices: list[int],
    start_index: int,
    stop_index: int,
) -> dict[str, Any]:
    graph = GraphBuilder()
    opener = dynprompt.get_node(opener_id)
    close = dynprompt.get_node(close_id)
    termination_sources = [
        value
        for name, value in close.get("inputs", {}).items()
        if name.startswith("termination") and is_link(value)
    ]
    initialize = graph.node(
        "UtilitySuiteSEGSSamplerLoopInitialize",
        "initialize",
        image=opener["inputs"]["image"],
        segs=opener["inputs"]["segs"],
        start_index=start_index,
        stop_index=stop_index,
    )
    previous_state = initialize.out(0)
    previous_progress = None
    loop_metadata = {}

    for position, index in enumerate(indices):
        iteration = graph.node(
            "UtilitySuiteSEGSSamplerLoopIteration",
            f"iteration_{position}",
            state=previous_state,
            index=index,
        )
        iteration.set_override_display_id(opener_id)
        copies = {}
        for node_id in sorted(body):
            original = dynprompt.get_node(node_id)
            copy = graph.node(original["class_type"], f"{position}_{node_id}")
            copy.set_override_display_id(node_id)
            copies[node_id] = copy
        for node_id, copy in copies.items():
            original = dynprompt.get_node(node_id)
            for name, input_value in original.get("inputs", {}).items():
                copy.set_input(name, _copy_link(input_value, opener_id, iteration, copies))
            if "_loop_end" in original:
                loop_metadata[copy.id] = {
                    "_loop_body": [copies[body_id].id for body_id in original["_loop_body"]],
                    "_loop_end": copies[original["_loop_end"]].id,
                }

        close_copy = graph.node(close["class_type"], f"{position}_{close_id}")
        close_copy.set_override_display_id(close_id)
        for name, input_value in close.get("inputs", {}).items():
            if name.startswith("termination"):
                continue
            close_copy.set_input(name, _copy_link(input_value, opener_id, iteration, copies))
        dependencies = [
            _copy_link(source, opener_id, iteration, copies)
            for source in termination_sources
        ]
        packed = graph.node(
            "UtilitySuiteSEGSSamplerLoopPackState",
            f"state_{position}",
            previous_state=iteration.out(0),
            image=close_copy.out(0),
            segs=close_copy.out(1),
            index=index,
            **{f"dependency{dependency_index}": dependency for dependency_index, dependency in enumerate(dependencies)},
        )
        previous_state = packed.out(0)
        progress_inputs = {
            "start_id": opener_id,
            "position": position + 1,
            "total": len(indices),
            "dependency0": close_copy.out(0),
            **{
                f"dependency{dependency_index + 1}": dependency
                for dependency_index, dependency in enumerate(dependencies)
            },
        }
        if previous_progress is not None:
            progress_inputs["previous_progress"] = previous_progress
        progress = graph.node("UtilitySuiteSEGSSamplerLoopProgress", f"progress_{position}", **progress_inputs)
        previous_progress = progress.out(0)

    finalized = graph.node("UtilitySuiteSEGSSamplerLoopFinalize", "finalize", state=previous_state)
    result_inputs = {"close_id": close_id, "image": finalized.out(0), "segs": finalized.out(1)}
    if previous_progress is not None:
        result_inputs["progress"] = previous_progress
    graph.node("UtilitySuiteSEGSSamplerLoopResult", "result", **result_inputs)
    expanded = graph.finalize()
    for node_id, metadata in loop_metadata.items():
        expanded[node_id].update(metadata)
    return expanded


class SEGSSamplerLoopStart(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSEGSSamplerLoopStart",
            display_name="SEGS Sampler Loop Start",
            category="Utility Suite/SEGS",
            loop_boundary="start",
            enable_expand=True,
            inputs=[
                io.Image.Input("image"),
                SEGS.Input("segs"),
                io.Int.Input("start_index", default=0, min=0, step=1),
                io.Int.Input("max_iteration", default=-1, min=-1, step=1),
            ],
            outputs=[
                LOOP_STATE.Output("loop"),
                io.Int.Output("segs_index"),
                SEGS.Output("segs"),
                io.Image.Output("image"),
                SEG_ELT.Output("seg_elt"),
                io.Image.Output("cropped_image"),
                io.Mask.Output("cropped_mask"),
                SEG_ELT_CROP_REGION.Output("crop_region"),
                SEG_ELT_BBOX.Output("bbox"),
                io.Float.Output("confidence"),
                io.String.Output("label"),
            ],
            hidden=[io.Hidden.dynprompt, _execution_list_hidden(), io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, image, segs, start_index, max_iteration):
        dynprompt = cls.hidden.dynprompt
        execution_list = cls.hidden.execution_list
        unique_id = cls.hidden.unique_id
        _require_native_loop_api(dynprompt, execution_list)
        _, entries, _ = _validate(image, segs)
        start, stop = _iteration_bounds(len(entries), start_index, max_iteration)
        loop = dynprompt.get_node(unique_id)
        if "_loop_body" not in loop or "_loop_end" not in loop:
            raise RuntimeError("SEGS Sampler Loop was not prepared by ComfyUI native loop validation.")
        body = set(loop["_loop_body"])
        close_id = loop["_loop_end"]
        expanded = _expand_loop(dynprompt, unique_id, body, close_id, list(range(start, stop)), start, stop)

        close = dynprompt.get_node(close_id)
        close_inputs = {
            name: value
            for name, value in close.get("inputs", {}).items()
            if name in {"feather", "tiled_decode", "operation"}
        }
        execution_list.add_node(close_id)
        execution_list.add_external_block(close_id)
        execution_list.inhibit_nodes(body)
        dynprompt.override_node(close_id, {"class_type": close["class_type"], "inputs": close_inputs})
        from server import PromptServer

        PromptServer.instance.send_progress_text(f"Iteration 0 / {stop - start}", unique_id)
        return io.NodeOutput(*([None] * 11), expand=expanded)

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("NaN")


class SEGSSamplerLoopEnd(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        terminations = io.Autogrow.TemplatePrefix(
            io.AnyType.Input(
                "termination",
                tooltip="Connect a preview or side-effect output that must execute on every SEG iteration. Its value is not returned.",
            ),
            prefix="termination",
            min=0,
            max=50,
        )
        return io.Schema(
            node_id="UtilitySuiteSEGSSamplerLoopEnd",
            display_name="SEGS Sampler Loop End",
            category="Utility Suite/SEGS",
            loop_boundary="end",
            inputs=[
                LOOP_STATE.Input("loop"),
                io.Noise.Input("noise"),
                io.Guider.Input("guider"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent"),
                io.Vae.Input("vae"),
                io.Int.Input("feather", default=5, min=0, max=100, step=1),
                io.Combo.Input("tiled_decode", options=["disabled", "enabled"], default="disabled"),
                io.Combo.Input("operation", options=["sequential", "individual"], default="sequential"),
                SEGS_UNSAMPLE.Input("unsample", optional=True),
                io.Autogrow.Input(
                    "terminations",
                    template=terminations,
                    optional=True,
                    tooltip="Connect outputs that must execute on every SEG iteration. Their values are not returned.",
                ),
            ],
            outputs=[io.Image.Output("image"), SEGS.Output("segs")],
            hidden=[_execution_list_hidden(), io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        loop=None,
        noise=None,
        guider=None,
        sampler=None,
        sigmas=None,
        latent=None,
        vae=None,
        feather=5,
        tiled_decode="disabled",
        operation="sequential",
        unsample=None,
        **kwargs,
    ):
        execution_list = cls.hidden.execution_list
        unique_id = cls.hidden.unique_id
        result = execution_list.get_external_block_result(unique_id)
        if result is not None:
            return io.NodeOutput(result[0][0], result[1][0])
        return io.NodeOutput(
            *_process_iteration(
                loop, noise, guider, sampler, sigmas, latent, vae, feather, tiled_decode, operation, unsample
            )
        )

class SEGSSamplerLoopInitialize(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSEGSSamplerLoopInitialize",
            inputs=[io.Image.Input("image"), SEGS.Input("segs"), io.Int.Input("start_index"), io.Int.Input("stop_index")],
            outputs=[LOOP_STATE.Output()],
            is_dev_only=True,
        )

    @classmethod
    def execute(cls, image, segs, start_index, stop_index):
        return io.NodeOutput(_initialize_state(image, segs, start_index, stop_index))


class SEGSSamplerLoopIteration(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSEGSSamplerLoopIteration",
            inputs=[LOOP_STATE.Input("state"), io.Int.Input("index")],
            outputs=SEGSSamplerLoopStart.define_schema().outputs,
            is_dev_only=True,
        )

    @classmethod
    def execute(cls, state, index):
        return io.NodeOutput(*_segment_outputs(state, index))


class SEGSSamplerLoopPackState(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSEGSSamplerLoopPackState",
            inputs=[LOOP_STATE.Input("previous_state"), io.Image.Input("image"), SEGS.Input("segs"), io.Int.Input("index")],
            outputs=[LOOP_STATE.Output()],
            is_dev_only=True,
            accept_all_inputs=True,
        )

    @classmethod
    def execute(cls, previous_state, image, segs, index, **kwargs):
        return io.NodeOutput(
            replace(
                previous_state,
                current_image=image,
                current_segs=segs,
                current_index=index + 1,
                processed_indices=(*previous_state.processed_indices, index),
            )
        )


class SEGSSamplerLoopFinalize(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSEGSSamplerLoopFinalize",
            inputs=[LOOP_STATE.Input("state")],
            outputs=[io.Image.Output(), SEGS.Output()],
            is_dev_only=True,
        )

    @classmethod
    def execute(cls, state):
        return io.NodeOutput(state.current_image, state.current_segs)


class SEGSSamplerLoopProgress(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSEGSSamplerLoopProgress",
            inputs=[io.String.Input("start_id"), io.Int.Input("position"), io.Int.Input("total")],
            outputs=[io.Int.Output()],
            is_output_node=True,
            is_dev_only=True,
            accept_all_inputs=True,
        )

    @classmethod
    def execute(cls, start_id, position, total, **kwargs):
        from server import PromptServer

        PromptServer.instance.send_progress_text(f"Iteration {position} / {total}", start_id)
        return io.NodeOutput(position)

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("NaN")


class SEGSSamplerLoopResult(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSEGSSamplerLoopResult",
            is_input_list=True,
            inputs=[io.String.Input("close_id"), io.Image.Input("image"), SEGS.Input("segs")],
            outputs=[],
            is_output_node=True,
            is_dev_only=True,
            accept_all_inputs=True,
            hidden=[_execution_list_hidden()],
        )

    @classmethod
    def execute(cls, close_id, image, segs, **kwargs):
        cls.hidden.execution_list.release_external_block(close_id[0], [image, segs])
        return io.NodeOutput()

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("NaN")


LOOP_NODE_CLASSES = [
    SEGSSamplerLoopStart,
    SEGSSamplerLoopEnd,
    SEGSSamplerLoopInitialize,
    SEGSSamplerLoopIteration,
    SEGSSamplerLoopPackState,
    SEGSSamplerLoopFinalize,
    SEGSSamplerLoopProgress,
    SEGSSamplerLoopResult,
]

__all__ = ["LOOP_NODE_CLASSES", "SEGSSamplerLoopEnd", "SEGSSamplerLoopStart"]
