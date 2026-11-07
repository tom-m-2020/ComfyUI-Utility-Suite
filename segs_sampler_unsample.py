from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import comfy.samplers
import torch
from comfy_api.latest import io
from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise

SEGS_UNSAMPLE = io.Custom("SEGS_UNSAMPLE")
SUPPORTED_UNSAMPLE_SAMPLERS = {"euler"}
UNSAMPLE_CFG = 1.0


@dataclass(frozen=True)
class SEGSUnsampleOptions:
    model: Any
    positive: Any
    negative: Any
    sampler_name: str
    scheduler: str
    steps: int
    steps_to_run: int


def validate_unsample_options(options: SEGSUnsampleOptions) -> None:
    if options.sampler_name not in comfy.samplers.SAMPLER_NAMES:
        raise ValueError(f"SEGS Sampler Unsample received unknown sampler {options.sampler_name!r}.")
    if options.sampler_name not in SUPPORTED_UNSAMPLE_SAMPLERS:
        raise ValueError(
            f"SEGS Sampler Unsample does not support reverse execution with {options.sampler_name!r}; "
            "supported samplers: euler."
        )
    if options.scheduler not in comfy.samplers.SCHEDULER_NAMES:
        raise ValueError(f"SEGS Sampler Unsample received unknown scheduler {options.scheduler!r}.")
    if isinstance(options.steps, bool) or not isinstance(options.steps, int) or options.steps < 1:
        raise ValueError(f"SEGS Sampler Unsample steps must be a positive integer, got {options.steps!r}.")
    if (
        isinstance(options.steps_to_run, bool)
        or not isinstance(options.steps_to_run, int)
        or options.steps_to_run == 0
        or options.steps_to_run < -1
        or options.steps_to_run > options.steps
    ):
        raise ValueError(
            "SEGS Sampler Unsample steps_to_run must be -1 or an integer in "
            f"1..{options.steps}, got {options.steps_to_run!r}."
        )


def effective_sigmas(options: SEGSUnsampleOptions) -> torch.Tensor:
    validate_unsample_options(options)
    model_sampling = options.model.get_model_object("model_sampling")
    sigmas = comfy.samplers.calculate_sigmas(model_sampling, options.scheduler, options.steps).detach().cpu()
    if sigmas.ndim != 1 or sigmas.numel() != options.steps + 1:
        raise ValueError(
            f"Scheduler {options.scheduler!r} returned {sigmas.numel()} sigmas for {options.steps} intervals; "
            f"expected {options.steps + 1}."
        )

    if float(sigmas[-1]) == 0.0:
        sigmas = comfy.samplers.calculate_sigmas(
            model_sampling, options.scheduler, options.steps + 1
        ).detach().cpu()
        if sigmas.ndim != 1 or sigmas.numel() != options.steps + 2 or float(sigmas[-1]) != 0.0:
            raise ValueError(
                f"Scheduler {options.scheduler!r} cannot produce the required positive-endpoint schedule."
            )
        sigmas = sigmas[:-1]

    if sigmas.numel() != options.steps + 1 or not bool(torch.all(torch.isfinite(sigmas)).item()):
        raise ValueError("SEGS Sampler Unsample requires a finite sigma for every schedule endpoint.")
    if not bool(torch.all(sigmas > 0).item()):
        raise ValueError("SEGS Sampler Unsample requires strictly positive sigma endpoints.")
    if not bool(torch.all(sigmas[:-1] > sigmas[1:]).item()):
        raise ValueError("SEGS Sampler Unsample requires a strictly descending generated schedule.")

    intervals = options.steps if options.steps_to_run == -1 else options.steps_to_run
    return sigmas[-(intervals + 1) :].clone()


def unsample_guider(options: SEGSUnsampleOptions) -> Any:
    guider = comfy.samplers.CFGGuider(options.model)
    guider.set_conds(options.positive, options.negative)
    guider.set_cfg(UNSAMPLE_CFG)
    return guider


def sample_with_optional_unsample(
    noise: Any,
    guider: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    latent: dict[str, Any],
    unsample: SEGSUnsampleOptions | None,
    sample_once: Callable[[Any, Any, Any, torch.Tensor, dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    if unsample is None:
        return sample_once(noise, guider, sampler, sigmas, latent)
    if not isinstance(unsample, SEGSUnsampleOptions):
        raise TypeError("unsample must be produced by SEGS Sampler Unsample.")

    forward_sigmas = effective_sigmas(unsample)
    empty_noise = Noise_EmptyNoise()
    raised = sample_once(
        empty_noise,
        unsample_guider(unsample),
        comfy.samplers.sampler_object(unsample.sampler_name),
        forward_sigmas.flip(0),
        latent,
    )
    return sample_once(empty_noise, guider, sampler, forward_sigmas, raised)


class SEGSSamplerUnsample(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSEGSSamplerUnsample",
            display_name="SEGS Sampler Unsample",
            category="Utility Suite/SEGS",
            description=(
                "Configures paired unsampling and resampling. Unsampling uses CFG 1.0 and positive sigma endpoints."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Combo.Input("sampler_name", options=comfy.samplers.SAMPLER_NAMES, default="euler"),
                io.Combo.Input("scheduler", options=comfy.samplers.SCHEDULER_NAMES, default="simple"),
                io.Int.Input("steps", default=20, min=1, max=10000, step=1),
                io.Int.Input("steps_to_run", default=-1, min=-1, max=10000, step=1),
            ],
            outputs=[SEGS_UNSAMPLE.Output("unsample")],
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        positive: Any,
        negative: Any,
        sampler_name: str,
        scheduler: str,
        steps: int,
        steps_to_run: int,
    ) -> io.NodeOutput:
        options = SEGSUnsampleOptions(
            model=model,
            positive=positive,
            negative=negative,
            sampler_name=sampler_name,
            scheduler=scheduler,
            steps=steps,
            steps_to_run=steps_to_run,
        )
        validate_unsample_options(options)
        return io.NodeOutput(options)


__all__ = [
    "SEGSSamplerUnsample",
    "SEGSUnsampleOptions",
    "effective_sigmas",
    "sample_with_optional_unsample",
]
