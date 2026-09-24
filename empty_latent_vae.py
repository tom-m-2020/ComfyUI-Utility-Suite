from __future__ import annotations

from numbers import Integral
from typing import Any

import comfy.model_management
import torch
from comfy_api.latest import io


def empty_latent_from_vae(vae: Any, width: int, height: int, batch_size: int) -> dict[str, torch.Tensor]:
    for name, value in (("width", width), ("height", height), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"Empty Latent from VAE requires positive integer {name}, got {value!r}.")

    channels = getattr(vae, "latent_channels", None)
    latent_dimensions = getattr(vae, "latent_dim", None)
    compression_method = getattr(vae, "spacial_compression_encode", None)
    if isinstance(channels, bool) or not isinstance(channels, Integral) or channels <= 0:
        raise ValueError(f"Empty Latent from VAE requires positive integer VAE latent_channels, got {channels!r}.")
    if latent_dimensions not in (2, 3):
        raise ValueError(
            "Empty Latent from VAE supports spatial image VAEs with latent_dim 2 or 3, "
            f"got {latent_dimensions!r}."
        )
    if not callable(compression_method):
        raise TypeError("Empty Latent from VAE expects a ComfyUI VAE with spatial compression metadata.")

    compression = compression_method()
    if isinstance(compression, bool) or not isinstance(compression, Integral) or compression <= 0:
        raise ValueError(
            "Empty Latent from VAE requires a positive integer VAE spatial compression ratio, "
            f"got {compression!r}."
        )

    latent_width = int(width) // int(compression)
    latent_height = int(height) // int(compression)
    if latent_width == 0 or latent_height == 0:
        raise ValueError(
            f"Requested size {(int(width), int(height))} is smaller than VAE spatial compression "
            f"ratio {int(compression)}."
        )

    shape = [int(batch_size), int(channels), latent_height, latent_width]
    if latent_dimensions == 3:
        shape.insert(2, 1)
    samples = torch.zeros(
        shape,
        device=comfy.model_management.intermediate_device(),
        dtype=comfy.model_management.intermediate_dtype(),
    )
    return {"samples": samples}


class EmptyLatentFromVAE(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteEmptyLatentFromVAE",
            display_name="Empty Latent from VAE",
            category="Utility Suite/Latent",
            description=(
                "Creates a fresh zero spatial latent from VAE channel, dimensionality, and "
                "compression metadata without encoding or decoding."
            ),
            inputs=[
                io.Vae.Input("vae"),
                io.Int.Input("width", default=1024, min=1, max=32768, step=1),
                io.Int.Input("height", default=1024, min=1, max=32768, step=1),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[io.Latent.Output("latent")],
        )

    @classmethod
    def execute(cls, vae: Any, width: int, height: int, batch_size: int) -> io.NodeOutput:
        return io.NodeOutput(empty_latent_from_vae(vae, width, height, batch_size))
