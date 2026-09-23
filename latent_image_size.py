from __future__ import annotations

from numbers import Integral
from typing import Any

import torch
from comfy_api.latest import io


def get_latent_image_size(latent: Any, vae: Any) -> tuple[int, int, int, int]:
    if not isinstance(latent, dict):
        raise TypeError("Get Image Size from Latent expects LATENT to be a dictionary.")

    samples = latent.get("samples")
    if not isinstance(samples, torch.Tensor):
        raise TypeError("Get Image Size from Latent expects LATENT['samples'] to be a torch.Tensor.")
    if samples.ndim not in (4, 5):
        raise ValueError(
            "Get Image Size from Latent supports 4-D [B,C,H,W] and "
            "5-D [B,C,T,H,W] spatial latents."
        )
    if any(dimension <= 0 for dimension in samples.shape):
        raise ValueError(f"Get Image Size from Latent expects nonempty samples, got {tuple(samples.shape)}.")

    compression_method = getattr(vae, "spacial_compression_decode", None)
    if not callable(compression_method):
        raise TypeError("Get Image Size from Latent expects a ComfyUI VAE with spatial compression metadata.")
    compression = compression_method()
    if isinstance(compression, bool) or not isinstance(compression, Integral) or compression <= 0:
        raise ValueError(
            "Get Image Size from Latent requires a positive integer VAE spatial compression ratio, "
            f"got {compression!r}."
        )

    batch_size = int(samples.shape[0])
    channels = int(samples.shape[1])
    height = int(samples.shape[-2]) * int(compression)
    width = int(samples.shape[-1]) * int(compression)
    return width, height, batch_size, channels


class GetImageSizeFromLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteGetImageSizeFromLatent",
            display_name="Get Image Size from Latent",
            category="Utility Suite/Latent",
            description=(
                "Reports the represented pixel size using the matching VAE's spatial compression "
                "metadata without decoding the latent."
            ),
            inputs=[
                io.Latent.Input("latent"),
                io.Vae.Input("vae"),
            ],
            outputs=[
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.Int.Output("batch_size"),
                io.Int.Output("channels"),
            ],
        )

    @classmethod
    def execute(cls, latent: Any, vae: Any) -> io.NodeOutput:
        return io.NodeOutput(*get_latent_image_size(latent, vae))
