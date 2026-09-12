from __future__ import annotations

import comfy.utils
import torch
from comfy_api.latest import io


def mask_bounding_box(
    mask: torch.Tensor,
    padding: int,
    image: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int, list[tuple[int, int, int, int]], list[dict[str, int]]]:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"Mask to Bounding Box expects rank-2 or rank-3 MASK, got shape {tuple(mask.shape)}.")
    if mask.shape[0] == 0 or mask.shape[1] == 0 or mask.shape[2] == 0:
        raise ValueError(f"Mask to Bounding Box requires nonempty dimensions, got shape {tuple(mask.shape)}.")
    if padding < 0:
        raise ValueError(f"padding must be nonnegative, got {padding}.")

    coordinates = torch.nonzero(mask, as_tuple=False)
    if coordinates.numel() == 0:
        raise ValueError("Mask to Bounding Box requires at least one nonzero mask pixel.")

    source_height, source_width = mask.shape[1:]
    x0 = max(0, int(coordinates[:, 2].min().item()) - padding)
    y0 = max(0, int(coordinates[:, 1].min().item()) - padding)
    x1 = min(source_width, int(coordinates[:, 2].max().item()) + 1 + padding)
    y1 = min(source_height, int(coordinates[:, 1].max().item()) + 1 + padding)

    if image is None:
        image = mask.unsqueeze(-1).repeat(1, 1, 1, 3)
    else:
        if image.ndim != 4:
            raise ValueError(f"image_optional must be rank-4 IMAGE, got shape {tuple(image.shape)}.")
        if image.shape[0] == 0:
            raise ValueError("image_optional must contain at least one image.")
        if image.shape[1:3] != mask.shape[1:]:
            image = comfy.utils.common_upscale(
                image.movedim(-1, 1),
                source_width,
                source_height,
                upscale_method="bicubic",
                crop="center",
            ).movedim(1, -1)
        if image.shape[0] < mask.shape[0]:
            repeats = mask.shape[0] - image.shape[0]
            image = torch.cat((image, image[-1:].repeat(repeats, 1, 1, 1)), dim=0)
        elif image.shape[0] > mask.shape[0]:
            image = image[: mask.shape[0]]

    width = x1 - x0
    height = y1 - y0
    bbox = [(x0, y0, width, height)]
    bounding_box = [{"x": x0, "y": y0, "width": width, "height": height}]
    return mask[:, y0:y1, x0:x1], image[:, y0:y1, x0:x1, :], x0, y0, width, height, bbox, bounding_box


class MaskToBoundingBox(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteMaskToBoundingBox",
            display_name="Mask to Bounding Box",
            category="Utility Suite/Mask",
            description="Crops a mask and optional image to their shared nonzero bounds without blurring.",
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input("padding", default=0, min=0, max=4096, step=1),
                io.Image.Input("image_optional", optional=True),
            ],
            outputs=[
                io.Mask.Output(display_name="MASK"),
                io.Image.Output(display_name="IMAGE"),
                io.Int.Output("x", display_name="x"),
                io.Int.Output("y", display_name="y"),
                io.Int.Output("width", display_name="width"),
                io.Int.Output("height", display_name="height"),
                io.BBOX.Output("bbox", display_name="bbox", is_output_list=True),
                io.BoundingBox.Output("bounding_box", display_name="bounding_box", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, mask: torch.Tensor, padding: int, image_optional: torch.Tensor | None = None) -> io.NodeOutput:
        return io.NodeOutput(*mask_bounding_box(mask, padding, image_optional))
