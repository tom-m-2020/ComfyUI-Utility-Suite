from __future__ import annotations

import torch
from comfy_api.latest import io


def image_batch_to_list(image: torch.Tensor) -> list[torch.Tensor]:
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError("image must be a rank-4 IMAGE tensor [N,H,W,C].")
    if image.shape[0] < 1:
        raise ValueError("image batch must contain at least one image.")
    return [image[index : index + 1] for index in range(image.shape[0])]


def image_list_to_batch(images: list[torch.Tensor]) -> torch.Tensor:
    if not isinstance(images, (list, tuple)) or not images:
        raise ValueError("images must be a non-empty IMAGE list.")

    first = images[0]
    if not isinstance(first, torch.Tensor) or first.ndim != 4:
        raise ValueError("images[0] must be a rank-4 IMAGE tensor [N,H,W,C].")
    if first.shape[0] < 1:
        raise ValueError("images[0] must contain at least one image.")

    expected_shape = tuple(first.shape[1:])
    for index, image in enumerate(images[1:], start=1):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError(f"images[{index}] must be a rank-4 IMAGE tensor [N,H,W,C].")
        if image.shape[0] < 1:
            raise ValueError(f"images[{index}] must contain at least one image.")
        if tuple(image.shape[1:]) != expected_shape:
            raise ValueError(
                f"images[{index}] has H/W/C {tuple(image.shape[1:])}; expected {expected_shape}. "
                "Ordinary IMAGE batches require common spatial dimensions and channels."
            )
        if image.dtype != first.dtype:
            raise ValueError(f"images[{index}] has dtype {image.dtype}; expected {first.dtype}.")
        if image.device != first.device:
            raise ValueError(f"images[{index}] is on {image.device}; expected {first.device}.")

    if len(images) == 1:
        return first
    return torch.cat(tuple(images), dim=0)


class ImageListToImageBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteImageListToImageBatch",
            display_name="Image List To Image Batch",
            category="Utility Suite/Image/Batch",
            description="Concatenates a Comfy image list into one ordinary IMAGE batch.",
            is_input_list=True,
            inputs=[io.Image.Input("images")],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, images: list[torch.Tensor]) -> io.NodeOutput:
        return io.NodeOutput(image_list_to_batch(images))


class ImageBatchToImageList(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteImageBatchToImageList",
            display_name="Image Batch To Image List",
            category="Utility Suite/Image/Batch",
            description="Splits an ordinary IMAGE batch into singleton-batch images carried as a Comfy list.",
            inputs=[io.Image.Input("image")],
            outputs=[io.Image.Output(display_name="images", is_output_list=True)],
        )

    @classmethod
    def execute(cls, image: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput(image_batch_to_list(image))
