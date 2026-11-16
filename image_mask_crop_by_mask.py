from __future__ import annotations

import torch
from comfy_api.latest import io


def _mask_batch(mask: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(mask).__name__}.")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or any(size == 0 for size in mask.shape):
        raise ValueError(f"{name} must be a nonempty rank-2 or rank-3 MASK, got shape {tuple(mask.shape)}.")
    return mask


def _crop_bounds(mask: torch.Tensor, index: int) -> tuple[int, int, int, int]:
    item = mask[min(index, mask.shape[0] - 1)].round()
    rows = torch.any(item > 0, dim=1)
    columns = torch.any(item > 0, dim=0)
    if not torch.any(rows) or not torch.any(columns):
        raise ValueError(f"Image And Mask Crop By Mask crop-mask item {min(index, mask.shape[0] - 1)} is empty.")
    y_coordinates = torch.where(rows)[0]
    x_coordinates = torch.where(columns)[0]
    return (
        int(x_coordinates[0].item()),
        int(y_coordinates[0].item()),
        int(x_coordinates[-1].item()) + 1,
        int(y_coordinates[-1].item()) + 1,
    )


def _crop_image(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] == 0:
        shape = tuple(image.shape) if isinstance(image, torch.Tensor) else type(image).__name__
        raise ValueError(f"image must be a nonempty rank-4 IMAGE, got {shape}.")
    if image.shape[1:3] != mask.shape[1:]:
        raise ValueError(
            f"image canvas {tuple(image.shape[1:3])} must match crop mask canvas {tuple(mask.shape[1:])}."
        )
    crops = []
    for index in range(image.shape[0]):
        x0, y0, x1, y1 = _crop_bounds(mask, index)
        crops.append(image[index : index + 1, y0:y1, x0:x1, :])
    shapes = {tuple(crop.shape[1:3]) for crop in crops}
    if len(shapes) != 1:
        raise ValueError("Image And Mask Crop By Mask requires equal crop sizes across the image batch.")
    return torch.cat(crops, dim=0)


def _crop_mask(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target = _mask_batch(target, "mask_to_be_cropped")
    if target.shape[1:] != mask.shape[1:]:
        raise ValueError(
            f"mask_to_be_cropped canvas {tuple(target.shape[1:])} must match crop mask canvas {tuple(mask.shape[1:])}."
        )
    crops = []
    for index in range(target.shape[0]):
        x0, y0, x1, y1 = _crop_bounds(mask, index)
        crops.append(target[index : index + 1, y0:y1, x0:x1])
    shapes = {tuple(crop.shape[1:]) for crop in crops}
    if len(shapes) != 1:
        raise ValueError("Image And Mask Crop By Mask requires equal crop sizes across the mask batch.")
    return torch.cat(crops, dim=0)


def image_and_mask_crop_by_mask(
    mask: torch.Tensor,
    image: torch.Tensor | None = None,
    mask_to_be_cropped: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if image is None and mask_to_be_cropped is None:
        raise ValueError("Image And Mask Crop By Mask requires image or mask_to_be_cropped.")
    mask = _mask_batch(mask, "mask")
    cropped_image = _crop_image(image, mask) if image is not None else None
    cropped_mask = _crop_mask(mask_to_be_cropped, mask) if mask_to_be_cropped is not None else None
    return cropped_image, cropped_mask


class ImageAndMaskCropByMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteImageAndMaskCropByMask",
            display_name="Image And Mask Crop By Mask",
            category="Utility Suite/Image",
            description="Crops an optional image and optional mask to the bounds of a required crop mask.",
            inputs=[
                io.Mask.Input("mask"),
                io.Image.Input("image", optional=True),
                io.Mask.Input("mask_to_be_cropped", optional=True),
            ],
            outputs=[io.Image.Output("image"), io.Mask.Output("mask")],
        )

    @classmethod
    def execute(
        cls,
        mask: torch.Tensor,
        image: torch.Tensor | None = None,
        mask_to_be_cropped: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        return io.NodeOutput(*image_and_mask_crop_by_mask(mask, image, mask_to_be_cropped))


__all__ = ["ImageAndMaskCropByMask", "image_and_mask_crop_by_mask"]
