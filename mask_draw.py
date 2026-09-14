from __future__ import annotations

import cv2
import numpy as np
import torch
from comfy_api.latest import io

COLORS = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
}
SHAPES = ("rectangle", "oval", "contour")


def _rectangle(draw: np.ndarray, bbox: tuple[int, int, int, int], line_width: int, fill: bool) -> None:
    x, y, width, height = bbox
    canvas_height, canvas_width = draw.shape
    for offset in range(line_width):
        if y + offset < canvas_height:
            draw[y + offset, x : x + width] = 255
        if y + height - offset < canvas_height:
            draw[y + height - offset, x : x + width] = 255
        if x + offset < canvas_width:
            draw[y : y + height, x + offset] = 255
        if x + width - offset < canvas_width:
            draw[y : y + height, x + width - offset] = 255
    if fill:
        draw[y : y + height, x : x + width] = 255


def _shape_mask(mask: torch.Tensor, shape: str, line_width: int, fill: bool) -> torch.Tensor:
    mask_uint8 = (mask.detach().cpu().numpy() * 255.0).astype(np.uint8)
    draw = np.zeros(mask_uint8.shape[1:], dtype=np.uint8)

    for mask_plane in mask_uint8:
        contours, hierarchy = cv2.findContours(mask_plane, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for index, contour in enumerate(contours):
            if hierarchy[0][index][3] != -1:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if shape == "rectangle":
                _rectangle(draw, (x, y, width, height), line_width, fill)
            elif shape == "oval":
                center = (x + width // 2, y + height // 2)
                axes = (max(0, (width - 1) // 2), max(0, (height - 1) // 2))
                cv2.ellipse(draw, center, axes, 0, 0, 360, 255, -1 if fill else line_width)
            else:
                cv2.drawContours(draw, [contour], 0, 255, -1 if fill else line_width)

    return torch.from_numpy(draw.astype(np.float32) / 255.0)


def draw_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    line_width: int,
    color: str,
    shape: str,
    fill: bool,
    opacity: float,
) -> torch.Tensor:
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError(f"Mask Draw expects rank-4 IMAGE with at least three channels, got shape {tuple(image.shape)}.")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"Mask Draw expects rank-2 or rank-3 MASK, got shape {tuple(mask.shape)}.")
    if mask.shape[1:3] != image.shape[1:3]:
        raise ValueError(
            f"Mask Draw requires matching image/mask dimensions, got image {tuple(image.shape[1:3])} "
            f"and mask {tuple(mask.shape[1:3])}."
        )
    if line_width < 1:
        raise ValueError(f"line_width must be at least 1, got {line_width}.")
    if color not in COLORS:
        raise ValueError(f"Unsupported color: {color!r}.")
    if shape not in SHAPES:
        raise ValueError(f"Unsupported shape: {shape!r}.")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError(f"opacity must be between 0 and 1, got {opacity}.")

    drawing = _shape_mask(mask, shape, line_width, fill).to(device=image.device, dtype=image.dtype)
    alpha = drawing.unsqueeze(0).unsqueeze(-1) * opacity
    selected_color = torch.tensor(COLORS[color], device=image.device, dtype=image.dtype).reshape(1, 1, 1, 3)
    result = image.clone()
    result[..., :3] = result[..., :3] * (1.0 - alpha) + selected_color * alpha
    return result


class MaskDraw(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteMaskDraw",
            display_name="Mask Draw",
            category="Utility Suite/Mask",
            description="Draws every mask component directly over an IMAGE batch.",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Int.Input("line_width", default=1, min=1, max=10, step=1),
                io.Combo.Input("color", options=list(COLORS)),
                io.Combo.Input("shape", options=list(SHAPES)),
                io.Boolean.Input("fill", default=False),
                io.Float.Input("opacity", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Image.Output("image", display_name="image")],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        line_width: int,
        color: str,
        shape: str,
        fill: bool,
        opacity: float,
    ) -> io.NodeOutput:
        return io.NodeOutput(draw_mask(image, mask, line_width, color, shape, fill, opacity))
