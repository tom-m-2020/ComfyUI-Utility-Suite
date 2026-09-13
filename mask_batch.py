from __future__ import annotations

import cv2
import numpy as np
import torch
from comfy_api.latest import io


def mask_to_mask_batch(
    mask: torch.Tensor,
    bbox_fill: bool,
    drop_size: int,
    contour_fill: bool,
) -> torch.Tensor:
    if mask.ndim == 4:
        mask = mask.squeeze(0).squeeze(0)
    elif mask.ndim == 3:
        mask = mask.squeeze(0)

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"Mask to Mask Batch expects a rank-2 or rank-3 MASK, got shape {tuple(mask.shape)}.")
    if mask.shape[1] == 0 or mask.shape[2] == 0:
        raise ValueError(f"Mask to Mask Batch requires nonempty spatial dimensions, got shape {tuple(mask.shape)}.")

    drop_size = max(drop_size, 1)
    mask_array = mask.numpy()
    source_height, source_width = mask_array.shape[1:]
    results: list[torch.Tensor] = []

    for mask_plane in mask_array:
        mask_uint8 = (mask_plane * 255.0).astype(np.uint8)
        contours, hierarchy_tree = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for index, contour in enumerate(contours):
            if hierarchy_tree[0][index][3] != -1:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            if width <= drop_size or height <= drop_size:
                continue

            separated = np.zeros_like(mask_uint8)
            cv2.drawContours(separated, [contour], 0, 255, -1)
            separated = (separated / 255.0).astype(np.float32)

            if contour_fill:
                result = separated
            else:
                result = mask_plane * separated

            if bbox_fill:
                result = result.copy()
                result[y : y + height, x : x + width] = 1.0

            quantized = (result * 255).astype(np.uint8).astype(np.float32) / 255.0
            results.append(torch.from_numpy(quantized))

    if not results:
        return torch.zeros((1, source_height, source_width), dtype=torch.float32, device="cpu")
    return torch.stack(results)


class MaskToMaskBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteMaskToMaskBatch",
            display_name="Mask to Mask Batch",
            category="Utility Suite/Mask",
            description="Separates mask contours into an ordinary full-canvas MASK batch.",
            inputs=[
                io.Mask.Input("mask"),
                io.Boolean.Input("bbox_fill", default=False, label_on="enabled", label_off="disabled"),
                io.Int.Input("drop_size", default=10, min=1, max=16384, step=1),
                io.Boolean.Input("contour_fill", default=False, label_on="enabled", label_off="disabled"),
            ],
            outputs=[io.Mask.Output("masks", display_name="masks")],
        )

    @classmethod
    def execute(
        cls,
        mask: torch.Tensor,
        bbox_fill: bool,
        drop_size: int,
        contour_fill: bool,
    ) -> io.NodeOutput:
        return io.NodeOutput(mask_to_mask_batch(mask, bbox_fill, drop_size, contour_fill))
