from __future__ import annotations

from dataclasses import dataclass

import torch
from comfy_api.latest import io

from .mask_bbox import quantized_mask_bounds
from .reorder_segs import SPATIAL_ORDERS, grid_coordinates

ORDERS = [*SPATIAL_ORDERS, "biggest", "smallest"]


@dataclass
class _AxisCluster:
    low: int
    high: int
    centers: list[float]

    @property
    def center(self) -> float:
        return sum(self.centers) / len(self.centers)


def _axis_indices(
    boxes: list[tuple[int, int, int, int]],
    low_axis: int,
    high_axis: int,
) -> list[int]:
    ordered = sorted(
        range(len(boxes)),
        key=lambda index: (
            (boxes[index][low_axis] + boxes[index][high_axis]) / 2,
            boxes[index][low_axis],
            boxes[index][high_axis],
            index,
        ),
    )
    clusters: list[_AxisCluster] = []
    assignments = [0] * len(boxes)
    for index in ordered:
        low = boxes[index][low_axis]
        high = boxes[index][high_axis]
        center = (low + high) / 2
        eligible = [
            (abs(center - cluster.center), cluster_index)
            for cluster_index, cluster in enumerate(clusters)
            if cluster.low <= center <= cluster.high
        ]
        if eligible:
            _, cluster_index = min(eligible)
            cluster = clusters[cluster_index]
            cluster.low = max(cluster.low, low)
            cluster.high = min(cluster.high, high)
            cluster.centers.append(center)
        else:
            cluster_index = len(clusters)
            clusters.append(_AxisCluster(low, high, [center]))
        assignments[index] = cluster_index
    return assignments


def _spatial_indices(
    boxes_by_index: list[tuple[int, tuple[int, int, int, int]]],
    order: str,
) -> list[int]:
    if not boxes_by_index:
        return []
    original_indices = [index for index, _ in boxes_by_index]
    boxes = [box for _, box in boxes_by_index]
    rows = _axis_indices(boxes, 1, 3)
    columns = _axis_indices(boxes, 0, 2)
    cells: dict[tuple[int, int], int] = {}
    for original_index, row, column in zip(original_indices, rows, columns):
        cell = (row, column)
        if cell in cells:
            raise ValueError(
                "Reorder Batch Mask cannot derive a unique position lattice: "
                f"batch items {cells[cell]} and {original_index} occupy logical cell {cell}."
            )
        cells[cell] = original_index
    traversal = grid_coordinates(max(rows) + 1, max(columns) + 1, order)
    return [cells[cell] for cell in traversal if cell in cells]


def reorder_batch_mask(mask: torch.Tensor, order: str) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"Reorder Batch Mask expects a torch.Tensor MASK, got {type(mask).__name__}.")
    if mask.ndim != 3:
        raise ValueError(f"Reorder Batch Mask expects MASK [B,H,W], got shape {tuple(mask.shape)}.")
    if any(dimension == 0 for dimension in mask.shape):
        raise ValueError(f"Reorder Batch Mask requires nonempty dimensions, got shape {tuple(mask.shape)}.")
    if order not in ORDERS:
        raise ValueError(f"Unsupported Reorder Batch Mask order {order!r}; expected one of {ORDERS}.")

    boxes = [quantized_mask_bounds(item) for item in mask]
    areas = [0 if box is None else (box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    if order == "biggest":
        indices = sorted(range(len(boxes)), key=lambda index: -areas[index])
    elif order == "smallest":
        indices = sorted(range(len(boxes)), key=lambda index: areas[index])
    else:
        present = [(index, box) for index, box in enumerate(boxes) if box is not None]
        indices = _spatial_indices(present, order)
        indices.extend(index for index, box in enumerate(boxes) if box is None)

    selection = torch.tensor(indices, dtype=torch.long, device=mask.device)
    return torch.index_select(mask, 0, selection)


class ReorderBatchMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteReorderBatchMask",
            display_name="Reorder Batch Mask",
            category="Utility Suite/Mask",
            description="Reorders a MASK batch from content bounds without changing mask pixels.",
            inputs=[io.Mask.Input("mask"), io.Combo.Input("order", options=ORDERS)],
            outputs=[io.Mask.Output("mask", display_name="mask")],
        )

    @classmethod
    def execute(cls, mask: torch.Tensor, order: str) -> io.NodeOutput:
        return io.NodeOutput(reorder_batch_mask(mask, order))


__all__ = ["ORDERS", "ReorderBatchMask", "reorder_batch_mask"]
