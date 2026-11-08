from __future__ import annotations

import logging
from typing import Any

from comfy_api.latest import io

from .segs_indices import parse_index_csv

SEGS = io.Custom("SEGS")
SEGS_INDICES = io.Custom("SEGS_INDICES")
LOGGER = logging.getLogger(__name__)


def validate_assignment_indices(index: object, count: int) -> list[int]:
    if not isinstance(index, list):
        raise TypeError("SEGS Assign (label) index must be a list of integers.")

    seen: set[int] = set()
    for position, seg_index in enumerate(index):
        if isinstance(seg_index, bool) or not isinstance(seg_index, int):
            raise TypeError(
                f"SEGS Assign (label) index[{position}] must be an integer, "
                f"got {type(seg_index).__name__}."
            )
        if seg_index < 0 or seg_index >= count:
            raise ValueError(
                f"SEGS Assign (label) index[{position}]={seg_index} "
                f"is outside the valid SEG range 0..{count - 1}."
            )
        if seg_index in seen:
            raise ValueError(f"SEGS Assign (label) index contains duplicate SEG position {seg_index}.")
        seen.add(seg_index)
    return index


def resolve_assignment_indices(index_text: str, index_input: Any, count: int) -> list[int]:
    if not isinstance(index_text, str):
        raise TypeError("SEGS Assign (label) manual index must be a string.")
    if index_text.strip():
        indices = parse_index_csv(index_text, "SEGS Assign (label)")
    elif index_input is not None:
        indices = index_input
    else:
        indices = list(range(count))
    return validate_assignment_indices(indices, count)


def assign_segs_labels(segs: Any, labels: list[str], indices: list[int]) -> tuple[Any, list[Any]]:
    labels = [label.strip() for label in labels]

    if len(labels) != len(indices):
        LOGGER.warning(
            "[Utility Suite] SEGS Assign (label): length of labels (%s) "
            "!= length of selected segs (%s)",
            len(labels),
            len(indices),
        )

    labeled_segs = list(segs[1])
    for position, seg_index in enumerate(indices):
        if len(labels) > position:
            labeled_segs[seg_index] = labeled_segs[seg_index]._replace(label=labels[position])

    return segs[0], labeled_segs


class SEGSAssignLabel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSEGSAssignLabel",
            display_name="SEGS Assign (label)",
            category="Utility Suite/SEGS",
            description="Assigns comma-separated labels sequentially to selected SEG entries.",
            inputs=[
                SEGS.Input("segs"),
                io.String.Input(
                    "labels",
                    multiline=True,
                    placeholder="List the label to be assigned in order of segs, separated by commas",
                ),
                SEGS_INDICES.Input("index", display_name="index", optional=True),
                io.String.Input(
                    "index_text",
                    display_name="index",
                    default="",
                    multiline=False,
                    tooltip="Optional comma-separated SEG positions. A nonempty value overrides the index socket.",
                ),
            ],
            outputs=[SEGS.Output("SEGS")],
        )

    @classmethod
    def execute(
        cls,
        segs: Any,
        labels: str,
        index: Any = None,
        index_text: str = "",
    ) -> io.NodeOutput:
        indices = resolve_assignment_indices(index_text, index, len(segs[1]))
        return io.NodeOutput(assign_segs_labels(segs, labels.split(","), indices))
