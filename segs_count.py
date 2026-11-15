from __future__ import annotations

from typing import Any

from comfy_api.latest import io

SEGS = io.Custom("SEGS")


def segs_count(segs: Any) -> int:
    if not isinstance(segs, (list, tuple)) or len(segs) != 2:
        raise TypeError("SEGS Count requires an Impact-compatible (source_size, entries) SEGS value.")
    entries = segs[1]
    if not isinstance(entries, list):
        raise TypeError(f"SEGS Count requires the SEGS entries to be a list, got {type(entries).__name__}.")
    return len(entries)


class SEGSCount(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSEGSCount",
            display_name="SEGS Count",
            category="Utility Suite/SEGS",
            description="Returns the number of SEG entries in an Impact-compatible SEGS collection.",
            inputs=[SEGS.Input("segs")],
            outputs=[io.Int.Output("count")],
        )

    @classmethod
    def execute(cls, segs: Any) -> io.NodeOutput:
        return io.NodeOutput(segs_count(segs))


__all__ = ["SEGSCount", "segs_count"]
