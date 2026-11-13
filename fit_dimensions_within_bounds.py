from __future__ import annotations

from comfy_api.latest import io


def fit_dimensions_within_bounds(
    destination_width: int,
    destination_height: int,
    source_width: int,
    source_height: int,
) -> tuple[int, int]:
    scale = min(destination_width / source_width, destination_height / source_height)
    return round(scale * source_width), round(scale * source_height)


class FitDimensionsWithinBounds(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteFitDimensionsWithinBounds",
            display_name="Fit Dimensions Within Bounds",
            category="Utility Suite/Image",
            description=(
                "Fits source dimensions within destination bounds while preserving the source aspect ratio."
            ),
            inputs=[
                io.Int.Input("destination_width", default=1024, min=1, max=32768, step=1),
                io.Int.Input("destination_height", default=1024, min=1, max=32768, step=1),
                io.Int.Input("source_width", default=1024, min=1, max=32768, step=1),
                io.Int.Input("source_height", default=1024, min=1, max=32768, step=1),
            ],
            outputs=[
                io.Int.Output("width"),
                io.Int.Output("height"),
            ],
        )

    @classmethod
    def execute(
        cls,
        destination_width: int,
        destination_height: int,
        source_width: int,
        source_height: int,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            *fit_dimensions_within_bounds(
                destination_width,
                destination_height,
                source_width,
                source_height,
            )
        )
