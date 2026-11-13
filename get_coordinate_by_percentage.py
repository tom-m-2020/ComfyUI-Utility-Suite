from __future__ import annotations

from comfy_api.latest import io


def get_coordinate_by_percentage(
    destination_width: int,
    destination_height: int,
    source_width: int,
    source_height: int,
    x_percent: int,
    y_percent: int,
) -> tuple[int, int]:
    x = round((destination_width - source_width) * (x_percent * 0.01))
    y = round((destination_height - source_height) * (y_percent * 0.01))
    return x, y


class GetCoordinateByPercentage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteGetCoordinateByPercentage",
            display_name="Get Coordinate by Percentage",
            category="Utility Suite/Image",
            description=(
                "Places a source rectangle within a destination using percentage positions, "
                "where 0 is the leading edge and 100 is the trailing edge."
            ),
            inputs=[
                io.Int.Input("destination_width", default=1024, min=1, max=32768, step=1),
                io.Int.Input("destination_height", default=1024, min=1, max=32768, step=1),
                io.Int.Input("source_width", default=1024, min=1, max=32768, step=1),
                io.Int.Input("source_height", default=1024, min=1, max=32768, step=1),
                io.Int.Input("x_percent", default=0, min=0, max=100, step=1),
                io.Int.Input("y_percent", default=0, min=0, max=100, step=1),
            ],
            outputs=[
                io.Int.Output("x"),
                io.Int.Output("y"),
            ],
        )

    @classmethod
    def execute(
        cls,
        destination_width: int,
        destination_height: int,
        source_width: int,
        source_height: int,
        x_percent: int,
        y_percent: int,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            *get_coordinate_by_percentage(
                destination_width,
                destination_height,
                source_width,
                source_height,
                x_percent,
                y_percent,
            )
        )
