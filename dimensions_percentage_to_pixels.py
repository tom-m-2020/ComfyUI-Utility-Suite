from __future__ import annotations

from comfy_api.latest import io


def dimensions_percentage_to_pixels(
    width: int,
    height: int,
    percent: int,
    divisible_by: int,
) -> tuple[int, int, int, int]:
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (width, height, percent, divisible_by)):
        raise TypeError("Dimensions Percentage to Pixels inputs must be integers.")
    if width < 0 or height < 0:
        raise ValueError("Dimensions Percentage to Pixels width and height must be at least 0.")
    if percent < 0:
        raise ValueError("Dimensions Percentage to Pixels percent must be at least 0.")
    if divisible_by < 1:
        raise ValueError("Dimensions Percentage to Pixels divisible_by must be at least 1.")

    scale = percent * 0.01
    new_width = round((width * scale) / divisible_by) * divisible_by
    new_height = round((height * scale) / divisible_by) * divisible_by
    return width, height, new_width, new_height


class DimensionsPercentageToPixels(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteDimensionsPercentageToPixels",
            display_name="Dimensions Percentage to Pixels",
            category="Utility Suite/Image",
            description="Scales dimensions by a percentage and rounds them to the nearest requested multiple.",
            inputs=[
                io.Int.Input("width", default=1024, min=0, step=1),
                io.Int.Input("height", default=1024, min=0, step=1),
                io.Int.Input("percent", default=100, min=0, step=1),
                io.Int.Input("divisible_by", default=2, min=1, step=1),
            ],
            outputs=[
                io.Int.Output("old_width"),
                io.Int.Output("old_height"),
                io.Int.Output("new_width"),
                io.Int.Output("new_height"),
            ],
        )

    @classmethod
    def execute(
        cls,
        width: int,
        height: int,
        percent: int,
        divisible_by: int,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            *dimensions_percentage_to_pixels(width, height, percent, divisible_by)
        )


__all__ = ["DimensionsPercentageToPixels", "dimensions_percentage_to_pixels"]
