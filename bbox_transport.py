from __future__ import annotations

from typing import Any

from comfy_api.latest import io


def bbox_list_to_collection(bboxes: list[Any] | tuple[Any, ...]) -> list[Any]:
    if not isinstance(bboxes, (list, tuple)):
        raise TypeError("BBOX List to Collection requires Comfy's whole-list input container.")
    return list(bboxes)


def bbox_collection_to_list(bboxes: Any) -> list[Any]:
    if not isinstance(bboxes, (list, tuple)):
        raise TypeError(f"BBOX collection must be a list or tuple, got {type(bboxes).__name__}.")
    if not bboxes:
        return []
    if len(bboxes) == 4 and all(isinstance(value, (int, float)) for value in bboxes):
        return [bboxes]
    return list(bboxes)


class BBOXListToCollection(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteBBOXListToCollection",
            display_name="BBOX List to Collection",
            category="Utility Suite/BBOX",
            description="Packs a genuine Comfy BBOX list into one ordinary legacy collection value.",
            is_input_list=True,
            inputs=[io.BBOX.Input("bboxes")],
            outputs=[io.BBOX.Output("bboxes", display_name="bboxes")],
        )

    @classmethod
    def execute(cls, bboxes: list[Any]) -> io.NodeOutput:
        return io.NodeOutput(bbox_list_to_collection(bboxes))


class BBOXCollectionToList(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteBBOXCollectionToList",
            display_name="BBOX Collection to List",
            category="Utility Suite/BBOX",
            description="Unpacks one legacy BBOX collection value into a genuine Comfy list.",
            inputs=[io.BBOX.Input("bboxes")],
            outputs=[io.BBOX.Output("bboxes", display_name="bboxes", is_output_list=True)],
        )

    @classmethod
    def execute(cls, bboxes: Any) -> io.NodeOutput:
        return io.NodeOutput(bbox_collection_to_list(bboxes))
