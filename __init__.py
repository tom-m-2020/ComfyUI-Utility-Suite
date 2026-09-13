from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .any_pipe import (
    PipeFromAny,
    PipeGetAny,
    PipeGetList,
    PipeSetAny,
    PipeSetList,
    PipeToEditAny,
)
from .image_batch import ImageBatchToImageList, ImageListToImageBatch
from .input_kind import ListBatchInspector
from .list_accumulator import ListAccumulatorAppend, ListAccumulatorToList
from .list_any import ListAnyAppend
from .mask_bbox import MaskToBoundingBox
from .mask_list import MaskFromList
from .segs_bbox import SEGSToBBOX
from .tiling import ImageTileBatch, ImageUntileBatch


class UtilitySuiteExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ImageTileBatch,
            ImageUntileBatch,
            ImageListToImageBatch,
            ImageBatchToImageList,
            ListBatchInspector,
            ListAnyAppend,
            ListAccumulatorAppend,
            ListAccumulatorToList,
            MaskFromList,
            MaskToBoundingBox,
            SEGSToBBOX,
            PipeSetAny,
            PipeSetList,
            PipeGetAny,
            PipeGetList,
            PipeToEditAny,
            PipeFromAny,
        ]


async def comfy_entrypoint() -> UtilitySuiteExtension:
    return UtilitySuiteExtension()


__all__ = ["UtilitySuiteExtension", "comfy_entrypoint"]
