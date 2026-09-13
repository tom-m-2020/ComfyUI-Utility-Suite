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
from .bbox_coordinates import BBOXCoordinates
from .bbox_transport import BBOXCollectionToList, BBOXListToCollection
from .image_batch import ImageBatchToImageList, ImageListToImageBatch
from .input_kind import ListBatchInspector
from .list_accumulator import ListAccumulatorAppend, ListAccumulatorToList
from .list_any import ListAnyAppend
from .mask_batch import MaskToMaskBatch
from .mask_batch_combine import MaskBatchToMask
from .mask_bbox import MaskToBoundingBox
from .mask_fill_combined import MaskFillCombined
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
            MaskToMaskBatch,
            MaskBatchToMask,
            MaskFillCombined,
            SEGSToBBOX,
            PipeSetAny,
            PipeSetList,
            PipeGetAny,
            PipeGetList,
            PipeToEditAny,
            PipeFromAny,
            BBOXListToCollection,
            BBOXCollectionToList,
            BBOXCoordinates,
        ]


async def comfy_entrypoint() -> UtilitySuiteExtension:
    return UtilitySuiteExtension()


__all__ = ["UtilitySuiteExtension", "comfy_entrypoint"]
