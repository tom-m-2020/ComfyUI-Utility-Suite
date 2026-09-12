from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .image_batch import ImageBatchToImageList, ImageListToImageBatch
from .input_kind import ListBatchInspector
from .list_any import ListAnyAppend
from .mask_list import MaskFromList
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
            MaskFromList,
        ]


async def comfy_entrypoint() -> UtilitySuiteExtension:
    return UtilitySuiteExtension()


__all__ = ["UtilitySuiteExtension", "comfy_entrypoint"]
