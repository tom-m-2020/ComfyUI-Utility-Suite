from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .any_pipe import (
    PipeAnySingle,
    PipeFromAny,
    PipeGetAny,
    PipeGetList,
    PipeSetAny,
    PipeSetList,
    PipeToEditAny,
)
from .bbox_coordinates import BBOXCoordinates
from .bbox_transport import BBOXCollectionToList, BBOXListToCollection
from .batch_mask_to_segs_comparison import BatchMaskToSEGSWithComparison
from .blur_mask_outward import BlurMaskOutward, BlurMaskOutwardGrowAmount
from .bridge_masks import BridgeMasks
from .combine_masks_segs import CombineMasksSEGSAndMASK, CombineMasksSEGSAndSEGS
from .empty_latent_vae import EmptyLatentFromVAE
from .each_mask_to_mask_batch import EachMaskToMaskBatch
from .feather_mask_boundary import FeatherMaskFromBoundary
from .filter_tile_segs import FilterTileSEGS
from .guide_size_seg_elt import GuideSizeSEGELT
from .image_batch import ImageBatchToImageList, ImageListToImageBatch
from .input_kind import ListBatchInspector
from .invert_mask_segs import InvertMaskSEGS
from .latent_image_size import GetImageSizeFromLatent
from .list_accumulator import ListAccumulatorAppend, ListAccumulatorToList
from .list_any import ListAnyAppend
from .mask_batch import MaskToMaskBatch
from .mask_batch_combine import MaskBatchToMask
from .mask_bbox import MaskToBoundingBox
from .mask_draw import MaskDraw
from .mask_fill_combined import MaskFillCombined
from .mask_fill_holes import MaskFillHoles
from .mask_list import MaskFromList
from .mask_tile_segs import MaskToTileSEGS
from .nested_inpainting_preprocessor import MultiInpaintingMaskProcessor
from .nested_inpainting_preprocessor_simple import MultiInpaintingMaskProcessorSimple
from .preview_segs_regions import PreviewSEGSRegions
from .reorder_batch_mask import ReorderBatchMask
from .reorder_segs import ReorderSEGS
from .reroute_optional import RerouteOptional
from .restore_tile_segs_gaps import RestoreTileSEGSGaps
from .seg_from_segs import SEGFromSEGS
from .segs_assign_label import SEGSAssignLabel
from .segs_bbox import SEGSToBBOX
from .segs_change_ratio import SEGSChangeRatio
from .segs_filter_empty_mask import SEGSFilterEmptyMask
from .segs_insert_by_index import SEGSInsertByIndex
from .segs_overlap_to_mask import SEGSOverlapToMASK
from .segs_range_filter_plus import SEGSRangeFilterPlus
from .segs_sampler_custom import SEGSSamplerCustom
from .segs_sampler_loop import LOOP_NODE_CLASSES
from .segs_sampler_unsample import SEGSSamplerUnsample
from .tiling import ImageTileBatch, ImageUntileBatch


class UtilitySuiteExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ImageTileBatch,
            ImageUntileBatch,
            BridgeMasks,
            BatchMaskToSEGSWithComparison,
            BlurMaskOutward,
            BlurMaskOutwardGrowAmount,
            CombineMasksSEGSAndMASK,
            CombineMasksSEGSAndSEGS,
            EmptyLatentFromVAE,
            EachMaskToMaskBatch,
            FeatherMaskFromBoundary,
            ImageListToImageBatch,
            ImageBatchToImageList,
            ListBatchInspector,
            InvertMaskSEGS,
            GetImageSizeFromLatent,
            ListAnyAppend,
            ListAccumulatorAppend,
            ListAccumulatorToList,
            MaskFromList,
            MaskToBoundingBox,
            MaskToMaskBatch,
            MaskBatchToMask,
            MaskFillCombined,
            MaskFillHoles,
            MaskDraw,
            MaskToTileSEGS,
            MultiInpaintingMaskProcessor,
            MultiInpaintingMaskProcessorSimple,
            SEGSToBBOX,
            SEGSChangeRatio,
            SEGSFilterEmptyMask,
            FilterTileSEGS,
            SEGSInsertByIndex,
            GuideSizeSEGELT,
            PreviewSEGSRegions,
            RestoreTileSEGSGaps,
            RerouteOptional,
            ReorderBatchMask,
            ReorderSEGS,
            SEGFromSEGS,
            SEGSAssignLabel,
            SEGSOverlapToMASK,
            SEGSRangeFilterPlus,
            SEGSSamplerCustom,
            SEGSSamplerUnsample,
            *LOOP_NODE_CLASSES,
            PipeSetAny,
            PipeAnySingle,
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
