from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import torch
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

LAYOUT_VERSION = 1
LAYOUT_ORDER = "source_row_column"
LAYOUT_COMPATIBILITY = "explicit_rectangles_v1"
LEGACY_LINEAR_MERGE_POLICY = "normalized_linear_overlap_v1"
SUPPORTED_LAYOUT_POLICIES = frozenset((LAYOUT_COMPATIBILITY, LEGACY_LINEAR_MERGE_POLICY))
TILE_LAYOUT = io.Custom("TILE_LAYOUT")


@dataclass(frozen=True)
class Rect:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class NeighborOverlaps:
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True)
class SpatialTileRecord:
    row: int
    column: int
    source_rect: Rect
    valid_rect: Rect
    neighbor_overlaps: NeighborOverlaps


@dataclass(frozen=True)
class RequestedTileParameters:
    mode: Literal["fixed_tile", "bounded_grid"]
    tile_width: int
    tile_height: int
    overlap_x: int
    overlap_y: int
    max_columns: int
    max_rows: int


@dataclass(frozen=True)
class TileLayout:
    version: int
    source_batch_size: int
    source_height: int
    source_width: int
    channels: int
    tile_tensor_height: int
    tile_tensor_width: int
    rows: int
    columns: int
    geometry_mode: Literal["fixed_tile", "bounded_grid"]
    requested_parameters: RequestedTileParameters
    spatial_records: tuple[SpatialTileRecord, ...]
    order: str = LAYOUT_ORDER
    merge_policy: str = LAYOUT_COMPATIBILITY

    @property
    def required_tile_count(self) -> int:
        return self.source_batch_size * len(self.spatial_records)


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("Tiling denominator must be positive.")
    return -(-numerator // denominator)


def _validate_requested_overlap(tile_length: int, overlap: int, axis: str) -> None:
    if overlap < 0:
        raise ValueError(f"overlap_{axis} must be nonnegative, got {overlap}.")
    if overlap >= tile_length:
        raise ValueError(
            f"overlap_{axis} must be smaller than tile_{'width' if axis == 'x' else 'height'} "
            f"({tile_length}), got {overlap}."
        )
    if overlap * 2 > tile_length:
        raise ValueError(
            f"overlap_{axis} must not exceed half of tile_"
            f"{'width' if axis == 'x' else 'height'} ({tile_length}), got {overlap}."
        )


def _fixed_count(source_length: int, tile_length: int, overlap: int) -> int:
    return max(1, _ceil_div(source_length - overlap, tile_length - overlap))


def _axis_starts(source_length: int, tile_length: int, overlap: int, count: int) -> tuple[int, ...]:
    if count == 1 or source_length <= tile_length:
        return (0,)

    stride = tile_length - overlap
    final_start = source_length - tile_length
    starts = [min(index * stride, final_start) for index in range(count)]
    starts[-1] = final_start
    if any(current <= previous for previous, current in pairwise(starts)):
        raise ValueError("Resolved tile positions are not strictly increasing.")
    return tuple(starts)


def _axis_overlap(starts: tuple[int, ...], tile_length: int, index: int) -> tuple[int, int]:
    left = 0 if index == 0 else max(0, starts[index - 1] + tile_length - starts[index])
    right = 0 if index == len(starts) - 1 else max(0, starts[index] + tile_length - starts[index + 1])
    return left, right


def build_tile_layout(
    image_shape: tuple[int, ...],
    mode: str,
    tile_width: int,
    tile_height: int,
    overlap_x: int,
    overlap_y: int,
    max_columns: int,
    max_rows: int,
) -> TileLayout:
    if len(image_shape) != 4:
        raise ValueError(f"Image Tile Batch expects rank-4 IMAGE [B,H,W,C], got shape {image_shape}.")

    batch, source_height, source_width, channels = image_shape
    if min(batch, source_height, source_width, channels) <= 0:
        raise ValueError(f"Image Tile Batch requires nonempty dimensions, got shape {image_shape}.")
    if mode not in ("fixed_tile", "bounded_grid"):
        raise ValueError(f"Unsupported tiling mode {mode!r}.")
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("tile_width and tile_height must be positive.")
    if max_columns <= 0 or max_rows <= 0:
        raise ValueError("max_columns and max_rows must be positive.")

    _validate_requested_overlap(tile_width, overlap_x, "x")
    _validate_requested_overlap(tile_height, overlap_y, "y")

    preferred_columns = _fixed_count(source_width, tile_width, overlap_x)
    preferred_rows = _fixed_count(source_height, tile_height, overlap_y)

    if mode == "fixed_tile":
        columns = preferred_columns
        rows = preferred_rows
        exact_width = tile_width
        exact_height = tile_height
    else:
        columns = min(preferred_columns, max_columns)
        rows = min(preferred_rows, max_rows)
        exact_width = _ceil_div(source_width + overlap_x * (columns - 1), columns)
        exact_height = _ceil_div(source_height + overlap_y * (rows - 1), rows)
        _validate_requested_overlap(exact_width, overlap_x, "x")
        _validate_requested_overlap(exact_height, overlap_y, "y")

    x_starts = _axis_starts(source_width, exact_width, overlap_x, columns)
    y_starts = _axis_starts(source_height, exact_height, overlap_y, rows)
    if len(x_starts) != columns or len(y_starts) != rows:
        raise ValueError("Resolved grid count does not match its tile positions.")

    records: list[SpatialTileRecord] = []
    for row, y0 in enumerate(y_starts):
        source_y1 = min(source_height, y0 + exact_height)
        top, bottom = _axis_overlap(y_starts, exact_height, row)
        for column, x0 in enumerate(x_starts):
            source_x1 = min(source_width, x0 + exact_width)
            left, right = _axis_overlap(x_starts, exact_width, column)
            valid_width = source_x1 - x0
            valid_height = source_y1 - y0
            records.append(
                SpatialTileRecord(
                    row=row,
                    column=column,
                    source_rect=Rect(x0, y0, source_x1, source_y1),
                    valid_rect=Rect(0, 0, valid_width, valid_height),
                    neighbor_overlaps=NeighborOverlaps(left, top, right, bottom),
                )
            )

    requested = RequestedTileParameters(
        mode=mode,
        tile_width=tile_width,
        tile_height=tile_height,
        overlap_x=overlap_x,
        overlap_y=overlap_y,
        max_columns=max_columns,
        max_rows=max_rows,
    )
    layout = TileLayout(
        version=LAYOUT_VERSION,
        source_batch_size=batch,
        source_height=source_height,
        source_width=source_width,
        channels=channels,
        tile_tensor_height=exact_height,
        tile_tensor_width=exact_width,
        rows=rows,
        columns=columns,
        geometry_mode=mode,
        requested_parameters=requested,
        spatial_records=tuple(records),
    )
    validate_tile_layout(layout)
    return layout


def validate_tile_layout(layout: TileLayout) -> None:
    if not isinstance(layout, TileLayout):
        raise TypeError(f"layout must be a TileLayout, got {type(layout).__name__}.")
    if layout.version != LAYOUT_VERSION:
        raise ValueError(f"Unsupported TILE_LAYOUT version {layout.version}; expected {LAYOUT_VERSION}.")
    if layout.order != LAYOUT_ORDER:
        raise ValueError(f"Unsupported TILE_LAYOUT order {layout.order!r}.")
    if layout.merge_policy not in SUPPORTED_LAYOUT_POLICIES:
        raise ValueError(f"Unsupported TILE_LAYOUT compatibility policy {layout.merge_policy!r}.")
    if layout.geometry_mode not in ("fixed_tile", "bounded_grid"):
        raise ValueError(f"Invalid layout geometry mode {layout.geometry_mode!r}.")
    if layout.requested_parameters.mode != layout.geometry_mode:
        raise ValueError("Layout mode disagrees with requested parameters.")
    if min(
        layout.source_batch_size,
        layout.source_height,
        layout.source_width,
        layout.channels,
        layout.tile_tensor_height,
        layout.tile_tensor_width,
        layout.rows,
        layout.columns,
    ) <= 0:
        raise ValueError("TILE_LAYOUT dimensions and counts must be positive.")
    if len(layout.spatial_records) != layout.rows * layout.columns:
        raise ValueError("TILE_LAYOUT spatial record count does not match rows * columns.")

    records_by_position: dict[tuple[int, int], SpatialTileRecord] = {}
    for index, record in enumerate(layout.spatial_records):
        expected_position = divmod(index, layout.columns)
        if (record.row, record.column) != expected_position:
            raise ValueError("TILE_LAYOUT records are not in row-major order.")
        if (record.row, record.column) in records_by_position:
            raise ValueError("TILE_LAYOUT contains a duplicate row/column record.")
        records_by_position[(record.row, record.column)] = record

        source = record.source_rect
        valid = record.valid_rect
        if not (0 <= source.x0 < source.x1 <= layout.source_width):
            raise ValueError(f"Tile {index} source X rectangle is out of bounds.")
        if not (0 <= source.y0 < source.y1 <= layout.source_height):
            raise ValueError(f"Tile {index} source Y rectangle is out of bounds.")
        if not (0 <= valid.x0 < valid.x1 <= layout.tile_tensor_width):
            raise ValueError(f"Tile {index} valid X rectangle is out of bounds.")
        if not (0 <= valid.y0 < valid.y1 <= layout.tile_tensor_height):
            raise ValueError(f"Tile {index} valid Y rectangle is out of bounds.")
        if source.width != valid.width or source.height != valid.height:
            raise ValueError(f"Tile {index} source and valid rectangle extents disagree.")
        if valid.x0 != 0 or valid.y0 != 0:
            raise ValueError(f"Tile {index} valid rectangle must begin at the tile origin in v1.")

        overlaps = record.neighbor_overlaps
        if min(overlaps.left, overlaps.top, overlaps.right, overlaps.bottom) < 0:
            raise ValueError(f"Tile {index} has a negative neighbor overlap.")

    for record in layout.spatial_records:
        source = record.source_rect
        overlaps = record.neighbor_overlaps
        left = 0
        right = 0
        top = 0
        bottom = 0
        if record.column > 0:
            neighbor = records_by_position[(record.row, record.column - 1)].source_rect
            left = max(0, min(source.x1, neighbor.x1) - max(source.x0, neighbor.x0))
        if record.column + 1 < layout.columns:
            neighbor = records_by_position[(record.row, record.column + 1)].source_rect
            right = max(0, min(source.x1, neighbor.x1) - max(source.x0, neighbor.x0))
        if record.row > 0:
            neighbor = records_by_position[(record.row - 1, record.column)].source_rect
            top = max(0, min(source.y1, neighbor.y1) - max(source.y0, neighbor.y0))
        if record.row + 1 < layout.rows:
            neighbor = records_by_position[(record.row + 1, record.column)].source_rect
            bottom = max(0, min(source.y1, neighbor.y1) - max(source.y0, neighbor.y0))
        if overlaps != NeighborOverlaps(left, top, right, bottom):
            raise ValueError(
                f"Tile ({record.row}, {record.column}) recorded neighbor overlaps do not match its rectangles."
            )


def tile_image_batch(image: torch.Tensor, layout: TileLayout) -> torch.Tensor:
    validate_tile_layout(layout)
    if tuple(image.shape) != (
        layout.source_batch_size,
        layout.source_height,
        layout.source_width,
        layout.channels,
    ):
        raise ValueError("Input image shape does not match TILE_LAYOUT source dimensions.")

    spatial_count = len(layout.spatial_records)
    tiles = torch.empty(
        (
            layout.required_tile_count,
            layout.tile_tensor_height,
            layout.tile_tensor_width,
            layout.channels,
        ),
        device=image.device,
        dtype=image.dtype,
    )

    for source_index in range(layout.source_batch_size):
        for spatial_index, record in enumerate(layout.spatial_records):
            output_index = source_index * spatial_count + spatial_index
            source = record.source_rect
            valid = record.valid_rect
            tile = tiles[output_index]
            source_slice = image[source_index, source.y0:source.y1, source.x0:source.x1, :]
            tile[valid.y0:valid.y1, valid.x0:valid.x1, :].copy_(source_slice)

            if valid.x1 < layout.tile_tensor_width:
                tile[:valid.y1, valid.x1:, :].copy_(
                    tile[:valid.y1, valid.x1 - 1:valid.x1, :].expand(
                        valid.height, layout.tile_tensor_width - valid.x1, layout.channels
                    )
                )
            if valid.y1 < layout.tile_tensor_height:
                tile[valid.y1:, :, :].copy_(
                    tile[valid.y1 - 1:valid.y1, :, :].expand(
                        layout.tile_tensor_height - valid.y1, layout.tile_tensor_width, layout.channels
                    )
                )

    return tiles


def _linear_ramp(length: int, ascending: bool, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if length <= 0:
        return torch.empty((0,), device=device, dtype=dtype)
    values = torch.arange(1, length + 1, device=device, dtype=dtype) / (length + 1)
    return values if ascending else values.flip(0)


def _tile_weight(
    record: SpatialTileRecord,
    device: torch.device,
    dtype: torch.dtype,
    ramp_cache: dict[tuple[int, bool], torch.Tensor],
) -> torch.Tensor:
    height = record.valid_rect.height
    width = record.valid_rect.width
    y_weight = torch.ones((height,), device=device, dtype=dtype)
    x_weight = torch.ones((width,), device=device, dtype=dtype)
    overlaps = record.neighbor_overlaps

    def ramp(length: int, ascending: bool) -> torch.Tensor:
        key = (length, ascending)
        if key not in ramp_cache:
            ramp_cache[key] = _linear_ramp(length, ascending, device, dtype)
        return ramp_cache[key]

    if overlaps.top:
        y_weight[:overlaps.top] *= ramp(overlaps.top, True)
    if overlaps.bottom:
        y_weight[-overlaps.bottom:] *= ramp(overlaps.bottom, False)
    if overlaps.left:
        x_weight[:overlaps.left] *= ramp(overlaps.left, True)
    if overlaps.right:
        x_weight[-overlaps.right:] *= ramp(overlaps.right, False)
    return y_weight[:, None, None] * x_weight[None, :, None]


def _validate_tiles(tiles: torch.Tensor, layout: TileLayout) -> None:
    validate_tile_layout(layout)
    if not isinstance(tiles, torch.Tensor):
        raise TypeError(f"tiles must be a torch.Tensor, got {type(tiles).__name__}.")
    if tiles.ndim != 4:
        raise ValueError(f"Image Untile Batch expects rank-4 IMAGE [N,H,W,C], got shape {tuple(tiles.shape)}.")
    if tiles.shape[0] != layout.required_tile_count:
        raise ValueError(
            f"Tile count mismatch: layout requires {layout.required_tile_count}, got {tiles.shape[0]}."
        )
    if tiles.shape[1] != layout.tile_tensor_height:
        raise ValueError(
            f"Tile height mismatch: layout requires {layout.tile_tensor_height}, got {tiles.shape[1]}."
        )
    if tiles.shape[2] != layout.tile_tensor_width:
        raise ValueError(
            f"Tile width mismatch: layout requires {layout.tile_tensor_width}, got {tiles.shape[2]}."
        )
    if tiles.shape[3] != layout.channels:
        raise ValueError(f"Tile channel mismatch: layout requires {layout.channels}, got {tiles.shape[3]}.")
    if not tiles.is_floating_point():
        raise TypeError(f"Image Untile Batch requires floating-point IMAGE tiles, got {tiles.dtype}.")


def _untile_linear(tiles: torch.Tensor, layout: TileLayout) -> torch.Tensor:
    accumulation_dtype = torch.float32 if tiles.dtype in (torch.float16, torch.bfloat16) else tiles.dtype
    accumulator = torch.zeros(
        (layout.source_batch_size, layout.source_height, layout.source_width, layout.channels),
        device=tiles.device,
        dtype=accumulation_dtype,
    )
    weight_sum = torch.zeros(
        (1, layout.source_height, layout.source_width, 1),
        device=tiles.device,
        dtype=accumulation_dtype,
    )
    ramp_cache: dict[tuple[int, bool], torch.Tensor] = {}

    for spatial_index, record in enumerate(layout.spatial_records):
        source = record.source_rect
        valid = record.valid_rect
        weight = _tile_weight(record, tiles.device, accumulation_dtype, ramp_cache)
        weight_sum[:, source.y0:source.y1, source.x0:source.x1, :] += weight.unsqueeze(0)

        for source_index in range(layout.source_batch_size):
            tile_index = source_index * len(layout.spatial_records) + spatial_index
            tile = tiles[tile_index, valid.y0:valid.y1, valid.x0:valid.x1, :]
            accumulator[source_index, source.y0:source.y1, source.x0:source.x1, :] += (
                tile.to(accumulation_dtype) * weight
            )

    if not bool(torch.all(weight_sum > 0).item()):
        raise ValueError("TILE_LAYOUT leaves at least one source pixel with zero accumulated weight.")

    result = accumulator / weight_sum
    return result.to(tiles.dtype)


def _overlap_midpoint(earlier: Rect, later: Rect, axis: str) -> int:
    if axis == "x":
        overlap_start = later.x0
        overlap_end = earlier.x1
    else:
        overlap_start = later.y0
        overlap_end = earlier.y1
    overlap_length = overlap_end - overlap_start
    if overlap_length < 0:
        raise ValueError(f"Adjacent TILE_LAYOUT records have a gap on axis {axis}.")
    return overlap_start + overlap_length // 2


def _hard_cut_ownership(layout: TileLayout) -> tuple[Rect, ...]:
    records = layout.spatial_records
    ownership: list[Rect] = []
    for record in records:
        source = record.source_rect
        if record.column == 0:
            x0 = 0
        else:
            left = records[record.row * layout.columns + record.column - 1].source_rect
            x0 = _overlap_midpoint(left, source, "x")
        if record.column + 1 == layout.columns:
            x1 = layout.source_width
        else:
            right = records[record.row * layout.columns + record.column + 1].source_rect
            x1 = _overlap_midpoint(source, right, "x")
        if record.row == 0:
            y0 = 0
        else:
            top = records[(record.row - 1) * layout.columns + record.column].source_rect
            y0 = _overlap_midpoint(top, source, "y")
        if record.row + 1 == layout.rows:
            y1 = layout.source_height
        else:
            bottom = records[(record.row + 1) * layout.columns + record.column].source_rect
            y1 = _overlap_midpoint(source, bottom, "y")

        owned = Rect(x0, y0, x1, y1)
        if not (
            source.x0 <= owned.x0 < owned.x1 <= source.x1
            and source.y0 <= owned.y0 < owned.y1 <= source.y1
        ):
            raise ValueError(
                f"Hard-cut ownership for tile ({record.row}, {record.column}) is outside its source rectangle."
            )
        ownership.append(owned)

    for row in range(layout.rows):
        row_rects = ownership[row * layout.columns:(row + 1) * layout.columns]
        if row_rects[0].x0 != 0 or row_rects[-1].x1 != layout.source_width:
            raise ValueError("Hard-cut ownership does not cover the source width.")
        if any(left.x1 != right.x0 for left, right in pairwise(row_rects)):
            raise ValueError("Hard-cut horizontal ownership has a gap or overlap.")
    for column in range(layout.columns):
        column_rects = ownership[column::layout.columns]
        if column_rects[0].y0 != 0 or column_rects[-1].y1 != layout.source_height:
            raise ValueError("Hard-cut ownership does not cover the source height.")
        if any(top.y1 != bottom.y0 for top, bottom in pairwise(column_rects)):
            raise ValueError("Hard-cut vertical ownership has a gap or overlap.")

    column_bounds = [(ownership[column].x0, ownership[column].x1) for column in range(layout.columns)]
    row_bounds = [
        (ownership[row * layout.columns].y0, ownership[row * layout.columns].y1)
        for row in range(layout.rows)
    ]
    for record, owned in zip(records, ownership):
        if (owned.x0, owned.x1) != column_bounds[record.column]:
            raise ValueError("Hard-cut X ownership is inconsistent across grid rows.")
        if (owned.y0, owned.y1) != row_bounds[record.row]:
            raise ValueError("Hard-cut Y ownership is inconsistent across grid columns.")
    return tuple(ownership)


def _untile_hard_cut(tiles: torch.Tensor, layout: TileLayout) -> torch.Tensor:
    ownership = _hard_cut_ownership(layout)
    result = torch.empty(
        (layout.source_batch_size, layout.source_height, layout.source_width, layout.channels),
        device=tiles.device,
        dtype=tiles.dtype,
    )
    spatial_count = len(layout.spatial_records)
    for source_index in range(layout.source_batch_size):
        for spatial_index, (record, owned) in enumerate(zip(layout.spatial_records, ownership)):
            source = record.source_rect
            valid = record.valid_rect
            tile_x0 = valid.x0 + owned.x0 - source.x0
            tile_y0 = valid.y0 + owned.y0 - source.y0
            tile_x1 = tile_x0 + owned.width
            tile_y1 = tile_y0 + owned.height
            if not (
                valid.x0 <= tile_x0 < tile_x1 <= valid.x1
                and valid.y0 <= tile_y0 < tile_y1 <= valid.y1
            ):
                raise ValueError(
                    f"Hard-cut ownership for tile ({record.row}, {record.column}) reaches padded pixels."
                )
            tile_index = source_index * spatial_count + spatial_index
            result[source_index, owned.y0:owned.y1, owned.x0:owned.x1, :].copy_(
                tiles[tile_index, tile_y0:tile_y1, tile_x0:tile_x1, :]
            )
    return result


def _limited_band(earlier: Rect, later: Rect, axis: str, blend_width: int) -> tuple[int, int]:
    if axis == "x":
        overlap_start = later.x0
        overlap_end = earlier.x1
    else:
        overlap_start = later.y0
        overlap_end = earlier.y1
    overlap_length = overlap_end - overlap_start
    if overlap_length < 0:
        raise ValueError(f"Adjacent TILE_LAYOUT records have a gap on axis {axis}.")
    effective_width = min(blend_width, overlap_length)
    cut = _overlap_midpoint(earlier, later, axis)
    left_half = effective_width // 2
    right_half = effective_width - left_half
    blend_start = cut - left_half
    blend_end = cut + right_half
    if blend_start < overlap_start or blend_end > overlap_end:
        raise ValueError(f"Limited-linear blend band is outside the overlap on axis {axis}.")
    return blend_start, blend_end


def _limited_axis_weight(
    record: SpatialTileRecord,
    layout: TileLayout,
    axis: str,
    blend_width: int,
    device: torch.device,
    dtype: torch.dtype,
    ramp_cache: dict[tuple[int, bool], torch.Tensor],
) -> torch.Tensor:
    source = record.source_rect
    if axis == "x":
        length = source.width
        start = source.x0
        before_index = record.row * layout.columns + record.column - 1
        after_index = record.row * layout.columns + record.column + 1
        has_before = record.column > 0
        has_after = record.column + 1 < layout.columns
    else:
        length = source.height
        start = source.y0
        before_index = (record.row - 1) * layout.columns + record.column
        after_index = (record.row + 1) * layout.columns + record.column
        has_before = record.row > 0
        has_after = record.row + 1 < layout.rows

    weight = torch.ones((length,), device=device, dtype=dtype)

    def ramp(ramp_length: int, ascending: bool) -> torch.Tensor:
        key = (ramp_length, ascending)
        if key not in ramp_cache:
            ramp_cache[key] = _linear_ramp(ramp_length, ascending, device, dtype)
        return ramp_cache[key]

    if has_before:
        earlier = layout.spatial_records[before_index].source_rect
        blend_start, blend_end = _limited_band(earlier, source, axis, blend_width)
        local_start = blend_start - start
        local_end = blend_end - start
        weight[:local_start] = 0
        if local_end > local_start:
            weight[local_start:local_end] = ramp(local_end - local_start, True)
    if has_after:
        later = layout.spatial_records[after_index].source_rect
        blend_start, blend_end = _limited_band(source, later, axis, blend_width)
        local_start = blend_start - start
        local_end = blend_end - start
        if local_end > local_start:
            weight[local_start:local_end] *= ramp(local_end - local_start, False)
        weight[local_end:] = 0
    return weight


def _untile_limited_linear(tiles: torch.Tensor, layout: TileLayout, blend_width: int) -> torch.Tensor:
    if blend_width == 0:
        return _untile_hard_cut(tiles, layout)

    _hard_cut_ownership(layout)
    accumulation_dtype = torch.float32 if tiles.dtype in (torch.float16, torch.bfloat16) else tiles.dtype
    accumulator = torch.zeros(
        (layout.source_batch_size, layout.source_height, layout.source_width, layout.channels),
        device=tiles.device,
        dtype=accumulation_dtype,
    )
    weight_sum = torch.zeros(
        (1, layout.source_height, layout.source_width, 1),
        device=tiles.device,
        dtype=accumulation_dtype,
    )
    ramp_cache: dict[tuple[int, bool], torch.Tensor] = {}
    spatial_count = len(layout.spatial_records)

    for spatial_index, record in enumerate(layout.spatial_records):
        source = record.source_rect
        valid = record.valid_rect
        x_weight = _limited_axis_weight(
            record, layout, "x", blend_width, tiles.device, accumulation_dtype, ramp_cache
        )
        y_weight = _limited_axis_weight(
            record, layout, "y", blend_width, tiles.device, accumulation_dtype, ramp_cache
        )
        weight = y_weight[:, None, None] * x_weight[None, :, None]
        weight_sum[:, source.y0:source.y1, source.x0:source.x1, :] += weight.unsqueeze(0)

        for source_index in range(layout.source_batch_size):
            tile_index = source_index * spatial_count + spatial_index
            tile = tiles[tile_index, valid.y0:valid.y1, valid.x0:valid.x1, :]
            accumulator[source_index, source.y0:source.y1, source.x0:source.x1, :] += (
                tile.to(accumulation_dtype) * weight
            )

    if not bool(torch.all(weight_sum > 0).item()):
        raise ValueError("TILE_LAYOUT leaves at least one source pixel with zero accumulated weight.")
    return (accumulator / weight_sum).to(tiles.dtype)


def untile_image_batch(
    tiles: torch.Tensor,
    layout: TileLayout,
    merge_mode: str = "linear",
    blend_width: int = 32,
) -> torch.Tensor:
    _validate_tiles(tiles, layout)
    if merge_mode == "linear":
        return _untile_linear(tiles, layout)
    if merge_mode == "hard_cut":
        return _untile_hard_cut(tiles, layout)
    if merge_mode == "limited_linear":
        if blend_width < 0:
            raise ValueError(f"blend_width must be nonnegative, got {blend_width}.")
        return _untile_limited_linear(tiles, layout, blend_width)
    raise ValueError(
        f"Unsupported merge_mode {merge_mode!r}; expected 'linear', 'hard_cut', or 'limited_linear'."
    )


class ImageTileBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteImageTileBatch",
            display_name="Image Tile Batch",
            category="Utility Suite/Image/Tiling",
            description="Splits an IMAGE into a source-major ordinary IMAGE batch with explicit layout metadata.",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("mode", options=["fixed_tile", "bounded_grid"]),
                io.Int.Input("tile_width", default=1024, min=1, max=32768),
                io.Int.Input("tile_height", default=1024, min=1, max=32768),
                io.Int.Input("overlap_x", default=128, min=0, max=16384),
                io.Int.Input("overlap_y", default=128, min=0, max=16384),
                io.Int.Input("max_columns", default=3, min=1, max=256),
                io.Int.Input("max_rows", default=3, min=1, max=256),
            ],
            outputs=[
                io.Image.Output(display_name="tiles"),
                TILE_LAYOUT.Output(display_name="layout"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        mode: str,
        tile_width: int,
        tile_height: int,
        overlap_x: int,
        overlap_y: int,
        max_columns: int,
        max_rows: int,
    ) -> io.NodeOutput:
        layout = build_tile_layout(
            tuple(image.shape),
            mode,
            tile_width,
            tile_height,
            overlap_x,
            overlap_y,
            max_columns,
            max_rows,
        )
        return io.NodeOutput(tile_image_batch(image, layout), layout)


class ImageUntileBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteImageUntileBatch",
            display_name="Image Untile Batch",
            category="Utility Suite/Image/Tiling",
            description="Reconstructs an IMAGE batch from tiles using the authoritative TILE_LAYOUT.",
            inputs=[
                io.Image.Input("tiles"),
                TILE_LAYOUT.Input("layout"),
                io.Combo.Input("merge_mode", options=["linear", "hard_cut", "limited_linear"], default="linear"),
                io.Int.Input("blend_width", default=32, min=0, max=32768),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(
        cls,
        tiles: torch.Tensor,
        layout: TileLayout,
        merge_mode: str = "linear",
        blend_width: int = 32,
    ) -> io.NodeOutput:
        return io.NodeOutput(untile_image_batch(tiles, layout, merge_mode, blend_width))


class UtilitySuiteTilingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ImageTileBatch, ImageUntileBatch]


async def comfy_entrypoint() -> UtilitySuiteTilingExtension:
    return UtilitySuiteTilingExtension()
