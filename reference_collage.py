from __future__ import annotations

import math
from dataclasses import dataclass

import comfy.utils
import torch
from comfy_api.latest import io


@dataclass(frozen=True)
class CollageSource:
    index: int
    width: int
    height: int
    weight: float


@dataclass(frozen=True)
class CollagePlacement:
    index: int
    source_width: int
    source_height: int
    cell_x: int
    cell_y: int
    cell_width: int
    cell_height: int
    x: int
    y: int
    width: int
    height: int

    @property
    def scale(self) -> tuple[float, float]:
        return self.width / self.source_width, self.height / self.source_height


@dataclass(frozen=True)
class CollageLayout:
    width: int
    height: int
    placements: tuple[CollagePlacement, ...]


@dataclass(frozen=True)
class _ItemRect:
    index: int
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class _Block:
    width: int
    height: int
    items: tuple[_ItemRect, ...]
    weight_sum: float
    count: int

    @property
    def average_weight(self) -> float:
        return self.weight_sum / self.count


def _round_positive(value: float) -> int:
    return max(1, math.floor(value + 0.5))


def _offset(slack: int, alignment: str) -> int:
    if alignment == "start":
        return 0
    if alignment == "center":
        return slack // 2
    if alignment == "end":
        return slack
    raise ValueError(f"Reference Collage alignment must be start, center, or end, got {alignment!r}.")


def parse_weights(weights: str, count: int) -> tuple[float, ...]:
    if not isinstance(weights, str):
        raise TypeError(f"Reference Collage weights must be a string, got {type(weights).__name__}.")
    parts = [part.strip() for part in weights.split(",") if part.strip()]
    if len(parts) > count:
        raise ValueError(f"Reference Collage received {len(parts)} weights for {count} images.")
    values = []
    for index, part in enumerate(parts):
        try:
            value = float(part)
        except ValueError as error:
            raise ValueError(f"Reference Collage weight {index} is not a number: {part!r}.") from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Reference Collage weight {index} must be positive and finite, got {part!r}.")
        values.append(value)
    values.extend([1.0] * (count - len(values)))
    return tuple(values)


def collect_images(images: dict[str, torch.Tensor], weights: str) -> tuple[list[torch.Tensor], tuple[CollageSource, ...]]:
    if not images:
        raise ValueError("Reference Collage requires image0.")

    def index(name: str) -> int:
        if not name.startswith("image"):
            raise ValueError(f"Reference Collage received invalid image input name {name!r}.")
        try:
            return int(name.removeprefix("image"))
        except ValueError as error:
            raise ValueError(f"Reference Collage received invalid image input name {name!r}.") from error

    ordered_names = sorted(images, key=index)
    ordered = [images[name] for name in ordered_names]
    parsed_weights = parse_weights(weights, len(ordered))
    first = ordered[0]
    for position, (name, image) in enumerate(zip(ordered_names, ordered)):
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] != 1:
            shape = tuple(image.shape) if isinstance(image, torch.Tensor) else type(image).__name__
            raise ValueError(f"Reference Collage {name} must be one IMAGE [1,H,W,C], got {shape}.")
        if image.shape[1] == 0 or image.shape[2] == 0 or image.shape[3] < 3:
            raise ValueError(f"Reference Collage {name} requires nonempty H/W and at least 3 channels.")
        if not image.is_floating_point():
            raise TypeError(f"Reference Collage {name} must use a floating-point IMAGE dtype, got {image.dtype}.")
        if position and image.shape[3] != first.shape[3]:
            raise ValueError(f"Reference Collage {name} has {image.shape[3]} channels; expected {first.shape[3]}.")
        if position and image.dtype != first.dtype:
            raise ValueError(f"Reference Collage {name} has dtype {image.dtype}; expected {first.dtype}.")
        if position and image.device != first.device:
            raise ValueError(f"Reference Collage {name} is on {image.device}; expected {first.device}.")
    sources = tuple(
        CollageSource(index(name), image.shape[2], image.shape[1], weight)
        for name, image, weight in zip(ordered_names, ordered, parsed_weights)
    )
    return ordered, sources


def _scaled_dimensions(width: int, height: int, scale: float) -> tuple[int, int]:
    if width >= height:
        scaled_width = _round_positive(width * scale)
        return scaled_width, _round_positive(height * scaled_width / width)
    scaled_height = _round_positive(height * scale)
    return _round_positive(width * scaled_height / height), scaled_height


def normalize_sources(
    sources: tuple[CollageSource, ...], mode: str, size: int, allow_upscale: bool
) -> tuple[CollageSource, ...]:
    if mode not in {"none", "longest_side", "shortest_side", "area"}:
        raise ValueError(
            "Reference Collage normalize_mode must be none, longest_side, shortest_side, or area, "
            f"got {mode!r}."
        )
    if size < 1:
        raise ValueError("Reference Collage normalize_size must be at least 1.")
    if mode == "none":
        return sources
    normalized = []
    for source in sources:
        if mode == "longest_side":
            scale = size / max(source.width, source.height)
        elif mode == "shortest_side":
            scale = size / min(source.width, source.height)
        else:
            scale = size / math.sqrt(source.width * source.height)
        if not allow_upscale:
            scale = min(scale, 1.0)
        width, height = _scaled_dimensions(source.width, source.height, scale)
        normalized.append(CollageSource(source.index, width, height, source.weight))
    return tuple(normalized)


def apply_weight_sizing(
    sources: tuple[CollageSource, ...], weight_mode: str
) -> tuple[CollageSource, ...]:
    if weight_mode not in {"area", "linear"}:
        raise ValueError(f"Reference Collage weight_mode must be area or linear, got {weight_mode!r}.")
    reference_area = math.exp(sum(math.log(source.width * source.height) for source in sources) / len(sources))
    reference_weight = math.exp(sum(math.log(source.weight) for source in sources) / len(sources))
    weighted = []
    for source in sources:
        relative_weight = source.weight / reference_weight
        target_area = reference_area * (relative_weight if weight_mode == "area" else relative_weight**2)
        aspect_ratio = source.width / source.height
        width = _round_positive(math.sqrt(target_area * aspect_ratio))
        height = _round_positive(width / aspect_ratio)
        weighted.append(CollageSource(source.index, width, height, source.weight))
    return tuple(weighted)


def _grid_sizes(row: tuple[CollageSource, ...], resize_mode: str) -> tuple[tuple[int, int], ...]:
    if resize_mode == "none":
        return tuple((source.width, source.height) for source in row)
    if resize_mode != "auto":
        raise ValueError(f"Reference Collage resize_mode must be none or auto, got {resize_mode!r}.")
    unit = max(source.height / math.sqrt(source.weight) for source in row)
    sizes = []
    for source in row:
        height = _round_positive(unit * math.sqrt(source.weight))
        width = _round_positive(source.width * height / source.height)
        sizes.append((width, height))
    return tuple(sizes)


def calculate_grid_layout(
    sources: tuple[CollageSource, ...], columns: int, gap: int, resize_mode: str, alignment: str
) -> CollageLayout:
    if columns < 1:
        raise ValueError("Reference Collage columns must be at least 1.")
    if gap < 0:
        raise ValueError("Reference Collage gap must be nonnegative.")
    if not sources:
        raise ValueError("Reference Collage requires at least one source.")
    column_count = min(columns, len(sources))
    rows = tuple(sources[start : start + column_count] for start in range(0, len(sources), column_count))
    row_sizes = tuple(_grid_sizes(row, resize_mode) for row in rows)
    column_widths = [0] * column_count
    row_heights = []
    for sizes in row_sizes:
        row_heights.append(max(height for _, height in sizes))
        for column, (width, _) in enumerate(sizes):
            column_widths[column] = max(column_widths[column], width)
    canvas_width = sum(column_widths) + gap * (column_count - 1)
    canvas_height = sum(row_heights) + gap * (len(rows) - 1)
    x_starts = []
    cursor = 0
    for width in column_widths:
        x_starts.append(cursor)
        cursor += width + gap
    placements = []
    y = 0
    for row, sizes, row_height in zip(rows, row_sizes, row_heights):
        for column, (source, (width, height)) in enumerate(zip(row, sizes)):
            cell_width = column_widths[column]
            placements.append(
                CollagePlacement(
                    source.index,
                    source.width,
                    source.height,
                    x_starts[column],
                    y,
                    cell_width,
                    row_height,
                    x_starts[column] + _offset(cell_width - width, alignment),
                    y + _offset(row_height - height, alignment),
                    width,
                    height,
                )
            )
        y += row_height + gap
    return CollageLayout(canvas_width, canvas_height, tuple(placements))


def _split_index(sources: tuple[CollageSource, ...]) -> int:
    total = sum(source.weight for source in sources)
    cumulative = 0.0
    best_index = 1
    best_difference = math.inf
    for index, source in enumerate(sources[:-1], start=1):
        cumulative += source.weight
        difference = abs(total - 2.0 * cumulative)
        if difference <= best_difference:
            best_index = index
            best_difference = difference
    return best_index


def _scale_block(block: _Block, scale: float) -> _Block:
    width = _round_positive(block.width * scale)
    height = _round_positive(block.height * scale)
    items = tuple(
        _ItemRect(
            item.index,
            math.floor(item.x * scale + 0.5),
            math.floor(item.y * scale + 0.5),
            _round_positive(item.width * scale),
            _round_positive(item.height * scale),
        )
        for item in block.items
    )
    return _Block(width, height, items, block.weight_sum, block.count)


def _move_block(block: _Block, x: int, y: int) -> tuple[_ItemRect, ...]:
    return tuple(_ItemRect(item.index, item.x + x, item.y + y, item.width, item.height) for item in block.items)


def _mosaic_block(
    sources: tuple[CollageSource, ...], gap: int, resize_mode: str, alignment: str, vertical: bool
) -> _Block:
    if len(sources) == 1:
        source = sources[0]
        return _Block(
            source.width,
            source.height,
            (_ItemRect(source.index, 0, 0, source.width, source.height),),
            source.weight,
            1,
        )
    split = _split_index(sources)
    first = _mosaic_block(sources[:split], gap, resize_mode, alignment, not vertical)
    second = _mosaic_block(sources[split:], gap, resize_mode, alignment, not vertical)
    if resize_mode == "auto":
        first_preference = math.sqrt(first.average_weight)
        second_preference = math.sqrt(second.average_weight)
        first_cross = first.height if vertical else first.width
        second_cross = second.height if vertical else second.width
        unit = max(first_cross / first_preference, second_cross / second_preference)
        first = _scale_block(first, _round_positive(unit * first_preference) / first_cross)
        second = _scale_block(second, _round_positive(unit * second_preference) / second_cross)
    elif resize_mode != "none":
        raise ValueError(f"Reference Collage resize_mode must be none or auto, got {resize_mode!r}.")

    if vertical:
        width = first.width + gap + second.width
        height = max(first.height, second.height)
        first_xy = (0, _offset(height - first.height, alignment))
        second_xy = (first.width + gap, _offset(height - second.height, alignment))
    else:
        width = max(first.width, second.width)
        height = first.height + gap + second.height
        first_xy = (_offset(width - first.width, alignment), 0)
        second_xy = (_offset(width - second.width, alignment), first.height + gap)
    return _Block(
        width,
        height,
        _move_block(first, *first_xy) + _move_block(second, *second_xy),
        first.weight_sum + second.weight_sum,
        first.count + second.count,
    )


def calculate_mosaic_layout(
    sources: tuple[CollageSource, ...], gap: int, resize_mode: str, alignment: str
) -> CollageLayout:
    if not sources:
        raise ValueError("Reference Collage requires at least one source.")
    if gap < 0:
        raise ValueError("Reference Collage gap must be nonnegative.")
    block = _mosaic_block(sources, gap, resize_mode, alignment, True)
    source_by_index = {source.index: source for source in sources}
    placements = tuple(
        CollagePlacement(
            item.index,
            source_by_index[item.index].width,
            source_by_index[item.index].height,
            item.x,
            item.y,
            item.width,
            item.height,
            item.x,
            item.y,
            item.width,
            item.height,
        )
        for item in sorted(block.items, key=lambda item: item.index)
    )
    return CollageLayout(block.width, block.height, placements)


def calculate_layout(
    sources: tuple[CollageSource, ...],
    layout: str,
    columns: int,
    gap: int,
    resize_mode: str,
    alignment: str,
    frame_gap: bool = False,
    sizes_are_final: bool = False,
) -> CollageLayout:
    if resize_mode not in {"none", "auto"}:
        raise ValueError(f"Reference Collage resize_mode must be none or auto, got {resize_mode!r}.")
    effective_resize_mode = "none" if sizes_are_final else resize_mode
    if layout == "grid":
        calculated = calculate_grid_layout(sources, columns, gap, effective_resize_mode, alignment)
    elif layout == "mosaic":
        calculated = calculate_mosaic_layout(sources, gap, effective_resize_mode, alignment)
    else:
        raise ValueError(f"Reference Collage layout must be grid or mosaic, got {layout!r}.")
    if not frame_gap or gap == 0:
        return calculated
    placements = tuple(
        CollagePlacement(
            item.index,
            item.source_width,
            item.source_height,
            item.cell_x + gap,
            item.cell_y + gap,
            item.cell_width,
            item.cell_height,
            item.x + gap,
            item.y + gap,
            item.width,
            item.height,
        )
        for item in calculated.placements
    )
    return CollageLayout(calculated.width + gap * 2, calculated.height + gap * 2, placements)


def _background_rgb(background: str) -> tuple[float, float, float]:
    if not isinstance(background, str) or len(background) != 7 or not background.startswith("#"):
        raise ValueError(f"Reference Collage background must be #RRGGBB, got {background!r}.")
    try:
        return tuple(int(background[index : index + 2], 16) / 255.0 for index in (1, 3, 5))
    except ValueError as error:
        raise ValueError(f"Reference Collage background must be #RRGGBB, got {background!r}.") from error


def render_collage(images: list[torch.Tensor], layout: CollageLayout, background: str) -> torch.Tensor:
    first = images[0]
    channels = first.shape[3]
    canvas = torch.zeros((1, layout.height, layout.width, channels), dtype=first.dtype, device=first.device)
    rgb = torch.tensor(_background_rgb(background), dtype=first.dtype, device=first.device)
    canvas[..., :3] = rgb
    if channels >= 4:
        canvas[..., 3] = 1
    by_index = {placement.index: placement for placement in layout.placements}
    for image, placement in zip(images, sorted(layout.placements, key=lambda item: item.index)):
        if image.shape[2] == placement.width and image.shape[1] == placement.height:
            resized = image
        else:
            resized = comfy.utils.common_upscale(
                image.movedim(-1, 1), placement.width, placement.height, "lanczos", "disabled"
            ).movedim(1, -1)
        target = by_index[placement.index]
        canvas[:, target.y : target.y + target.height, target.x : target.x + target.width, :] = resized
    return canvas


def reference_collage(
    images: dict[str, torch.Tensor],
    layout: str,
    columns: int,
    weights: str,
    gap: int,
    resize_mode: str,
    alignment: str,
    background: str,
    frame_gap: bool = False,
    normalize_mode: str = "none",
    normalize_size: int = 512,
    allow_upscale: bool = True,
    weight_mode: str = "area",
) -> tuple[torch.Tensor, CollageLayout]:
    ordered, sources = collect_images(images, weights)
    normalized = normalize_sources(sources, normalize_mode, normalize_size, allow_upscale)
    normalized = apply_weight_sizing(normalized, weight_mode)
    calculated = calculate_layout(
        normalized, layout, columns, gap, resize_mode, alignment, frame_gap, True
    )
    return render_collage(ordered, calculated, background), calculated


class ReferenceCollage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        images = io.Autogrow.TemplatePrefix(io.Image.Input("image"), prefix="image", min=1, max=100)
        return io.Schema(
            node_id="UtilitySuiteReferenceCollage",
            display_name="Reference Collage",
            category="Utility Suite/Image",
            description="Builds a content-sized grid or deterministic weighted mosaic from singleton reference images.",
            inputs=[
                io.Autogrow.Input("images", template=images),
                io.Combo.Input("layout", options=["grid", "mosaic"], default="grid"),
                io.Int.Input("columns", default=2, min=1, max=100, step=1),
                io.String.Input("weights", default="1"),
                io.Int.Input("gap", default=0, min=0, max=32768, step=1),
                io.Combo.Input("resize_mode", options=["none", "auto"], default="none"),
                io.Combo.Input("alignment", options=["start", "center", "end"], default="start"),
                io.Color.Input("background", default="#000000"),
                io.Boolean.Input("frame_gap", default=False),
                io.Combo.Input(
                    "normalize_mode", options=["none", "longest_side", "shortest_side", "area"], default="none"
                ),
                io.Int.Input("normalize_size", default=512, min=1, max=32768, step=1),
                io.Boolean.Input("allow_upscale", default=True),
                io.Combo.Input("weight_mode", options=["area", "linear"], default="area"),
            ],
            outputs=[io.Image.Output("collage")],
        )

    @classmethod
    def execute(
        cls,
        images: io.Autogrow.Type,
        layout: str,
        columns: int,
        weights: str,
        gap: int,
        resize_mode: str,
        alignment: str,
        background: str,
        frame_gap: bool,
        normalize_mode: str,
        normalize_size: int,
        allow_upscale: bool,
        weight_mode: str,
    ) -> io.NodeOutput:
        collage, _ = reference_collage(
            images,
            layout,
            columns,
            weights,
            gap,
            resize_mode,
            alignment,
            background,
            frame_gap,
            normalize_mode,
            normalize_size,
            allow_upscale,
            weight_mode,
        )
        return io.NodeOutput(collage)


__all__ = [
    "CollageLayout",
    "CollagePlacement",
    "CollageSource",
    "ReferenceCollage",
    "apply_weight_sizing",
    "calculate_grid_layout",
    "calculate_layout",
    "calculate_mosaic_layout",
    "collect_images",
    "normalize_sources",
    "parse_weights",
    "reference_collage",
    "render_collage",
]
