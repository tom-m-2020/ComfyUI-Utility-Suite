Tiling and inpainting/outpainting logics.

---

Recommended addition:
- Impact Pack

## SEGS Sampler Custom

`SEGS Sampler Custom` enhances Impact-compatible SEGS with ComfyUI's separate
`NOISE`, `GUIDER`, `SAMPLER`, and `SIGMAS` components. `sequential` processes
each crop from the progressively updated canvas. `individual` enhances stored
or original crops independently, then pastes them in SEG order.

Widget defaults are `guide_size=512`, `guide_size_for=bbox`,
`scale_to_guide_size=upscale`, `max_size=1024`, `force_inpaint=enabled`,
`feather=5`, `noise_mask=enabled`, `noise_mask_feather=20`,
`tiled_encode=disabled`, `tiled_decode=disabled`, and `operation=sequential`.

## Power LoRA Loader

`Power LoRA Loader` applies an ordered set of enabled LoRAs to a `MODEL`
through ComfyUI's native model-only LoRA loader. Use `+ Add LoRA` to search the
configured LoRA folders, then enable, set the signed strength, reorder, or
remove each row in the node.

## Seed Generator

`Seed Generator` outputs either a fixed seed or a fresh server-generated seed
for every queue. Its compact controls can enable per-queue randomization,
create a new fixed random seed, or reuse the last seed reported by execution.

## Dimensions Percentage to Pixels

`Dimensions Percentage to Pixels` returns the original dimensions and new
dimensions scaled by an integer percentage, rounded to the nearest requested
multiple. Width, height, and percent may be zero; percent has no maximum.

## Image And Mask Crop By Mask

`Image And Mask Crop By Mask` crops an optional image, an optional target mask,
or both to the per-item bounds of a required crop mask. At least one optional
target must be connected, and all connected tensors must share one canvas.

## Reference Collage

`Reference Collage` accepts auto-growing singleton `image0`, `image1`, ...
sockets and creates either a content-sized grid or deterministic weighted
guillotine mosaic. Images preserve aspect ratio, socket order, dtype, device,
and channels. Normalization can retain source dimensions or establish a common
longest side, shortest side, or approximate pixel area before weighted sizing.
`area` weights target displayed-area ratios directly; `linear` weights target
linear scale ratios, so their displayed areas vary with the square of the
weights. A geometric-mean reference keeps weights relative rather than making
equal non-unit weights enlarge every image. Integer dimensions can introduce
small ratio errors.

Normalized and weighted dimensions are authoritative layout targets. Both
`resize_mode` values remain accepted, but no second per-image auto-resize is
applied after weighted sizing because that would invalidate the requested area
ratios. The original low-level `none`/`auto` solver behavior remains available
to direct callers. Tensors are resized only once, during final compositing.
Gaps are internal by default; `frame_gap` also applies the selected gap to all
four outer canvas edges. `columns` applies to grid mode; mosaic uses its
deterministic recursive splits.
