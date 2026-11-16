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
