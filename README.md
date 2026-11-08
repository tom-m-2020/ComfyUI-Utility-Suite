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
