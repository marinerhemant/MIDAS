# Phase 0 -- survey the raw frames

> Part of the snapshot-xrd doc set. Spine: [`README.md`](README.md).

Look before modelling. Nothing in later phases replaces this.

1. **Frame list and order.** Sort by the trailing integer of the file name, not lexically. Note the
   frame rate and exposure from the acquisition log.
2. **Per-frame statistics over the whole series:** total counts, maximum pixel, number of pixels
   above a few thresholds. Events (melting, shots, ramps) show as steps; a frame with high total
   counts and a low maximum pixel is diffuse (liquid, amorphous).
3. **A few frames at the key times**, log scale, no overlays. Count spots by eye.
4. **A max projection** over the series shows which rings exist at all, but it is dominated by the
   brightest (often hottest) frames: do not read lattice parameters off it.
5. **The calibrant frame**, raw, no overlay. If it looks empty, fix the display, not the conclusion.
6. **Decide scope:** if the rings are smooth in single frames, lineouts work and the
   `calibrate-integrate` / `xrd-ct` skills apply instead.
