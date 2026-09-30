# Phase 1 -- geometry

> Part of the snapshot-xrd doc set. Spine: [`README.md`](README.md).

1. **Calibrate** on a calibrant frame at the same detector distance (`calibrate-integrate` skill).
   A mixed calibrant is fine if the calibration supports several phases and drops overlapping rings.
2. **Orientation.** The raw array may need a flip to match the geometry. Decide it with a radial
   profile of the calibrant in each candidate orientation: the right one gives sharp peaks on the
   predicted positions, the wrong ones give smeared, offset profiles. Never decide it from ring
   contours drawn over an image (thin contours can cover thin rings).
3. **What the geometry must deliver here.** Candidate tests use ratios, angles and a free scale, so a
   small distance or wavelength error acts like a uniform scale and is largely absorbed. Absolute
   lattice parameters (and therefore temperatures) inherit the full geometry error; state it.
4. **Internal check:** the matrix lattice parameter on frames taken before any event should match
   its expected room-temperature value within the stated geometry uncertainty.
5. **Mask:** detector gaps and dead pixels (raw sentinels), shadows, and anything fixed in the
   frame. Keep a margin from invalid pixels for spot detection (`margin_px`).
