# Phase 5 -- traces: matrix scale and halo

> Part of the snapshot-xrd doc set. Spine: [`README.md`](README.md).

- **Matrix scale** per window: one isotropic scale of the matrix lines from ring centroids
  (`fit_matrix_scale`), median over usable rings. It rises with temperature and with any uniform
  tensile stress; the two are indistinguishable from peak positions alone. Report it as a relative
  thermometer with that caveat, and the absolute value with the geometry caveat.
- Near an event the matrix spans a range of scales inside one frame; the per-window median is a
  mixture there.
- **Halo index**: mean counts in a diffuse band minus a quiet band, both clear of every ring. It
  marks melt / amorphous states; repeated events show as repeated excursions.
- Use means, not medians, for few-count profiles (medians rail at 0 and 1).
