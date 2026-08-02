# midas-plotting

Standard plots for MIDAS reconstructions.

```python
from midas_plotting import read_mic, orientation_map, compare_maps

m = read_mic("Ce5Y_mr.2.mic")
print(m.summary())
orientation_map(m, space_group=225, cmin=0.3)
```

```bash
midas-plot Ce5Y.0.mic Ce5Y_sum3thr2.0.mic --kind orientation --cmin 0.3 \
    --titles "baseline|sum3+thr2" -o compare.png
```

## Why

IPF colouring, `.mic` parsing and map plotting had been re-implemented in
several one-off analysis scripts, each with its own conventions. Two things that
kept going wrong and are now handled in one place:

- **Euler→RGB is not an orientation map.** Two orientations a fraction of a
  degree apart can produce very different Euler triplets near gimbal lock, so a
  single grain renders as several colours. `ipf_rgb` colours by the crystal
  direction along a sample axis instead.
- **A permissive confidence cut fills the whole grid.** The fit returns *an*
  orientation for every voxel it evaluates, so plotting at C ≥ 0.1 shows
  plausible microstructure whether or not material is there. `orientation_map`
  annotates the figure when asked to plot below `TRUST_FLOOR` (0.3).

Symmetry operators come from `midas_stress`; nothing is hand-listed here.

Implemented Laue families: cubic (SG 195–230) and hexagonal (168–194).
Anything else raises rather than silently falling back to cubic.
