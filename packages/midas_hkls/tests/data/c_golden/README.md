# C golden hkls tables for non-cubic cells

Real output of the compiled C `GetHKLList` (CLI mode, `--stdout`), the only non-cubic C reference in the repo
(the `hkls.csv` files under FF_HEDM/Example and NF_HEDM/Example are all cubic Au).

Provenance
- host: copland (ssh copland, account s1iduser), 2026-10-21
- binary: `~/opt/MIDAS/build/bin/GetHKLList`, built 2026-04-30, prints `Version: MIDAS v11.0 (8be17019)`;
  source tree `~/opt/MIDAS` at commit 2b21a5e9 (2026-05-21)
- common arguments: `--wl 0.222910 --lsd 1000000 --maxR 300000 --stdout`
- each file keeps the table (header line `h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius`) cut at the first
  ring boundary after 260 rows; the C log lines are dropped.

| file | `--sg` | `--lp` |
|---|---|---|
| hexagonal_194_hkls.csv | 194 | 2.95 2.95 4.68 90 90 120 |
| trigonal_hex_167_hkls.csv | 167 | 4.76 4.76 12.99 90 90 120 |
| tetragonal_123_hkls.csv | 123 | 3.0 3.0 4.5 90 90 90 |
| orthorhombic_62_hkls.csv | 62 | 3.0 4.0 5.0 90 90 90 |
| monoclinic_14_b105_hkls.csv | 14 | 4.0 5.0 6.0 90 105 90 |
| monoclinic_14_b95_hkls.csv | 14 | 4.0 5.0 6.0 90 95 90 |
| triclinic_2_hkls.csv | 2 | 4.0 5.0 6.0 80 95 100 |
| rhombohedral_167r_hkls.csv | 167 | 5.0 5.0 5.0 55 55 55 |

What they pin
- `g = B (h, k, l)` with the MIDAS B matrix (a* along x, b* in the xy plane): all eight cells to ~4e-16.
- the textbook `midas_hkls.Lattice` frame does NOT reproduce the C g vectors for any non-orthogonal cell
  (max |dg| 0.07 to 0.83), and `frame_rotation` maps it onto them.
- the Python `emit_nf_hkls_csv` reproduces the C rows (7 cells exactly, ring numbers included).

Known C limitation these files expose: `GetHKLList` hard-codes the hkl box to 0..10 in h, k, l (it prints
"Will go from 0 to 10 in h; 0 to 10 in k; 0 to 10 in l" whatever the cell). For the long-c trigonal cell
(c = 12.99 A) it never emits (0, 0, +-12), d = 1.0825 A, which is inside the detector range; the Python emitter
sizes the box from d_min and does. The test compares inside the C box only.

Regenerate: run the commands above on a host with the MIDAS build and trim as described.
