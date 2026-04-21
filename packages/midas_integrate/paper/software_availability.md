# Software availability paragraphs

Drop these into Paper B (J. Synchrotron Rad.) and Paper C (J. Appl. Cryst.)
before submission. Both assume `midas-integrate` v0.1.0 has been published on
PyPI with a Zenodo-minted DOI for the specific release.

---

## Paper B — J. Synchrotron Rad.

### Software availability

The CPU and GPU radial integrators described in this paper are packaged
as **`midas-integrate`** (CPU) and **`midas-integrate-gpu`** (CUDA 12)
and distributed via the Python Package Index. Installation:

```
pip install midas-integrate                  # CPU
pip install 'midas-integrate[gpu]'           # + GPU (Linux x86_64)
```

The CPU package ships pre-built manylinux_2_28 (Linux x86_64) and macOS
(both x86_64 and arm64) wheels; the GPU package ships a Linux x86_64 wheel
built inside an `nvidia/cuda:12.4.1` container. The bundled CeO₂ Pilatus
calibrant used for reproducibility (figures XX–YY) ships inside the
`midas-auto-calibrate` companion wheel; `pip install midas-integrate`
pulls it automatically. All benchmarks reported in Table X were run with
`midas-integrate v0.1.0` on AMD EPYC 7763 (CPU) and NVIDIA H100 (GPU);
scripts to regenerate the tables are at
[`packages/midas_integrate/scripts/`](https://github.com/marinerhemant/MIDAS/tree/master/packages/midas_integrate/scripts).
The exact release used here is archived on Zenodo with DOI
10.5281/zenodo.XXXXXXX. Source code is BSD-3-Clause licensed at
<https://github.com/marinerhemant/MIDAS>.

### Data availability

Calibration frames and example diffraction series used for the
benchmarks are bundled with the `midas-auto-calibrate` wheel
(Pilatus 172 μm at 71.676 keV); the full four-detector dataset from
Paper A (Pilatus 6M, Varex 4343CT, CeO₂ 10 s, Ceria aero) is deposited
at Zenodo DOI 10.5281/zenodo.YYYYYYY.

---

## Paper C — J. Appl. Cryst. (Cardinal-angle aliasing)

### Software availability

The cardinal-angle aliasing correction described in Section X is
implemented in **`midas-integrate`** v0.1.0 as the `GradientCorrection`
option in `IntegrationConfig`, exposed through the `Mapper` class:

```python
import midas_integrate as mi

cfg = mi.IntegrationConfig(
    # … geometry …
    sub_pixel_cardinal_width=0.5,   # enables the cardinal-angle fix
)
mapper = mi.Mapper(cfg)
```

The reproduction scripts for Figure XX (uncorrected vs corrected cake
projections) are shipped inside the wheel at
`midas_integrate.figures.paper_c` and can be run via the
`midas-integrate-fig-c` console entry point. Package source is
BSD-3-Clause at <https://github.com/marinerhemant/MIDAS>; the
specific release used here is archived at Zenodo DOI
10.5281/zenodo.XXXXXXX.

### Data availability

Synthetic calibrant test frames demonstrating the aliasing artefact are
bundled in the `midas-auto-calibrate` wheel. The measured Pilatus
calibration data used in Section Y is available at Zenodo DOI
10.5281/zenodo.YYYYYYY.

---

## Notes for the user filling these in

- Replace `10.5281/zenodo.XXXXXXX` with the actual DOI after Zenodo
  minting — see `mac.data.zenodo_url()` in midas-auto-calibrate.
- If benchmarks re-run, update the hardware names (EPYC 7763, H100) to
  match the revised runs.
- Paper B's Table X — the calibration runs are in
  `packages/midas_auto_calibrate/scripts/regen_table4.py`; integration
  timing scripts land in `packages/midas_integrate/scripts/regen_timing_table.py`
  (v0.2).
