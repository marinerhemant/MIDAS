# Cross-check against the C reference

> Part of the **FF-HEDM doc set**. The spine — scope gate, install gate, hard rules,
> halt conditions and the order of operations — is [`README.md`](README.md). Section
> numbers (§n) are continuous across the set; the index in the spine says which file
> holds which.

---

## 13. Cross-check against the C reference (`FF_HEDM/src`)

The C chain is the reference implementation. When a python result looks wrong, run it —
it is what found five defects in this pipeline. Recipe below; findings in the Lab Notebook.

### 13a. Build it — the shipped binaries are stale

`FF_HEDM/bin/*` on the beamline hosts were compiled in Apr/May 2026 and
`FitPosOrStrainsOMP.c` has changed since. Build fresh. chutoro has no internet, so reuse
the already-fetched dependency tree instead of letting FetchContent phone home:

```bash
cmake -S ~s1iduser/opt/MIDAS_canonical -B $HOME/opt/ffbuild \
  -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DBUILD_OMP=ON \
  -DFETCHCONTENT_BASE_DIR=/home/beams12/S1IDUSER/opt/MIDAS/build/_deps \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build $HOME/opt/ffbuild --target IndexerOMP FitPosOrStrainsOMP ProcessGrains -j 16
```

Note the build rule also copies the binaries into the source tree's `FF_HEDM/bin/`.

`FitPosOrStrainsOMP`'s usage string says `param.txt nBlocks blockNr …`; the code reads
`blockNr = argv[2], nBlocks = argv[3]` (lines 2325-2326), same as `IndexerOMP`. The usage
string is wrong — pass `blockNr nBlocks`.

### 13b. `Spots.bin` is 10 columns now; legacy `IndexerOMP` reads 9

| | `FF_HEDM/src/IndexerOMP.c` | `midas_index/c_src/IndexerUnified.c` |
|---|---|---|
| `N_COL_OBSSPOTS` | 9 (line 63) | 10 (line 100) — col 9 = `ScanNr` |

`midas_transforms.bin_data` writes the **10**-column layout, and
`midas_index/bin/midas_indexer` (built from `IndexerUnified.c`) is the maintained C
indexer the pipeline already calls. Feed the 10-column file to legacy `IndexerOMP` and it
strides through the array wrongly: on this dataset it reported

```
WARNING: SpotId 1177.000000 not found in spots file! Ignoring this spotID.   (×168 of 189)
```

and wrote an all-zero `IndexBest.bin`, after which `FitPosOrStrainsOMP` exits in 0.01 s and
`ProcessGrains` says *"OrientPos file was not found … nothing was indexed"*. **That cascade
is a format mismatch, not a parameter problem** — do not go tuning `Completeness` in
response to it. The tree documents the difference in
`midas_index/dev/c_indexer_diff.md`.

To run the legacy chain anyway, drop col 9 (row order is preserved and `Data.bin`/
`nData.bin` store row indices, not byte offsets):

```python
a = np.fromfile("Spots.bin", dtype=np.float64).reshape(-1, 10)
np.ascontiguousarray(a[:, :9]).tofile("Spots9.bin")
```

After that the warning count drops to 0.

### 13c. `ProcessGrains` needs no re-indexing

It reads only `Results/{Key,OrientPosFit,ProcessKey}.bin`, `Output/FitBest.bin`,
`SpotsToIndex.csv` and `InputAllExtraInfoFittingAll.csv` — never `Spots.bin`. So you can
point it straight at a python pipeline's output and compare grain reduction in isolation:

```bash
cd <copy of layer dir> && $HOME/opt/ffbuild/bin/ProcessGrains -paramFN paramstest.txt -nCPUs 16
```


### 13d. Where the findings are

Everything the comparison turned up — five fixed defects, the Σ3 twin verification, and
the claims that had to be retracted — is in **`LAB_NOTEBOOK.md`**. Read it before
re-investigating anything in this pipeline; several attractive hypotheses are recorded
there as *refuted*, with the measurement that killed them.

---
