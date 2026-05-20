"""Pipeline — chains all four stages with intermediates kept on-device.

The four stages (``merge_overlapping_peaks``, ``calc_radius``, ``fit_setup``,
``bin_data``) are also exposed as standalone functions; this class threads
them together with no CSV / binary disk round-trips between stages.

Disk writes happen only at ``Pipeline.dump(out_dir)`` (or are skipped
entirely when the next consumer — typically ``midas-index`` — is in-process).

Typical usage:

    from midas_transforms import Pipeline

    pipe = Pipeline.from_zarr("scan.zip", device="cuda")
    result = pipe.run()             # all four stages on GPU
    pipe.dump("/path/to/output")    # writes 9 files
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from .bin_data import BinDataResult, bin_data
from .device import resolve_device, resolve_dtype
from .fit_setup import FitSetupResult, fit_setup
from .merge import MergeResult, merge_overlapping_peaks
from .params import ZarrParams, read_zarr_params
from .radius import RadiusResult, calc_radius


@dataclass
class PipelineResult:
    merge: Optional[MergeResult] = None
    radius: Optional[RadiusResult] = None
    fit_setup: Optional[FitSetupResult] = None
    bins: Optional[BinDataResult] = None


@dataclass
class Pipeline:
    """Drives all four transforms end-to-end with on-device intermediates."""

    zarr_params: ZarrParams
    allpeaks_ps_bin: Optional[Path] = None
    allpeaks_px_bin: Optional[Path] = None
    result_folder: Path = Path(".")
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float64
    overlap_length: Optional[float] = None

    _result: Optional[PipelineResult] = None

    @classmethod
    def from_zarr(
        cls,
        zarr_path: Union[str, Path],
        *,
        allpeaks_ps_bin: Optional[Union[str, Path]] = None,
        allpeaks_px_bin: Optional[Union[str, Path]] = None,
        result_folder: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        overlap_length: Optional[float] = None,
    ) -> "Pipeline":
        zp = read_zarr_params(zarr_path)
        rf = Path(result_folder) if result_folder is not None else Path(zarr_path).parent
        dev = resolve_device(device)
        dt = resolve_dtype(dev, dtype)
        if allpeaks_ps_bin is None:
            cand = rf / "Temp" / "AllPeaks_PS.bin"
            if not cand.exists():
                cand = rf / "AllPeaks_PS.bin"
            allpeaks_ps_bin = cand if cand.exists() else None
        if allpeaks_px_bin is None and zp.UsePixelOverlap:
            cand = rf / "Temp" / "AllPeaks_PX.bin"
            if not cand.exists():
                cand = rf / "AllPeaks_PX.bin"
            allpeaks_px_bin = cand if cand.exists() else None
        return cls(
            zarr_params=zp,
            allpeaks_ps_bin=Path(allpeaks_ps_bin) if allpeaks_ps_bin else None,
            allpeaks_px_bin=Path(allpeaks_px_bin) if allpeaks_px_bin else None,
            result_folder=rf,
            device=dev,
            dtype=dt,
            overlap_length=overlap_length,
        )

    def run(self) -> PipelineResult:
        """Run all four stages with no disk I/O between them.

        Returns a ``PipelineResult`` whose fields hold tensors on
        ``self.device``. Disk writes are deferred to ``dump()``.
        """
        pr = PipelineResult()
        # 1. Merge.
        if self.allpeaks_ps_bin is None:
            raise FileNotFoundError(
                "Pipeline requires AllPeaks_PS.bin (the midas-peakfit / "
                "PeaksFittingOMPZarrRefactor consolidated blob); pass "
                "allpeaks_ps_bin= explicitly to Pipeline.from_zarr()."
            )
        pr.merge = merge_overlapping_peaks(
            allpeaks_ps_bin=self.allpeaks_ps_bin,
            allpeaks_px_bin=self.allpeaks_px_bin,
            result_folder=self.result_folder,
            overlap_length=self.overlap_length,
            skip_frame=self.zarr_params.SkipFrame,
            use_maxima_positions=bool(self.zarr_params.UseMaximaPositions),
            use_pixel_overlap=bool(self.zarr_params.UsePixelOverlap),
            nr_pixels=self.zarr_params.NrPixels,
            device=self.device, dtype=self.dtype,
            write=False,
        )
        merge_arr = pr.merge.peaks.detach().cpu().numpy().astype(np.float64)

        # 2. Radius.
        pr.radius = calc_radius(
            result_folder=self.result_folder,
            zarr_params=self.zarr_params,
            result_array=merge_arr,
            start_nr=1, end_nr=self.zarr_params.EndNr if self.zarr_params.EndNr > 0 else len(merge_arr),
            device=self.device, dtype=self.dtype,
            write=False,
        )
        radius_arr = pr.radius.spots.detach().cpu().numpy().astype(np.float64)

        # 3. FitSetup.
        pr.fit_setup = fit_setup(
            result_folder=self.result_folder,
            zarr_params=self.zarr_params,
            radius_array=radius_arr,
            device=self.device, dtype=self.dtype,
            write=False,
        )
        spots_inputall = pr.fit_setup.spots_inputall.detach().cpu().numpy().astype(np.float64)
        spots_extra = pr.fit_setup.extra.detach().cpu().numpy().astype(np.float64)

        # 4. BinData.
        pr.bins = bin_data(
            result_folder=self.result_folder,
            paramstest=pr.fit_setup.paramstest,
            spots_inputall=spots_inputall,
            extra_inputall=spots_extra,
            device=self.device, dtype=self.dtype,
            write=False,
        )

        self._result = pr
        return pr

    def dump(self, out_dir: Union[str, Path]) -> None:
        """Write all intermediate and final outputs to ``out_dir``.

        Mirrors what each stage's CLI would have written separately —
        for compatibility with the ff_MIDAS workflow's per-stage checkpointing.
        """
        if self._result is None:
            raise RuntimeError("Call Pipeline.run() before Pipeline.dump().")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        from .io import binary as bio
        from .io import csv as csv_io
        from .params import write_paramstest

        pr = self._result
        if pr.merge is not None:
            csv_io.write_result_csv(
                out_dir / f"Result_StartNr_1_EndNr_{self.zarr_params.EndNr}.csv",
                pr.merge.peaks.detach().cpu().numpy(),
            )
            with open(out_dir / "MergeMap.csv", "w") as f:
                f.write("MergedSpotID FrameNr PeakID\n")
                for (sid, fn, pid) in pr.merge.merge_map:
                    f.write(f"{sid} {fn} {pid}\n")
        if pr.radius is not None:
            csv_io.write_radius_csv(
                out_dir / f"Radius_StartNr_1_EndNr_{self.zarr_params.EndNr}.csv",
                pr.radius.spots.detach().cpu().numpy(),
            )
        if pr.fit_setup is not None:
            csv_io.write_inputall_csv(out_dir / "InputAll.csv", pr.fit_setup.spots_inputall.detach().cpu().numpy())
            csv_io.write_inputall_extra_csv(
                out_dir / "InputAllExtraInfoFittingAll.csv",
                pr.fit_setup.extra.detach().cpu().numpy(),
            )
            csv_io.write_spots_to_index(
                out_dir / "SpotsToIndex.csv",
                pr.fit_setup.spot_ids_to_index.detach().cpu().numpy().tolist(),
            )
            if pr.fit_setup.paramstest is not None:
                write_paramstest(pr.fit_setup.paramstest, out_dir / "paramstest.txt")
        if pr.bins is not None:
            bio.write_spots_bin(out_dir / "Spots.bin", pr.bins.spots.detach().cpu().numpy())
            bio.write_extrainfo_bin(out_dir / "ExtraInfo.bin", pr.bins.extra_info.detach().cpu().numpy())
            if pr.bins.data is not None and pr.bins.ndata is not None:
                # Phase 5: always write int64-pair Data.bin / nData.bin (FF
                # uses scan_nr=0).
                import numpy as np
                data_np = pr.bins.data.detach().cpu().numpy().astype(np.int64)
                data_pairs = np.zeros((data_np.size, 2), dtype=np.uint64)
                data_pairs[:, 0] = data_np.astype(np.uint64)
                ndata_np = pr.bins.ndata.detach().cpu().numpy().reshape(-1, 2).astype(np.uint64)
                bio.write_data_ndata_bin_scanning(
                    out_dir / "Data.bin", out_dir / "nData.bin",
                    data_pairs, ndata_np,
                )
                positions_path = out_dir / "positions.csv"
                if not positions_path.exists():
                    positions_path.write_text("0.000000\n")
