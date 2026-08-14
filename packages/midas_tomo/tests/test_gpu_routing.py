"""GPU requests must never be serviced silently by the CPU.

These run without CUDA and without the engine built, because they test
*routing* decisions made in Python before any binary is invoked. That matters:
the bug they pin was found on a GPU host but is invisible there without a
discriminating check, and reproducing it needed no GPU at all.

The bug: ``api.run_engine`` defaults to ``backend="auto"``, which prefers the
ctypes shared library. CMake builds that library from the CPU sources -- the
CUDA target is a separate *executable* -- so ``gpu=True`` through the library
reached C that printed "built without CUDA" to stdout and reconstructed on the
CPU, returning 0. On copland (2x A6000) ``scripts/verify_gpu.py`` measured a
GPU-vs-CPU difference of exactly 0.000e+00 and passed.
"""

from __future__ import annotations

import pytest

from midas_tomo import api, backend_c, backend_lib


class TestGpuNeverFallsBackSilently:
    def test_library_backend_refuses_gpu(self, tmp_path):
        par = tmp_path / "x.par"
        par.write_text("")
        with pytest.raises(ValueError, match="cannot use the GPU"):
            api.run_engine(par, 1, backend="library", gpu=True)

    def test_backend_lib_refuses_gpu_before_loading(self, tmp_path):
        """Refused up front, so it raises even where the library is absent."""
        par = tmp_path / "x.par"
        par.write_text("")
        with pytest.raises(backend_lib.TomoLibraryError, match="no CUDA path"):
            backend_lib.run_param_file(par, 1, gpu=True)

    def test_backend_lib_refuses_gpu_on_the_sinos_path_too(self, tmp_path):
        import numpy as np
        par = tmp_path / "x.par"
        par.write_text("")
        with pytest.raises(backend_lib.TomoLibraryError, match="no CUDA path"):
            backend_lib.run_param_file_with_sinos(
                par, np.zeros((2, 4, 8), dtype=np.float32), 1, gpu=True)

    def test_the_error_names_the_way_out(self, tmp_path):
        par = tmp_path / "x.par"
        par.write_text("")
        with pytest.raises(ValueError) as exc:
            api.run_engine(par, 1, backend="library", gpu=True)
        # An error that says "no" without saying "do this instead" just moves
        # the problem; assert the remedy is present.
        assert "subprocess" in str(exc.value)


class TestFftEngineSelection:
    def test_names_map_to_the_c_enum(self):
        # Values pinned against c_src/midas_fft.h: MIDAS_FFT_FFTW 0, POCKET 1.
        assert backend_lib.fft_engine_code("fftw") == 0
        assert backend_lib.fft_engine_code("pocketfft") == 1

    def test_none_means_the_engine_default(self):
        assert (backend_lib.fft_engine_code(None)
                == backend_lib.FFT_ENGINES[backend_lib.DEFAULT_FFT_ENGINE])

    def test_default_is_pocketfft(self):
        assert backend_lib.DEFAULT_FFT_ENGINE == "pocketfft"

    @pytest.mark.parametrize("name", ["FFTW", " PocketFFT ", "fftw"])
    def test_names_are_case_and_space_insensitive(self, name):
        assert backend_lib.fft_engine_code(name) in (0, 1)

    def test_unknown_name_raises_rather_than_defaulting(self):
        """A typo must not quietly select pocketfft.

        The whole point of fft_engine='fftw' is reproducing a historical run.
        Silently substituting the default would report success while using the
        other transform.
        """
        with pytest.raises(ValueError, match="unknown FFT engine"):
            backend_lib.fft_engine_code("fftw3")

    def test_subprocess_path_rejects_an_unknown_engine(self, tmp_path):
        par = tmp_path / "x.par"
        par.write_text("")
        with pytest.raises(ValueError, match="unknown FFT engine"):
            backend_c.run_binary(par, 1, fft_engine="numpy")


def test_run_engine_still_rejects_a_bad_backend_name(tmp_path):
    par = tmp_path / "x.par"
    par.write_text("")
    with pytest.raises(ValueError, match="backend must be"):
        api.run_engine(par, 1, backend="cuda")
