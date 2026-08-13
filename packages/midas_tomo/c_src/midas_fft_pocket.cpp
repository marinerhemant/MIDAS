// midas_fft_pocket.cpp -- pocketfft behind a C boundary.
//
// pocketfft has no user-visible plan objects: twiddle factors are cached
// internally, keyed by transform size. That is why this file is so much
// smaller than the FFTW path it parallels -- there is no plan lifetime to
// manage, no wisdom to import or export, and nothing written to the working
// directory.
//
// Conventions, both matching FFTW so the two backends are interchangeable:
//   * forward uses exp(-i.2.pi.k.n/N), i.e. FFTW_FORWARD
//   * neither direction normalises (scale = 1)

#include "vendor/pocketfft_hdronly.h"

#include <cstddef>
#include <complex>
#include <vector>

extern "C" {

void midas_pocketfft_c2c_1d(float *data, int n, int forward) {
  if (n <= 0 || data == nullptr)
    return;
  const pocketfft::shape_t shape{static_cast<std::size_t>(n)};
  const pocketfft::stride_t stride{
      static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))};
  const pocketfft::shape_t axes{0};
  auto *p = reinterpret_cast<std::complex<float> *>(data);
  pocketfft::c2c(shape, stride, stride, axes, forward != 0, p, p, 1.0f);
}

void midas_pocketfft_c2c_2d(float *data, int ny, int nx, int forward) {
  if (ny <= 0 || nx <= 0 || data == nullptr)
    return;
  // Row-major (ny, nx), matching fftwf_plan_dft_2d(ny, nx, ...): the LAST
  // axis is contiguous. Getting this pair the wrong way round transposes
  // every reconstruction, so it is spelled out rather than inferred.
  const pocketfft::shape_t shape{static_cast<std::size_t>(ny),
                                 static_cast<std::size_t>(nx)};
  const auto elem = static_cast<std::ptrdiff_t>(sizeof(std::complex<float>));
  const pocketfft::stride_t stride{elem * static_cast<std::ptrdiff_t>(nx), elem};
  const pocketfft::shape_t axes{0, 1};
  auto *p = reinterpret_cast<std::complex<float> *>(data);
  pocketfft::c2c(shape, stride, stride, axes, forward != 0, p, p, 1.0f);
}

int midas_pocketfft_available(void) { return 1; }

int midas_fftw_available(void) {
#ifdef MIDAS_TOMO_HAVE_FFTW
  return 1;
#else
  return 0;
#endif
}

const char *midas_fft_engine_name(int engine) {
  switch (engine) {
  case 0:
    return "fftw";
  case 1:
    return "pocketfft";
  default:
    return "unknown";
  }
}

} // extern "C"
