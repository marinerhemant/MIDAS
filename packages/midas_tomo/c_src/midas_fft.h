/* midas_fft.h -- FFT backend selection for the gridrec engine.
 *
 * Two backends:
 *
 *   FFTW3f      the historical one. GPL-2.0-or-later, so it cannot be shipped
 *               inside a BSD-licensed wheel; the user supplies it.
 *   pocketfft   BSD-3-Clause, vendored under c_src/vendor/. Same licence as
 *               MIDAS, so it CAN ship, which is the whole point: with it the
 *               package builds with nothing but a compiler.
 *
 * When FFTW is present at build time both are available and selectable at
 * run time. When it is absent this header supplies the handful of fftwf_*
 * types and stubs the engine's declarations need, so tomo_gridrec.c still
 * compiles unchanged and simply never takes the FFTW branch.
 *
 * NOTE ON RESULTS: the two backends compute the same transform but not the
 * same bits. Expect a low-order-bit difference, measured rather than assumed
 * (see tests/test_fft_engines.py).
 */

#ifndef MIDAS_FFT_H
#define MIDAS_FFT_H

#include <stddef.h>
#include <stdlib.h>

/* ------------------------------------------------------------ engine ids */
#define MIDAS_FFT_FFTW 0
#define MIDAS_FFT_POCKET 1

#ifdef __cplusplus
extern "C" {
#endif

/* In-place complex-to-complex transforms on interleaved float pairs.
 * `data` holds 2*n (1-D) or 2*ny*nx (2-D) floats. `forward` uses the
 * exp(-i.2.pi.k.n/N) sign, matching FFTW_FORWARD; neither normalises,
 * matching FFTW. */
void midas_pocketfft_c2c_1d(float *data, int n, int forward);
void midas_pocketfft_c2c_2d(float *data, int ny, int nx, int forward);

/* 1 when the pocketfft backend was compiled in. */
int midas_pocketfft_available(void);

/* 1 when FFTW was compiled in. */
int midas_fftw_available(void);

/* Name of an engine id, for messages. */
const char *midas_fft_engine_name(int engine);

#ifdef __cplusplus
}
#endif

/* ------------------------------------------- fftwf_* when FFTW is absent
 *
 * Just enough for the engine to compile and link. Every one of these is
 * unreachable in a no-FFTW build because the engine forces the pocketfft
 * branch, but they must exist for the FFTW branch to be a valid translation
 * unit. Deliberately NOT no-ops that silently return wrong data: the plan
 * constructors return NULL and fftwf_execute aborts loudly, so a routing
 * mistake fails immediately instead of producing a plausible zero image.
 */
#ifndef MIDAS_TOMO_HAVE_FFTW

#include <stdio.h>

typedef float fftwf_complex[2];
typedef void *fftwf_plan;

#define FFTW_FORWARD (-1)
#define FFTW_BACKWARD (1)
#define FFTW_MEASURE (0U)
#define FFTW_ESTIMATE (1U << 6)
#define FFTW_WISDOM_ONLY (1U << 21)

static inline void *fftwf_malloc(size_t n) { return malloc(n); }
static inline void fftwf_free(void *p) { free(p); }

static inline fftwf_plan fftwf_plan_dft_1d(int n, fftwf_complex *in,
                                           fftwf_complex *out, int sign,
                                           unsigned flags) {
  (void)n; (void)in; (void)out; (void)sign; (void)flags;
  return NULL;
}
static inline fftwf_plan fftwf_plan_dft_2d(int ny, int nx, fftwf_complex *in,
                                           fftwf_complex *out, int sign,
                                           unsigned flags) {
  (void)ny; (void)nx; (void)in; (void)out; (void)sign; (void)flags;
  return NULL;
}
static inline void fftwf_execute(fftwf_plan p) {
  (void)p;
  fprintf(stderr,
          "FATAL: fftwf_execute() reached in a build without FFTW. The engine "
          "should have taken the pocketfft branch; this is a routing bug, not "
          "a configuration problem.\n");
  abort();
}
static inline void fftwf_destroy_plan(fftwf_plan p) { (void)p; }
static inline int fftwf_import_wisdom_from_filename(const char *f) {
  (void)f;
  return 0;
}
static inline void fftwf_export_wisdom_to_filename(const char *f) { (void)f; }
static inline char *fftwf_export_wisdom_to_string(void) {
  return (char *)calloc(1, sizeof(char));
}
static inline int fftwf_import_wisdom_from_string(const char *s) {
  (void)s;
  return 0;
}

#endif /* !MIDAS_TOMO_HAVE_FFTW */

#endif /* MIDAS_FFT_H */
