/* bigdet_test.c — exercise the BigDetector active-area path of the shared
 * forward model, which `parity_test.c` deliberately leaves off (it runs with
 * BigDetSize = 0 so it can compare against the legacy bodies).
 *
 * Three things are checked, all of which are new or newly reachable:
 *
 *   1. mask OFF (bigdet = NULL) == mask ON with every bit set. If these
 *      differ, the mask path is doing something other than filtering.
 *   2. a mask with a cleared region actually removes spots, and removes them
 *      from `nSpotsFracCalc` too -- i.e. from the completeness DENOMINATOR,
 *      not just the emitted list. That is the whole point of the feature.
 *   3. the bounds guard: a grid far too small to contain the predicted spots
 *      must drop them rather than index past the end of the bitset. Before
 *      the guard this was an out-of-bounds read of an mmap'd file, and `yl`
 *      is bounded only by BoxSizes, which is routinely +/-1e6 um.
 *
 * Build (from packages/midas_ckernel):
 *   cc -O2 -I c_src tests/bigdet_test.c c_src/forward.c c_src/MIDAS_Math.c \
 *      -lm -o /tmp/ck_bigdet && /tmp/ck_bigdet
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "forward.h"
#include "MIDAS_Math.h"
#include "MIDAS_Limits.h"

#define PX 200.0
/* S must cover the outermost ring or the bounds guard (correctly) drops
 * everything: at a=3.6 A, lambda=0.2066 A, Lsd=1e6 um the (311) ring sits at
 * ~193000 um = ~964 px, so S/2 must exceed that. 512 was tried first and every
 * spot vanished -- which is the guard working, not a bug. */
#define S 2048                    /* BigDetSize */
#define NWORDS ((S * S) / 32 + 1)

static unsigned int g_mask[NWORDS];

static void set_all(int on) {
  memset(g_mask, on ? 0xFF : 0x00, sizeof(g_mask));
}

static void clear_cell(int yc, int zc) {
  long long k = (long long)yc + (long long)S * zc;
  if (k < 0 || k >= (long long)S * S) return;
  g_mask[k / 32] &= ~(1u << (k % 32));
}

/* Same integer arithmetic as forward.c:183-187. */
static void cell_of(double yl, double zl, int *yc, int *zc) {
  *yc = (int)floor((S / 2) - (int)(-yl / PX));
  *zc = (int)floor((int)(zl / PX) + (S / 2));
}

int main(void) {
  /* One cubic phase, a handful of rings, full omega range, generous box. */
  const int n_hkls_l = 4;
  double **hk = malloc((size_t)n_hkls_l * sizeof(double *));
  double *buf = calloc((size_t)n_hkls_l * MIDAS_CK_NCOLS, sizeof(double));
  for (int i = 0; i < n_hkls_l; i++) hk[i] = buf + (size_t)i * MIDAS_CK_NCOLS;

  /* Canonical hkl layout: [0..2] G, [3] ringnr, [4] Ds, [5] theta,
   * [6] RingRadius, [8] v, [9] v*v. A cubic a=3.6 A at lambda=0.2066 A. */
  const double a = 3.6, wl = 0.2066, lsd = 1000000.0;
  int hkl[4][3] = {{1, 1, 1}, {2, 0, 0}, {2, 2, 0}, {3, 1, 1}};
  for (int i = 0; i < n_hkls_l; i++) {
    double h = hkl[i][0], k = hkl[i][1], l = hkl[i][2];
    double len = sqrt(h * h + k * k + l * l);
    double ds = a / len;                       /* d-spacing */
    double th = asin(wl / (2.0 * ds)) * 180.0 / M_PI;
    hk[i][0] = h / a; hk[i][1] = k / a; hk[i][2] = l / a;
    hk[i][3] = (double)(i + 1);
    hk[i][4] = ds;
    hk[i][5] = th;
    hk[i][6] = lsd * tan(2.0 * th * M_PI / 180.0);
    double v = sin(th * M_PI / 180.0) * len / a * a;  /* |G| sin(theta) form */
    v = sin(th * M_PI / 180.0) * (len / a);
    hk[i][8] = v;
    hk[i][9] = v * v;
  }

  double om[1][2] = {{-180.0, 180.0}};
  double bx[1][4] = {{-1e6, 1e6, -1e6, 1e6}};
  double OM[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

  int cap = 2 * n_hkls_l + 8;
  double **sp = malloc((size_t)cap * sizeof(double *));
  double *spbuf = calloc((size_t)cap * MIDAS_CK_NCOLS, sizeof(double));
  for (int i = 0; i < cap; i++) sp[i] = spbuf + (size_t)i * MIDAS_CK_NCOLS;

  int n_off = 0, nf_off = 0;
  midas_ck_calc_diffraction_spots(OM, lsd, NULL, hk, n_hkls_l, om, bx, 1, 0.0,
                                  NULL, 0, NULL, 0, sp, &n_off, &nf_off);
  if (n_off <= 0) {
    printf("FAIL: baseline produced %d spots; the fixture is degenerate\n",
           n_off);
    return 1;
  }

  /* --- 1. all bits set must equal no mask at all --------------------- */
  set_all(1);
  MidasCkBigDet bd = {S, g_mask, PX};
  int n_all = 0, nf_all = 0;
  double *snap = malloc((size_t)cap * MIDAS_CK_NCOLS * sizeof(double));
  memcpy(snap, spbuf, (size_t)cap * MIDAS_CK_NCOLS * sizeof(double));
  memset(spbuf, 0, (size_t)cap * MIDAS_CK_NCOLS * sizeof(double));
  midas_ck_calc_diffraction_spots(OM, lsd, NULL, hk, n_hkls_l, om, bx, 1, 0.0,
                                  NULL, 0, &bd, 0, sp, &n_all, &nf_all);
  if (n_all != n_off || nf_all != nf_off ||
      memcmp(snap, spbuf, (size_t)n_off * MIDAS_CK_NCOLS * sizeof(double))) {
    printf("FAIL: all-bits-set mask changed the result (%d vs %d spots)\n",
           n_all, n_off);
    return 1;
  }
  printf("[all-set]  %d spots, identical to no-mask: PASS\n", n_all);

  /* --- 2. clearing the cells of real spots must remove them ---------- */
  set_all(1);
  int n_target = n_off / 2;
  if (n_target < 1) n_target = 1;
  for (int i = 0; i < n_target; i++) {
    int yc, zc;
    cell_of(snap[(size_t)i * MIDAS_CK_NCOLS + 4],
            snap[(size_t)i * MIDAS_CK_NCOLS + 5], &yc, &zc);
    /* clear a 3x3 block so sub-cell rounding cannot save the spot */
    for (int dy = -1; dy <= 1; dy++)
      for (int dz = -1; dz <= 1; dz++) clear_cell(yc + dy, zc + dz);
  }
  int n_cut = 0, nf_cut = 0;
  midas_ck_calc_diffraction_spots(OM, lsd, NULL, hk, n_hkls_l, om, bx, 1, 0.0,
                                  NULL, 0, &bd, 0, sp, &n_cut, &nf_cut);
  if (n_cut != n_off - n_target) {
    printf("FAIL: cleared %d spots' cells, expected %d spots, got %d\n",
           n_target, n_off - n_target, n_cut);
    return 1;
  }
  if (nf_cut != nf_off - n_target) {
    printf("FAIL: masked spots left the emitted list but NOT the "
           "completeness denominator (nSpotsFracCalc %d, expected %d)\n",
           nf_cut, nf_off - n_target);
    return 1;
  }
  printf("[cleared]  %d -> %d spots, denominator %d -> %d: PASS\n", n_off,
         n_cut, nf_off, nf_cut);

  /* --- 3. bounds guard: a grid too small must drop, not read OOB ----- */
  set_all(1);
  MidasCkBigDet tiny = {4, g_mask, PX};   /* 4x4 cells; every spot is outside */
  int n_tiny = 0, nf_tiny = 0;
  midas_ck_calc_diffraction_spots(OM, lsd, NULL, hk, n_hkls_l, om, bx, 1, 0.0,
                                  NULL, 0, &tiny, 0, sp, &n_tiny, &nf_tiny);
  if (n_tiny != 0) {
    printf("FAIL: %d spots survived a 4x4 grid they cannot fit in; the "
           "bounds guard is not firing\n", n_tiny);
    return 1;
  }
  printf("[bounds]   4x4 grid dropped all %d spots, no OOB read: PASS\n",
         n_off);

  printf("PASS (bigdet)\n");
  free(snap); free(sp); free(spbuf); free(hk); free(buf);
  return 0;
}
