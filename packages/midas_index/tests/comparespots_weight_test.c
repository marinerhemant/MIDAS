/* comparespots_weight_test.c — drive CompareSpots directly.
 *
 * The completeness weighting added on 2026-08-23 lives entirely inside
 * IndexerUnified.c: a 12th (optional) hkls.csv column becomes hkls[i][10], and
 * CompareSpots accumulates weighted twins of nMatchesFracCalc /
 * nTspotsFracCalc under the identical skipRadialFilter test.
 *
 * CompareSpots is `static`, so this TU #includes the whole translation unit
 * (with main() renamed out of the way) rather than linking against it. That is
 * deliberate: the alternative is a full synthetic indexing fixture --
 * Spots.bin, the ~1 GB binned Data.bin/nData.bin, positions.csv -- to test
 * twenty lines of arithmetic. Here the bins are eight entries built by hand.
 *
 * Three things are asserted, and the first is the one that protects every
 * existing result:
 *
 *   1. raw (ConfidenceMetric 0) is BIT-IDENTICAL with and without weights
 *      present. Weighted sums are still accumulated in raw mode; if they ever
 *      leak into the returned ratio, this fails.
 *   2. a UNIFORM weight is a no-op in weighted mode. Both sides of the ratio
 *      scale together, so w=1 everywhere must reproduce raw exactly. This is
 *      the all-ones trap: a weighting that is silently never applied also
 *      passes a test that only ever uses uniform weights, which is why (3)
 *      exists.
 *   3. a NON-UNIFORM weight must MOVE the number, in the predicted direction:
 *      down-weighting a matched reflection lowers completeness, and
 *      down-weighting a missed one raises it.
 *
 * Build (from packages/midas_index):
 *   cc -std=gnu99 -fopenmp -O2 -I c_src -I <builddir> \
 *      tests/comparespots_weight_test.c c_src/MIDAS_Math.c \
 *      c_src/GetMisorientation.c c_src/forward.c -lm -o /tmp/cs_weight
 */
#define main midas_indexer_main_under_test
#include "IndexerUnified.c"
#undef main

#include <assert.h>

#define N_ETA 72        /* EtaBinSize 5 deg */
#define N_OME 72        /* OmeBinSize 5 deg */
#define N_RING 3

/* One theoretical spot row wide enough for the shared layout. */
static double *mk_theor(int n) {
  double *buf = (double *)calloc((size_t)n * MIDAS_CK_NCOLS, sizeof(double));
  return buf;
}

/* Fill a theoretical spot: ring, eta, omega, hkl index, y/z. */
static void set_theor(double **T, int i, int ihkl, int ring, double eta,
                      double omega, double y, double z) {
  T[i][2] = (double)ihkl;      /* indexhkl — the weight join key */
  T[i][6] = omega;
  T[i][9] = (double)ring;
  T[i][10] = y;                /* displaced y (caller-filled) */
  T[i][11] = z;                /* displaced z */
  T[i][12] = eta;              /* eta at displaced position */
  T[i][14] = sin(omega * deg2rad);
  T[i][15] = cos(omega * deg2rad);
}

/* Put one observed spot in the bin that TheorSpots[i] will look in.
 *
 * Each bin gets its own one-entry slice of `data`, keyed to the observed row.
 * An earlier version pointed every bin at base 0, so the three spots
 * overwrote each other and only one ever matched — the harness reported
 * 1/4 instead of 3/4. */
static void add_obs(size_t row, int ring, double eta, double omega,
                    double y, double z, double ringrad) {
  ObsSpotsLab[row * 10 + 0] = y;
  ObsSpotsLab[row * 10 + 1] = z;
  ObsSpotsLab[row * 10 + 2] = omega;
  ObsSpotsLab[row * 10 + 3] = ringrad;
  ObsSpotsLab[row * 10 + 4] = (double)(row + 1);   /* SpotID */
  ObsSpotsLab[row * 10 + 5] = (double)ring;
  ObsSpotsLab[row * 10 + 6] = eta;
  ObsSpotsLab[row * 10 + 8] = 0.0;                 /* RadiusDistIdeal */
  ObsSpotsLab[row * 10 + 9] = 0.0;                 /* ScanNr (FF) */

  int iEta = (int)floor((180 + eta) / 5.0);
  int iOme = (int)floor((180 + omega) / 5.0);
  size_t Pos = (size_t)(ring - 1) * N_ETA * N_OME + (size_t)iEta * N_OME + iOme;
  /* one entry per bin, at its own base == the observed row index */
  data[row * 2 + 0] = row;
  data[row * 2 + 1] = 0;                           /* scannrobs */
  ndata[Pos * 2 + 0] = 1;                          /* count */
  ndata[Pos * 2 + 1] = row;                        /* base offset */
}

int main(void) {
  /* ---- geometry-free params: wide margins so matching is unambiguous ---- */
  struct TParams P;
  memset(&P, 0, sizeof(P));
  P.EtaBinSize = 5.0;  P.OmeBinSize = 5.0;
  P.InvEtaBinSize = 1.0 / 5.0;  P.InvOmeBinSize = 1.0 / 5.0;
  P.MarginOme = 2.0;  P.MarginRad = 1000.0;  P.MarginRadial = 1000.0;
  P.StepsizeOrient = 0.1;
  P.ConfidenceMetric = 0;  P.ForbiddenF2Threshold = 1e-6;
  P.nRingsToRejectCalc = 0;
  n_eta_bins = N_ETA;  n_ome_bins = N_OME;

  RealType etamargins[MAX_N_RINGS];
  for (int i = 0; i < MAX_N_RINGS; i++) etamargins[i] = 5.0;

  /* ---- bins ---- */
  size_t nbins = (size_t)N_RING * N_ETA * N_OME;
  ndata = (size_t *)calloc(nbins * 2, sizeof(size_t));
  data = (size_t *)calloc(64 * 2, sizeof(size_t));

  ObsSpotsLab = (RealType *)calloc(16 * 10, sizeof(RealType));
  n_spots = 0;

  /* ---- reflections: 4 hkls on 2 rings, one MISSED ---- */
  n_hkls = 4;
  for (int i = 0; i < n_hkls; i++) {
    memset(hkls[i], 0, 11 * sizeof(double));
    hkls[i][3] = (i < 2) ? 1.0 : 2.0;      /* ringnr */
    hkls[i][10] = 1.0;                     /* weight, default */
  }

  int nT = 4;
  double *tbuf = mk_theor(nT);
  double *T[4];
  for (int i = 0; i < nT; i++) T[i] = tbuf + (size_t)i * MIDAS_CK_NCOLS;

  /*        i  ihkl ring   eta    omega     y       z    */
  set_theor(T, 0, 0,  1,   10.0,  20.0,  -100.0, 500.0);
  set_theor(T, 1, 1,  1,   50.0,  60.0,  -300.0, 400.0);
  set_theor(T, 2, 2,  2,   90.0, 100.0,  -500.0,   0.0);
  set_theor(T, 3, 3,  2,  130.0, 140.0,  -400.0,-300.0);

  /* three of the four have an observed partner; hkl 3 is MISSED */
  add_obs(0, 1,  10.0,  20.0, -100.0, 500.0, 0.0);
  add_obs(1, 1,  50.0,  60.0, -300.0, 400.0, 0.0);
  add_obs(2, 2,  90.0, 100.0, -500.0,   0.0, 0.0);

  RealType **GS = allocMatrix(nT + 4, 16);
  int nMatch = 0, nMFrac = 0;
  double wM = 0.0, wT = 0.0;

  /* ---- 1. raw ---- */
  P.ConfidenceMetric = 0;
  CompareSpots(T, nT, 0.0, P.MarginRad, P.MarginRadial, etamargins,
               P.MarginOme, P.StepsizeOrient, 1, 0.0, 0.0, &P,
               &nMatch, GS, &nMFrac, NULL, 0, &wM, &wT);
  double frac_raw = FracMatched(&P, nMFrac, nT, wM, wT);
  printf("[raw]      matched %d/%d  frac %.10f\n", nMFrac, nT, frac_raw);
  if (nMFrac != 3) { printf("FAIL: expected 3 matches, got %d\n", nMFrac); return 1; }
  if (fabs(frac_raw - 0.75) > 1e-12) { printf("FAIL: raw frac != 0.75\n"); return 1; }

  /* ---- 2. weighted with UNIFORM w=1 must equal raw exactly ---- */
  P.ConfidenceMetric = 2;
  nMatch = 0; nMFrac = 0; wM = 0.0; wT = 0.0;
  CompareSpots(T, nT, 0.0, P.MarginRad, P.MarginRadial, etamargins,
               P.MarginOme, P.StepsizeOrient, 1, 0.0, 0.0, &P,
               &nMatch, GS, &nMFrac, NULL, 0, &wM, &wT);
  double frac_uniform = FracMatched(&P, nMFrac, nT, wM, wT);
  printf("[uniform]  wM %.4f wT %.4f  frac %.10f\n", wM, wT, frac_uniform);
  if (frac_uniform != frac_raw) {
    printf("FAIL: uniform weight changed the ratio (%.17g vs %.17g)\n",
           frac_uniform, frac_raw);
    return 1;
  }

  /* ---- 3a. down-weight a MATCHED reflection -> frac must DROP ---- */
  hkls[0][10] = 0.1;                       /* hkl 0 is matched */
  nMatch = 0; nMFrac = 0; wM = 0.0; wT = 0.0;
  CompareSpots(T, nT, 0.0, P.MarginRad, P.MarginRadial, etamargins,
               P.MarginOme, P.StepsizeOrient, 1, 0.0, 0.0, &P,
               &nMatch, GS, &nMFrac, NULL, 0, &wM, &wT);
  double frac_dm = FracMatched(&P, nMFrac, nT, wM, wT);
  double want_dm = (0.1 + 1.0 + 1.0) / (0.1 + 1.0 + 1.0 + 1.0);
  printf("[matched   down] wM %.4f wT %.4f  frac %.10f (want %.10f)\n",
         wM, wT, frac_dm, want_dm);
  if (fabs(frac_dm - want_dm) > 1e-12) { printf("FAIL: wrong weighted ratio\n"); return 1; }
  if (!(frac_dm < frac_raw)) { printf("FAIL: down-weighting a MATCH must lower frac\n"); return 1; }

  /* ---- 3b. down-weight the MISSED reflection -> frac must RISE ---- */
  hkls[0][10] = 1.0;
  hkls[3][10] = 0.1;                       /* hkl 3 is the missed one */
  nMatch = 0; nMFrac = 0; wM = 0.0; wT = 0.0;
  CompareSpots(T, nT, 0.0, P.MarginRad, P.MarginRadial, etamargins,
               P.MarginOme, P.StepsizeOrient, 1, 0.0, 0.0, &P,
               &nMatch, GS, &nMFrac, NULL, 0, &wM, &wT);
  double frac_du = FracMatched(&P, nMFrac, nT, wM, wT);
  double want_du = 3.0 / (3.0 + 0.1);
  printf("[unmatched down] wM %.4f wT %.4f  frac %.10f (want %.10f)\n",
         wM, wT, frac_du, want_du);
  if (fabs(frac_du - want_du) > 1e-12) { printf("FAIL: wrong weighted ratio\n"); return 1; }
  if (!(frac_du > frac_raw)) { printf("FAIL: down-weighting a MISS must raise frac\n"); return 1; }

  /* ---- 4. filtered mode: a zero-F2 reflection leaves BOTH sides ---- */
  P.ConfidenceMetric = 1;
  hkls[3][10] = 0.0;                       /* the missed one is forbidden */
  nMatch = 0; nMFrac = 0; wM = 0.0; wT = 0.0;
  CompareSpots(T, nT, 0.0, P.MarginRad, P.MarginRadial, etamargins,
               P.MarginOme, P.StepsizeOrient, 1, 0.0, 0.0, &P,
               &nMatch, GS, &nMFrac, NULL, 0, &wM, &wT);
  double frac_filt = FracMatched(&P, nMFrac, nT, wM, wT);
  printf("[filtered]  wM %.4f wT %.4f  frac %.10f\n", wM, wT, frac_filt);
  if (fabs(frac_filt - 1.0) > 1e-12) {
    printf("FAIL: a forbidden MISSED reflection must leave the denominator, "
           "giving frac 1.0; got %.10f\n", frac_filt);
    return 1;
  }

  /* ---- 5. raw is unaffected by any of it ---- */
  P.ConfidenceMetric = 0;
  nMatch = 0; nMFrac = 0; wM = 0.0; wT = 0.0;
  CompareSpots(T, nT, 0.0, P.MarginRad, P.MarginRadial, etamargins,
               P.MarginOme, P.StepsizeOrient, 1, 0.0, 0.0, &P,
               &nMatch, GS, &nMFrac, NULL, 0, &wM, &wT);
  double frac_raw2 = FracMatched(&P, nMFrac, nT, wM, wT);
  if (frac_raw2 != frac_raw) {
    printf("FAIL: raw mode moved after weights were set (%.17g vs %.17g)\n",
           frac_raw2, frac_raw);
    return 1;
  }
  printf("[raw again] frac %.10f — unchanged with weights present\n", frac_raw2);

  printf("PASS (comparespots weighting)\n");
  return 0;
}
