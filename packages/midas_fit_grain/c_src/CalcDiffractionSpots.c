//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
//  CalcDiffractionSpots.c — refiner-side ADAPTER over the shared forward model.
//
//  Originally a standalone copy of CalcDiffrSpots_Furnace. It now forwards to
//  the single shared simulator midas_ck_calc_diffraction_spots() (forward.c,
//  canonical source packages/midas_ckernel), so the c-omp indexer and refiner
//  use ONE forward model. The legacy 9-column output contract this refiner
//  depends on is preserved by remapping the shared 19-column row:
//
//     legacy refiner col   shared row col   meaning
//     ------------------   --------------   -------------------------
//        [0] yl                 [4]         lab-frame detector y (µm)
//        [1] zl                 [5]         lab-frame detector z (µm)
//        [2] omega              [6]         omega (deg)
//        [3] GCr0               [16]        Ds·Gc0/|Gc| (spatial objective)
//        [4] GCr1               [17]
//        [5] GCr2               [18]
//        [6] distance           [3]         sample-to-detector (µm)
//        [7] RingNr             [9]
//        [8] nrhkls             (spotid)    running id (informational)
//
//  The refiner draws RingRadius from its own hkl table, so RingRadii is passed
//  NULL (radius read from canonical hkls[*][6]); the BigDetector active-area
//  mask is forwarded from the legacy externs.
//
//  Created by Hemant Sharma on 12/3/13. Adapter form 2026-05-21.
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "MIDAS_Limits.h"
#include "forward.h"

// Legacy BigDetector externs (defined in the refiner translation unit).
extern int BigDetSize;
extern int *BigDetector;
extern long long int totNrPixelsBigDetector;
extern double pixelsize;

int CalcDiffractionSpots(double Distance, double ExcludePoleAngle,
                         double OmegaRanges[MAX_N_OMEGA_RANGES][2],
                         int NoOfOmegaRanges, double **hkls, int n_hkls,
                         double BoxSizes[MAX_N_OMEGA_RANGES][4], int *nTspots,
                         double OrientMatr[3][3], double **TheorSpots) {
  *nTspots = 0;
  if (TheorSpots == NULL) {
    printf("Memory error: could not allocate memory for output matrix. Memory "
           "full?\n");
    return 1;
  }

  // Convert the refiner-native hkl table ([0..2]=G,[3]=Ds,[4]=theta,
  // [5]=RingRadius,[6]=RingNr) to the canonical layout the shared forward
  // expects. Allocated per call for OpenMP thread-safety (n_hkls is small;
  // the refiner recomputes hkls each strain evaluation anyway).
  double **hk = (double **)malloc((size_t)n_hkls * sizeof(double *));
  double *hkbuf = (double *)malloc((size_t)n_hkls * 10 * sizeof(double));
  double **out = (double **)malloc((size_t)(2 * n_hkls) * sizeof(double *));
  double *outbuf =
      (double *)malloc((size_t)(2 * n_hkls) * MIDAS_CK_NCOLS * sizeof(double));
  if (!hk || !hkbuf || !out || !outbuf) {
    free(hk); free(hkbuf); free(out); free(outbuf);
    printf("Memory error in CalcDiffractionSpots adapter.\n");
    return 1;
  }
  for (int i = 0; i < n_hkls; i++) hk[i] = hkbuf + (size_t)i * 10;
  for (int i = 0; i < 2 * n_hkls; i++)
    out[i] = outbuf + (size_t)i * MIDAS_CK_NCOLS;

  midas_ck_hkls_from_refiner(hkls, n_hkls, hk);

  MidasCkBigDet bd = {BigDetSize, (const unsigned int *)BigDetector, pixelsize};
  const MidasCkBigDet *bdp = (BigDetSize != 0 && BigDetector) ? &bd : NULL;

  int nsp = 0;
  midas_ck_calc_diffraction_spots(OrientMatr, Distance, /*RingRadii=*/NULL, hk,
                                  n_hkls, OmegaRanges, BoxSizes, NoOfOmegaRanges,
                                  ExcludePoleAngle, /*ringsToReject=*/NULL,
                                  /*nRingsToReject=*/0, bdp, /*orient_id=*/0,
                                  out, &nsp, /*nSpotsFracCalc=*/NULL);

  // Remap shared 19-col rows -> legacy refiner 9-col contract.
  // col 8 = legacy `nrhkls`: a STABLE per-reflection id = indexhkl*2 + 1 + k,
  // where k is the kept-spot index within that hkl (the two ±ω solutions).
  // This MUST be anchored to the HKL index (out[s][2]=indexhkl), NOT to
  // midas_ck's running spot counter (out[s][1]) — the running counter
  // renumbers whenever the set of kept spots changes (different orient/strain),
  // which breaks the obs↔theor spot-id matching the position objective relies
  // on across forward evaluations. midas_ck emits spots grouped by indexhkl.
  int prev_hkl = -1, within = 0;
  for (int s = 0; s < nsp; s++) {
    int ih = (int)out[s][2];        // indexhkl
    if (ih == prev_hkl) {
      within++;
    } else {
      within = 0;
      prev_hkl = ih;
    }
    TheorSpots[s][0] = out[s][4];   // yl
    TheorSpots[s][1] = out[s][5];   // zl
    TheorSpots[s][2] = out[s][6];   // omega
    TheorSpots[s][3] = out[s][16];  // GCr0
    TheorSpots[s][4] = out[s][17];  // GCr1
    TheorSpots[s][5] = out[s][18];  // GCr2
    TheorSpots[s][6] = out[s][3];   // distance
    TheorSpots[s][7] = out[s][9];   // RingNr
    TheorSpots[s][8] = ih * 2 + 1 + within; // stable nrhkls (legacy scheme)
  }
  *nTspots = nsp;

  free(hk); free(hkbuf); free(out); free(outbuf);
  return 0;
}
