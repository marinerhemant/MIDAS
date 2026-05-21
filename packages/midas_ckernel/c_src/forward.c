/* forward.c — MIDAS shared diffraction forward model. See forward.h.
 *
 * The CalcOmega / CalcSpotPosition kernels are byte-for-byte the indexer's
 * (IndexerUnified.c) precomputed-v form, so re-pointing the indexer at this
 * file keeps its 500/500 bit-level parity. The main loop adds the refiner's
 * BigDetector active-area mask and the scaled reciprocal vector GCr (cols
 * 16-18), both gated so the indexer path is unaffected.
 */
#include "forward.h"

#include <math.h>
#include <stdlib.h>

#include "MIDAS_Math.h"

#define deg2rad (M_PI / 180.0)
#define rad2deg (180.0 / M_PI)
#define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))

/* CalcSpotPosition — identical in both legacy copies. */
void midas_ck_calc_spot_position(RealType RingRadius, RealType eta,
                                 RealType *yl, RealType *zl) {
  RealType etaRad = deg2rad * eta;
  *yl = -(sin(etaRad) * RingRadius);
  *zl = cos(etaRad) * RingRadius;
}
#define CalcSpotPosition midas_ck_calc_spot_position

/* CalcOmega — indexer precomputed-v form (IndexerUnified.c:331). Takes v, vSq;
 * outputs omegas/etas plus cosOmes/sinOmes. */
void midas_ck_calc_omega(RealType x, RealType y, RealType z, RealType v,
                         RealType vSq, RealType omegas[4], RealType etas[4],
                         RealType cosOmes[4], RealType sinOmes[4], int *nsol) {
  *nsol = 0;
  RealType ome;
  RealType almostzero = 1e-12;
  if (fabs(y) < almostzero) {
    if (x != 0) {
      RealType cosome1 = -v / x;
      if (fabs(cosome1) <= 1.0) {
        ome = acos(cosome1) * rad2deg;
        RealType sinome1 = sqrt(1 - cosome1 * cosome1);
        omegas[*nsol] = ome;
        cosOmes[*nsol] = cosome1;
        sinOmes[*nsol] = sinome1;
        *nsol = *nsol + 1;
        omegas[*nsol] = -ome;
        cosOmes[*nsol] = cosome1;
        sinOmes[*nsol] = -sinome1;
        *nsol = *nsol + 1;
      }
    }
  } else {
    RealType y2 = y * y;
    RealType inv_y2 = 1.0 / y2;
    RealType a = 1 + ((x * x) * inv_y2);
    RealType b = (2 * v * x) * inv_y2;
    RealType c = (vSq * inv_y2) - 1;
    RealType discr = b * b - 4 * a * c;
    RealType ome1a, ome1b, ome2a, ome2b, cosome1, cosome2;
    RealType eqa, eqb, diffa, diffb;
    if (discr >= 0) {
      cosome1 = (-b + sqrt(discr)) / (2 * a);
      if (fabs(cosome1) <= 1.0) {
        ome1a = acos(cosome1);
        ome1b = -ome1a;
        RealType sinome1 = sqrt(1 - cosome1 * cosome1);
        eqa = -x * cosome1 + y * sinome1;
        diffa = fabs(eqa - v);
        eqb = -x * cosome1 - y * sinome1;
        diffb = fabs(eqb - v);
        if (diffa < diffb) {
          omegas[*nsol] = ome1a * rad2deg;
          cosOmes[*nsol] = cosome1;
          sinOmes[*nsol] = sinome1;
          *nsol = *nsol + 1;
        } else {
          omegas[*nsol] = ome1b * rad2deg;
          cosOmes[*nsol] = cosome1;
          sinOmes[*nsol] = -sinome1;
          *nsol = *nsol + 1;
        }
      }
      cosome2 = (-b - sqrt(discr)) / (2 * a);
      if (fabs(cosome2) <= 1) {
        ome2a = acos(cosome2);
        ome2b = -ome2a;
        RealType sinome2 = sqrt(1 - cosome2 * cosome2);
        eqa = -x * cosome2 + y * sinome2;
        diffa = fabs(eqa - v);
        eqb = -x * cosome2 - y * sinome2;
        diffb = fabs(eqb - v);
        if (diffa < diffb) {
          omegas[*nsol] = ome2a * rad2deg;
          cosOmes[*nsol] = cosome2;
          sinOmes[*nsol] = sinome2;
          *nsol = *nsol + 1;
        } else {
          omegas[*nsol] = ome2b * rad2deg;
          cosOmes[*nsol] = cosome2;
          sinOmes[*nsol] = -sinome2;
          *nsol = *nsol + 1;
        }
      }
    }
  }
  RealType gw[3];
  RealType gv[3] = {x, y, z};
  RealType eta;
  int indexOme;
  for (indexOme = 0; indexOme < *nsol; indexOme++) {
    RotateAroundZ(gv, omegas[indexOme], gw);
    CalcEtaAngle(gw[1], gw[2], &eta);
    etas[indexOme] = eta;
  }
}

#define CalcOmega midas_ck_calc_omega

int midas_ck_calc_diffraction_spots(
    RealType OrientMatrix[3][3], RealType distance, const RealType *RingRadii,
    double **hkls, int n_hkls, RealType OmegaRange[][2], RealType BoxSizes[][4],
    int NOmegaRanges, RealType ExcludePoleAngle, const int *ringsToReject,
    int nRingsToReject, const MidasCkBigDet *bigdet, int orient_id,
    RealType **spots, int *nspots, int *nSpotsFracCalc) {
  if (!spots) return 1;
  int i, OmegaRangeNo, indexhkl;
  RealType Gc[3], omegas[4], cosOmes[4], sinOmes[4], etas[4], yl, zl;
  int nspotsPlane;
  int spotnr = 0, spotid = 0;
  int nSpotsForFracCalc = 0;
  int useBigDet = (bigdet && bigdet->big_det_size != 0 && bigdet->mask);

  for (indexhkl = 0; indexhkl < n_hkls; indexhkl++) {
    double Ghkl[3] = {hkls[indexhkl][0], hkls[indexhkl][1], hkls[indexhkl][2]};
    int ringnr = (int)(hkls[indexhkl][3]);
    RealType Ds = hkls[indexhkl][4];
    RealType theta = hkls[indexhkl][5];
    /* Indexer passes its param-file RingRadii[] (indexed by ringnr); refiner
     * passes NULL and the radius comes from the hkl table (hkls[*][6]). */
    RealType RingRadius = RingRadii ? RingRadii[ringnr] : hkls[indexhkl][6];

    /* RingsToReject fraction tally (indexer). */
    int SpotToFrac = 1;
    for (int rc = 0; rc < nRingsToReject; rc++) {
      if (ringnr == ringsToReject[rc]) { SpotToFrac = 0; break; }
    }

    MatrixMultF(OrientMatrix, Ghkl, Gc);
    /* v, vSq precomputed in canonical layout (R2: indexer's |G_hkl| form). */
    CalcOmega(Gc[0], Gc[1], Gc[2], hkls[indexhkl][8], hkls[indexhkl][9], omegas,
              etas, cosOmes, sinOmes, &nspotsPlane);

    /* Scaled reciprocal unit vector GCr (refiner spatial objective). */
    RealType NGc = sqrt(Gc[0] * Gc[0] + Gc[1] * Gc[1] + Gc[2] * Gc[2]);
    RealType GCr0 = Ds * Gc[0] / NGc;
    RealType GCr1 = Ds * Gc[1] / NGc;
    RealType GCr2 = Ds * Gc[2] / NGc;

    for (i = 0; i < nspotsPlane; i++) {
      RealType Omega = omegas[i];
      RealType Eta = etas[i];
      if (isnan(Omega) || isnan(Eta)) continue;
      RealType EtaAbs = fabs(Eta);
      if ((EtaAbs < ExcludePoleAngle) || ((180 - EtaAbs) < ExcludePoleAngle))
        continue;
      CalcSpotPosition(RingRadius, etas[i], &yl, &zl);

      int KeepSpot = 0;
      for (OmegaRangeNo = 0; OmegaRangeNo < NOmegaRanges; OmegaRangeNo++) {
        if ((Omega > OmegaRange[OmegaRangeNo][0]) &&
            (Omega < OmegaRange[OmegaRangeNo][1]) &&
            (yl > BoxSizes[OmegaRangeNo][0]) &&
            (yl < BoxSizes[OmegaRangeNo][1]) &&
            (zl > BoxSizes[OmegaRangeNo][2]) &&
            (zl < BoxSizes[OmegaRangeNo][3])) {
          KeepSpot = 1;
          break;
        }
      }
      /* BigDetector active-area mask (refiner). */
      if (KeepSpot && useBigDet) {
        int YCInt = (int)floor((bigdet->big_det_size / 2) -
                               (int)(-yl / bigdet->pixelsize));
        int ZCInt = (int)floor((int)(zl / bigdet->pixelsize) +
                               (bigdet->big_det_size / 2));
        long long int idx =
            (long long int)(YCInt + bigdet->big_det_size * ZCInt);
        if (!TestBit(bigdet->mask, idx)) KeepSpot = 0;
      }

      if (KeepSpot) {
        spots[spotnr][0] = orient_id;
        spots[spotnr][1] = spotid;
        spots[spotnr][2] = indexhkl;
        spots[spotnr][3] = distance;
        spots[spotnr][4] = yl;
        spots[spotnr][5] = zl;
        spots[spotnr][6] = omegas[i];
        spots[spotnr][7] = etas[i];
        spots[spotnr][8] = theta;
        spots[spotnr][9] = ringnr;
        spots[spotnr][14] = sinOmes[i];
        spots[spotnr][15] = cosOmes[i];
        spots[spotnr][16] = GCr0;
        spots[spotnr][17] = GCr1;
        spots[spotnr][18] = GCr2;
        spotnr++;
        spotid++;
        if (SpotToFrac) nSpotsForFracCalc++;
      }
    }
  }
  *nspots = spotnr;
  if (nSpotsFracCalc) *nSpotsFracCalc = nSpotsForFracCalc;
  return 0;
}

void midas_ck_hkls_from_refiner(double **in, int n_hkls, double **out) {
  for (int i = 0; i < n_hkls; i++) {
    double hc = in[i][0], kc = in[i][1], lc = in[i][2];
    double Ds = in[i][3], theta = in[i][4], RingRadius = in[i][5];
    double RingNr = in[i][6];
    out[i][0] = hc;
    out[i][1] = kc;
    out[i][2] = lc;
    out[i][3] = RingNr;
    out[i][4] = Ds;
    out[i][5] = theta;
    out[i][6] = RingRadius;
    out[i][7] = sin(theta * deg2rad);
    double len = sqrt(hc * hc + kc * kc + lc * lc);
    double v = out[i][7] * len;
    out[i][8] = v;
    out[i][9] = v * v;
  }
}
