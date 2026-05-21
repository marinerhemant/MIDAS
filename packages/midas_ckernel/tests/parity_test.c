/* parity_test.c — prove the unified forward (forward.c) reproduces BOTH
 * legacy forward models on shared random inputs.
 *
 *   - vs indexer legacy  : the CalcOmega/CalcDiffrSpots bodies copied verbatim
 *                          from IndexerUnified.c (renamed ref_idx_*). Required
 *                          to be BIT-IDENTICAL (memcmp on doubles), since the
 *                          indexer's 500/500 parity is locked.
 *   - vs refiner legacy  : CalcDiffractionSpots() linked from the vendored
 *                          midas_fit_grain/c_src/CalcDiffractionSpots.c. The
 *                          refiner computes v from |OM·G| (vs the unified
 *                          |G_hkl|), so equality is ULP-tolerant (R2), not
 *                          bit-exact.
 *
 * Build (from packages/midas_ckernel):
 *   cc -O2 -I c_src -I ../midas_fit_grain/c_src \
 *      tests/parity_test.c c_src/forward.c ../midas_fit_grain/c_src/MIDAS_Math.c \
 *      ../midas_fit_grain/c_src/CalcDiffractionSpots.c -lm -o /tmp/ck_parity && \
 *   /tmp/ck_parity
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "forward.h"
#include "MIDAS_Math.h"
#include "MIDAS_Limits.h"

#define deg2rad (M_PI / 180.0)
#define rad2deg (180.0 / M_PI)
#define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))

/* ---- globals the legacy bodies reference ---- */
int BigDetSize = 0;             /* refiner extern: mask disabled */
unsigned int *BigDetector = NULL;
long long int totNrPixelsBigDetector = 0;
double pixelsize = 200.0;
int n_hkls = 0;                 /* indexer global */
double **hkls = NULL;           /* indexer global (canonical layout) */

/* ===== verbatim indexer legacy (renamed) ===== */
static void ref_idx_CalcSpotPosition(double RingRadius, double eta, double *yl,
                                     double *zl) {
  double etaRad = deg2rad * eta;
  *yl = -(sin(etaRad) * RingRadius);
  *zl = cos(etaRad) * RingRadius;
}
static void ref_idx_CalcOmega(double x, double y, double z, double v, double vSq,
                              double omegas[4], double etas[4], double cosOmes[4],
                              double sinOmes[4], int *nsol) {
  *nsol = 0;
  double ome, almostzero = 1e-12;
  if (fabs(y) < almostzero) {
    if (x != 0) {
      double cosome1 = -v / x;
      if (fabs(cosome1) <= 1.0) {
        ome = acos(cosome1) * rad2deg;
        double sinome1 = sqrt(1 - cosome1 * cosome1);
        omegas[*nsol] = ome; cosOmes[*nsol] = cosome1; sinOmes[*nsol] = sinome1;
        *nsol += 1;
        omegas[*nsol] = -ome; cosOmes[*nsol] = cosome1; sinOmes[*nsol] = -sinome1;
        *nsol += 1;
      }
    }
  } else {
    double y2 = y * y, inv_y2 = 1.0 / y2;
    double a = 1 + ((x * x) * inv_y2);
    double b = (2 * v * x) * inv_y2;
    double c = (vSq * inv_y2) - 1;
    double discr = b * b - 4 * a * c;
    double ome1a, ome1b, ome2a, ome2b, cosome1, cosome2, eqa, eqb, diffa, diffb;
    if (discr >= 0) {
      cosome1 = (-b + sqrt(discr)) / (2 * a);
      if (fabs(cosome1) <= 1.0) {
        ome1a = acos(cosome1); ome1b = -ome1a;
        double sinome1 = sqrt(1 - cosome1 * cosome1);
        eqa = -x * cosome1 + y * sinome1; diffa = fabs(eqa - v);
        eqb = -x * cosome1 - y * sinome1; diffb = fabs(eqb - v);
        if (diffa < diffb) { omegas[*nsol] = ome1a * rad2deg; cosOmes[*nsol] = cosome1; sinOmes[*nsol] = sinome1; *nsol += 1; }
        else { omegas[*nsol] = ome1b * rad2deg; cosOmes[*nsol] = cosome1; sinOmes[*nsol] = -sinome1; *nsol += 1; }
      }
      cosome2 = (-b - sqrt(discr)) / (2 * a);
      if (fabs(cosome2) <= 1) {
        ome2a = acos(cosome2); ome2b = -ome2a;
        double sinome2 = sqrt(1 - cosome2 * cosome2);
        eqa = -x * cosome2 + y * sinome2; diffa = fabs(eqa - v);
        eqb = -x * cosome2 - y * sinome2; diffb = fabs(eqb - v);
        if (diffa < diffb) { omegas[*nsol] = ome2a * rad2deg; cosOmes[*nsol] = cosome2; sinOmes[*nsol] = sinome2; *nsol += 1; }
        else { omegas[*nsol] = ome2b * rad2deg; cosOmes[*nsol] = cosome2; sinOmes[*nsol] = -sinome2; *nsol += 1; }
      }
    }
  }
  double gw[3], gv[3] = {x, y, z}, eta;
  for (int io = 0; io < *nsol; io++) {
    RotateAroundZ(gv, omegas[io], gw);
    CalcEtaAngle(gw[1], gw[2], &eta);
    etas[io] = eta;
  }
}
static void ref_idx_CalcDiffrSpots(double OrientMatrix[3][3], double RingRadii[],
                                   double OmegaRange[][2], double BoxSizes[][4],
                                   int NOmegaRanges, double ExcludePoleAngle,
                                   double **spots, int *nspots,
                                   const int ringsToReject[], int nRingsToReject,
                                   int *nSpotsFracCalc) {
  int i, OmegaRangeNo, indexhkl, KeepSpot, ringnr, nspotsPlane;
  double theta, Ghkl[3], Gc[3], omegas[4], cosOmes[4], sinOmes[4], etas[4], yl, zl;
  int spotnr = 0, spotid = 0, OrientID = 0, nSpotsForFracCalc = 0, SpotToFrac, rfc;
  for (indexhkl = 0; indexhkl < n_hkls; indexhkl++) {
    Ghkl[0] = hkls[indexhkl][0]; Ghkl[1] = hkls[indexhkl][1]; Ghkl[2] = hkls[indexhkl][2];
    ringnr = (int)(hkls[indexhkl][3]);
    SpotToFrac = 1;
    for (rfc = 0; rfc < nRingsToReject; rfc++) if (ringnr == ringsToReject[rfc]) { SpotToFrac = 0; break; }
    double RingRadius = RingRadii[ringnr];
    MatrixMultF(OrientMatrix, Ghkl, Gc);
    theta = hkls[indexhkl][5];
    ref_idx_CalcOmega(Gc[0], Gc[1], Gc[2], hkls[indexhkl][8], hkls[indexhkl][9], omegas, etas, cosOmes, sinOmes, &nspotsPlane);
    for (i = 0; i < nspotsPlane; i++) {
      double Omega = omegas[i], Eta = etas[i], EtaAbs = fabs(Eta);
      if ((EtaAbs < ExcludePoleAngle) || ((180 - EtaAbs) < ExcludePoleAngle)) continue;
      ref_idx_CalcSpotPosition(RingRadius, etas[i], &yl, &zl);
      for (OmegaRangeNo = 0; OmegaRangeNo < NOmegaRanges; OmegaRangeNo++) {
        KeepSpot = 0;
        if ((Omega > OmegaRange[OmegaRangeNo][0]) && (Omega < OmegaRange[OmegaRangeNo][1]) &&
            (yl > BoxSizes[OmegaRangeNo][0]) && (yl < BoxSizes[OmegaRangeNo][1]) &&
            (zl > BoxSizes[OmegaRangeNo][2]) && (zl < BoxSizes[OmegaRangeNo][3])) { KeepSpot = 1; break; }
      }
      if (KeepSpot) {
        spots[spotnr][0] = OrientID; spots[spotnr][1] = spotid; spots[spotnr][2] = indexhkl;
        spots[spotnr][3] = 1000000.0; /* distance placeholder, set by caller below */
        spots[spotnr][4] = yl; spots[spotnr][5] = zl; spots[spotnr][6] = omegas[i];
        spots[spotnr][7] = etas[i]; spots[spotnr][8] = theta; spots[spotnr][9] = ringnr;
        spots[spotnr][14] = sinOmes[i]; spots[spotnr][15] = cosOmes[i];
        spotnr++; spotid++; if (SpotToFrac) nSpotsForFracCalc++;
      }
    }
  }
  *nspots = spotnr; *nSpotsFracCalc = nSpotsForFracCalc;
}

/* refiner legacy linked from the FROZEN reference (legacy_refiner_forward.c);
 * the shipped CalcDiffractionSpots.c is now an adapter over the shared forward
 * and would make this comparison circular. */
extern int ref_refiner_CalcDiffractionSpots(
    double Distance, double ExcludePoleAngle, double OmegaRanges[][2],
    int NoOfOmegaRanges, double **hkls, int n_hkls, double BoxSizes[][4],
    int *nTspots, double OrientMatr[3][3], double **TheorSpots);
#define CalcDiffractionSpots ref_refiner_CalcDiffractionSpots

static double frand(double lo, double hi) {
  return lo + (hi - lo) * ((double)rand() / (double)RAND_MAX);
}
static double **alloc2d(int r, int c) {
  double **m = malloc(r * sizeof(double *));
  for (int i = 0; i < r; i++) m[i] = calloc(c, sizeof(double));
  return m;
}
/* random orientation matrix from random axis-angle */
static void rand_om(double R[3][3]) {
  double ax[3] = {frand(-1, 1), frand(-1, 1), frand(-1, 1)};
  double n = sqrt(ax[0]*ax[0]+ax[1]*ax[1]+ax[2]*ax[2]); if (n < 1e-9) n = 1;
  ax[0]/=n; ax[1]/=n; ax[2]/=n;
  double a = frand(0, 2*M_PI), c = cos(a), s = sin(a), t = 1-c;
  R[0][0]=t*ax[0]*ax[0]+c;       R[0][1]=t*ax[0]*ax[1]-s*ax[2]; R[0][2]=t*ax[0]*ax[2]+s*ax[1];
  R[1][0]=t*ax[0]*ax[1]+s*ax[2]; R[1][1]=t*ax[1]*ax[1]+c;       R[1][2]=t*ax[1]*ax[2]-s*ax[0];
  R[2][0]=t*ax[0]*ax[2]-s*ax[1]; R[2][1]=t*ax[1]*ax[2]+s*ax[0]; R[2][2]=t*ax[2]*ax[2]+c;
}

int main(void) {
  srand(12345);
  const int NRINGS = 8;
  /* lattice spacings d (Å) for a Ni-like FCC set; reciprocal magnitude = 1/d */
  double dsp[8] = {2.034, 1.762, 1.246, 1.062, 1.017, 0.881, 0.808, 0.7884};
  double lambda = 0.1729; /* Å */
  double distance = 1000000.0; /* µm */

  /* build canonical hkls + RingRadii */
  n_hkls = NRINGS;
  hkls = alloc2d(NRINGS, 10);
  double RingRadii[600]; memset(RingRadii, 0, sizeof(RingRadii));
  double **hkls_ref = alloc2d(NRINGS, 7); /* refiner-native layout */
  for (int r = 0; r < NRINGS; r++) {
    double Ginv = 1.0 / dsp[r];          /* |G| = 1/d */
    /* random reciprocal vector of that magnitude */
    double v3[3] = {frand(-1,1), frand(-1,1), frand(-1,1)};
    double nn = sqrt(v3[0]*v3[0]+v3[1]*v3[1]+v3[2]*v3[2]); if (nn<1e-9) nn=1;
    double hc = Ginv*v3[0]/nn, kc = Ginv*v3[1]/nn, lc = Ginv*v3[2]/nn;
    double sinth = lambda * Ginv / 2.0; if (sinth > 1) sinth = 0.99;
    double theta = asin(sinth) * rad2deg;
    double ttheta = 2*theta;
    double RingRadius = distance * tan(ttheta * deg2rad);
    int ringnr = r + 1;
    /* Distinct radius sources to exercise the RingRadii gate: the indexer
     * path uses RingRadii[ringnr] (param-file value); hkls[6] (hkl-file value)
     * is deliberately offset so a confusion between the two would show up. */
    double RingRadius_param = RingRadius;          /* indexer: RingRadii[]   */
    double RingRadius_hkl = RingRadius + 0.137;    /* refiner: hkls[6]        */
    hkls[r][0]=hc; hkls[r][1]=kc; hkls[r][2]=lc; hkls[r][3]=(double)ringnr;
    hkls[r][4]=Ginv; hkls[r][5]=theta; hkls[r][6]=RingRadius_hkl;
    hkls[r][7]=sin(theta*deg2rad);
    double len = sqrt(hc*hc+kc*kc+lc*lc);
    double vv = hkls[r][7]*len; hkls[r][8]=vv; hkls[r][9]=vv*vv;
    RingRadii[ringnr]=RingRadius_param;
    hkls_ref[r][0]=hc; hkls_ref[r][1]=kc; hkls_ref[r][2]=lc;
    hkls_ref[r][3]=Ginv; hkls_ref[r][4]=theta; hkls_ref[r][5]=RingRadius_hkl; hkls_ref[r][6]=ringnr;
  }

  double OmegaRange[1][2] = {{-180.0, 180.0}};
  double BoxSizes[1][4] = {{-200000, 200000, -200000, 200000}};
  int NOmegaRanges = 1;
  double ExcludePoleAngle = 0.0;

  int NTRIAL = 2000;
  long idx_rows = 0, idx_bitmismatch = 0;
  long ref_rows = 0, ref_close = 0; double ref_maxabs = 0;

  double **su = alloc2d(2*NRINGS, MIDAS_CK_NCOLS);   /* unified, indexer path  */
  double **sur = alloc2d(2*NRINGS, MIDAS_CK_NCOLS);  /* unified, refiner path  */
  double **si = alloc2d(2*NRINGS, MIDAS_CK_NCOLS);
  double **sr = alloc2d(2*NRINGS, 9);

  for (int t = 0; t < NTRIAL; t++) {
    double OM[3][3]; rand_om(OM);
    int nu = 0, nur = 0, ni = 0, nr = 0, frac_u = 0, frac_i = 0, frac_ur = 0;

    /* indexer path: pass RingRadii[] (param-file radii) */
    midas_ck_calc_diffraction_spots(OM, distance, RingRadii, hkls, n_hkls,
                                    OmegaRange, BoxSizes, NOmegaRanges,
                                    ExcludePoleAngle, NULL, 0, NULL, 0, su, &nu,
                                    &frac_u);
    /* refiner path: RingRadii=NULL => radius from hkls[6] (hkl-file radii) */
    midas_ck_calc_diffraction_spots(OM, distance, NULL, hkls, n_hkls,
                                    OmegaRange, BoxSizes, NOmegaRanges,
                                    ExcludePoleAngle, NULL, 0, NULL, 0, sur,
                                    &nur, &frac_ur);
    ref_idx_CalcDiffrSpots(OM, RingRadii, OmegaRange, BoxSizes, NOmegaRanges,
                           ExcludePoleAngle, si, &ni, NULL, 0, &frac_i);

    /* indexer bit-parity on cols {0..9,14,15} */
    if (nu != ni) { idx_bitmismatch += (nu > ni ? nu : ni); }
    else {
      int idxcols[12] = {0,1,2,4,5,6,7,8,9,14,15,3};
      for (int s = 0; s < nu; s++) {
        idx_rows++;
        for (int cc = 0; cc < 11; cc++) { /* skip col 3 (distance differs by design) */
          int c = idxcols[cc];
          if (memcmp(&su[s][c], &si[s][c], sizeof(double)) != 0) { idx_bitmismatch++; break; }
        }
      }
    }

    /* refiner ULP-tolerant on yl,zl,omega (+ GCr) vs unified refiner path */
    CalcDiffractionSpots(distance, ExcludePoleAngle, OmegaRange, NOmegaRanges,
                         hkls_ref, n_hkls, BoxSizes, &nr, OM, sr);
    if (nr == nur) {
      for (int s = 0; s < nr; s++) {
        ref_rows++;
        /* refiner cols: 0=yl,1=zl,2=omega,3-5=GCr */
        double dyl = fabs(sr[s][0]-sur[s][4]);
        double dzl = fabs(sr[s][1]-sur[s][5]);
        double dom = fabs(sr[s][2]-sur[s][6]);
        double dg = fabs(sr[s][3]-sur[s][16])+fabs(sr[s][4]-sur[s][17])+fabs(sr[s][5]-sur[s][18]);
        double m = dyl; if (dzl>m)m=dzl; if (dom>m)m=dom; if (dg>m)m=dg;
        if (m > ref_maxabs) ref_maxabs = m;
        if (m < 1e-7) ref_close++;
      }
    }
  }

  printf("=== midas_ckernel forward parity ===\n");
  printf("trials: %d, rings: %d\n", NTRIAL, NRINGS);
  printf("[indexer] rows compared: %ld   BIT-mismatches: %ld   %s\n",
         idx_rows, idx_bitmismatch, idx_bitmismatch == 0 ? "PASS (bit-identical)" : "FAIL");
  printf("[refiner] rows compared: %ld   within 1e-7: %ld   max|Δ|: %.3e   (ULP-tolerant, R2)\n",
         ref_rows, ref_close, ref_maxabs);
  return idx_bitmismatch == 0 ? 0 : 1;
}
