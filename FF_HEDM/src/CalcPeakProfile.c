// CalcPeakProfile.c — updated with tilt-transformed pixel quad support
//                     and optional radial gradient correction (bilinear resampling)

#include <math.h>
#include <stdio.h>
#ifdef __linux__
#include <malloc.h>
#endif
#ifdef __MACH__
#include <stdlib.h>
#endif
#include "DetectorGeometry.h"
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823

#define TestBit(A, k) (A[(k / 32)] & (1 << (k % 32)))
extern size_t mapMaskSize;
extern int *mapMask;
extern int nRejects;
extern int nGood;

static inline double **allocMatrix(int nrows, int ncols) {
  double **arr;
  int i;
  arr = malloc(nrows * sizeof(*arr));
  if (arr == NULL) {
    return NULL;
  }
  for (i = 0; i < nrows; i++) {
    arr[i] = malloc(ncols * sizeof(*arr[i]));
    if (arr[i] == NULL) {
      return NULL;
    }
  }
  return arr;
}

static inline void FreeMemMatrix(double **mat, int nrows) {
  int r;
  for (r = 0; r < nrows; r++) {
    free(mat[r]);
  }
  free(mat);
}

/* ── Bilinear interpolation of Average at fractional (fy, fz) ──────────
   Returns interpolated intensity clamped to image boundaries. */
static inline double bilinear_sample(const double *Average, int NrPixelsY,
                                     int NrPixelsZ, double fy, double fz) {
  int iy0 = (int)floor(fy);
  int iz0 = (int)floor(fz);
  double dy = fy - iy0;
  double dz = fz - iz0;
  if (iy0 < 0)              { iy0 = 0;              dy = 0; }
  if (iy0 >= NrPixelsY - 1) { iy0 = NrPixelsY - 2;  dy = 1; }
  if (iz0 < 0)              { iz0 = 0;              dz = 0; }
  if (iz0 >= NrPixelsZ - 1) { iz0 = NrPixelsZ - 2;  dz = 1; }
  return Average[(size_t)iz0 * NrPixelsY + iy0]         * (1 - dy) * (1 - dz) +
         Average[(size_t)iz0 * NrPixelsY + iy0 + 1]     * dy       * (1 - dz) +
         Average[(size_t)(iz0 + 1) * NrPixelsY + iy0]   * (1 - dy) * dz       +
         Average[(size_t)(iz0 + 1) * NrPixelsY + iy0 + 1] * dy     * dz;
}

/* ── Read pixel intensity, optionally with gradient correction ──────────
   When gradientCorrection=1, shift the read position radially by −deltaR
   (where deltaR = R_pixel − R_bin_center) and bilinear-interpolate.
   This suppresses cardinal-angle aliasing. */
static inline double read_pixel_intensity(const double *Average, int NrPixelsY,
                                          int NrPixelsZ, int iy, int iz,
                                          double ybc, double zbc,
                                          double RbinCenterPx,
                                          int gradientCorrection) {
  if (!gradientCorrection) {
    return Average[(size_t)iz * NrPixelsY + iy];
  }
  double dy = (double)iy - ybc;
  double dz = (double)iz - zbc;
  double R = sqrt(dy * dy + dz * dz);
  if (R < 1.0) {
    return Average[(size_t)iz * NrPixelsY + iy];
  }
  double deltaR = R - RbinCenterPx;
  double read_y = (double)iy - deltaR * dy / R;
  double read_z = (double)iz - deltaR * dz / R;
  return bilinear_sample(Average, NrPixelsY, NrPixelsZ, read_y, read_z);
}

// CalcPeakProfileParallel: compute area-weighted mean intensity for a radial
// profile bin.  When tilt parameters are provided (TRs != NULL), uses the
// actual pixel quadrilateral in remapped (Y,Z) space for exact area.
// Otherwise falls back to unit-square approximation.
// When gradientCorrection=1, intensity is read via radial resampling at
// the R-bin center with bilinear interpolation.

inline void CalcPeakProfileParallel(int *Indices, int NrEachIndexBin, int idx,
                                    double *Average, double Rmi, double Rma,
                                    double EtaMi, double EtaMa, double ybc,
                                    double zbc, double px, int NrPixelsY,
                                    double *ReturnValue,
                                    double TRs[3][3], double Lsd, double RhoD,
                                    double p0, double p1, double p2, double p3,
                                    double p4, double p5, double p6,
                                    int gradientCorrection, int NrPixelsZ) {
  double **EdgesIn = allocMatrix(50, 2);
  double **EdgesOut = allocMatrix(50, 2);

  double RmiPx = Rmi / px;
  double RmaPx = Rma / px;
  double RbinCenterPx = (RmiPx + RmaPx) * 0.5;

  int i;
  double SumIntensity = 0, TotArea = 0;
  for (i = 0; i < NrEachIndexBin; i++) {
    if (mapMaskSize != 0) {
      if (TestBit(mapMask, Indices[i])) {
        *ReturnValue = 0;
        FreeMemMatrix(EdgesIn, 50);
        FreeMemMatrix(EdgesOut, 50);
        return;
      }
    }
    int iy = Indices[i] % NrPixelsY;
    int iz = Indices[i] / NrPixelsY;

    double ThisArea;
    if (TRs != NULL) {
      // Compute 4 pixel corners in remapped (Y,Z) space
      double cornerYZ[4][2];
      int k, l;
      for (k = 0; k < 2; k++) {
        for (l = 0; l < 2; l++) {
          double Y = (double)iy + dg_dy[k];
          double Z = (double)iz + dg_dz[l];
          double Rt, Eta;
          dg_pixel_to_REta(Y, Z, ybc, zbc, TRs, Lsd, RhoD, p0, p1, p2, p3,
                           p4, p5, p6, px, 0, 0, 0, &Rt, &Eta, NULL);
          dg_REta_to_YZ(Rt, Eta, &cornerYZ[k * 2 + l][0],
                        &cornerYZ[k * 2 + l][1]);
        }
      }
      ThisArea = dg_calc_pixel_bin_area_quad(cornerYZ, RmiPx, RmaPx, EtaMi,
                                             EtaMa, EdgesIn, EdgesOut);
    } else {
      // Fallback: unit-square (untilted)
      double pixY = -(double)(iy - ybc);
      double pixZ = (double)(iz - zbc);
      ThisArea = dg_calc_pixel_bin_area(pixY, pixZ, RmiPx, RmaPx, EtaMi,
                                        EtaMa, EdgesIn, EdgesOut);
    }

    double pixIntensity = read_pixel_intensity(Average, NrPixelsY, NrPixelsZ,
                                               iy, iz, ybc, zbc,
                                               RbinCenterPx,
                                               gradientCorrection);
    TotArea += ThisArea;
    SumIntensity += pixIntensity * ThisArea;
  }
  SumIntensity /= TotArea;
  if (TotArea == 0) {
    SumIntensity = 0;
  }
  FreeMemMatrix(EdgesIn, 50);
  FreeMemMatrix(EdgesOut, 50);
  *ReturnValue = SumIntensity;
}

inline void CalcPeakProfileRaw(int *Indices, int NrEachIndexBin, int idx,
                               double *Average, double Rmi, double Rma,
                               double EtaMi, double EtaMa, double ybc,
                               double zbc, double px, int NrPixelsY,
                               double *outSumIntensity, double *outTotalArea,
                               double TRs[3][3], double Lsd, double RhoD,
                               double p0, double p1, double p2, double p3,
                               double p4, double p5, double p6,
                               int gradientCorrection, int NrPixelsZ) {
  double **EdgesIn = allocMatrix(50, 2);
  double **EdgesOut = allocMatrix(50, 2);

  double RmiPx = Rmi / px;
  double RmaPx = Rma / px;
  double RbinCenterPx = (RmiPx + RmaPx) * 0.5;

  int i;
  double SumIntensity = 0, TotArea = 0;
  for (i = 0; i < NrEachIndexBin; i++) {
    if (mapMaskSize != 0) {
      if (TestBit(mapMask, Indices[i])) {
        continue;
      }
    }
    int iy = Indices[i] % NrPixelsY;
    int iz = Indices[i] / NrPixelsY;

    double ThisArea;
    if (TRs != NULL) {
      double cornerYZ[4][2];
      int k, l;
      for (k = 0; k < 2; k++) {
        for (l = 0; l < 2; l++) {
          double Y = (double)iy + dg_dy[k];
          double Z = (double)iz + dg_dz[l];
          double Rt, Eta;
          dg_pixel_to_REta(Y, Z, ybc, zbc, TRs, Lsd, RhoD, p0, p1, p2, p3,
                           p4, p5, p6, px, 0, 0, 0, &Rt, &Eta, NULL);
          dg_REta_to_YZ(Rt, Eta, &cornerYZ[k * 2 + l][0],
                        &cornerYZ[k * 2 + l][1]);
        }
      }
      ThisArea = dg_calc_pixel_bin_area_quad(cornerYZ, RmiPx, RmaPx, EtaMi,
                                             EtaMa, EdgesIn, EdgesOut);
    } else {
      double pixY = -(double)(iy - ybc);
      double pixZ = (double)(iz - zbc);
      ThisArea = dg_calc_pixel_bin_area(pixY, pixZ, RmiPx, RmaPx, EtaMi,
                                        EtaMa, EdgesIn, EdgesOut);
    }

    double pixIntensity = read_pixel_intensity(Average, NrPixelsY, NrPixelsZ,
                                               iy, iz, ybc, zbc,
                                               RbinCenterPx,
                                               gradientCorrection);
    TotArea += ThisArea;
    SumIntensity += pixIntensity * ThisArea;
  }
  *outSumIntensity = SumIntensity;
  *outTotalArea = TotArea;
  FreeMemMatrix(EdgesIn, 50);
  FreeMemMatrix(EdgesOut, 50);
}
