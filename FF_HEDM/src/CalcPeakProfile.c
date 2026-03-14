// CalcPeakProfile.c — updated with tilt-transformed pixel quad support

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

// CalcPeakProfileParallel: compute area-weighted mean intensity for a radial
// profile bin.  When tilt parameters are provided (TRs != NULL), uses the
// actual pixel quadrilateral in remapped (Y,Z) space for exact area.
// Otherwise falls back to unit-square approximation.

inline void CalcPeakProfileParallel(int *Indices, int NrEachIndexBin, int idx,
                                    double *Average, double Rmi, double Rma,
                                    double EtaMi, double EtaMa, double ybc,
                                    double zbc, double px, int NrPixelsY,
                                    double *ReturnValue,
                                    double TRs[3][3], double Lsd, double RhoD,
                                    double p0, double p1, double p2, double p3,
                                    double p4, double p5) {
  double **EdgesIn = allocMatrix(50, 2);
  double **EdgesOut = allocMatrix(50, 2);

  double RmiPx = Rmi / px;
  double RmaPx = Rma / px;

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
                           p4, p5, px, 0, 0, 0, &Rt, &Eta, NULL);
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

    TotArea += ThisArea;
    SumIntensity += Average[Indices[i]] * ThisArea;
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
                               double p4, double p5) {
  double **EdgesIn = allocMatrix(50, 2);
  double **EdgesOut = allocMatrix(50, 2);

  double RmiPx = Rmi / px;
  double RmaPx = Rma / px;

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
                           p4, p5, px, 0, 0, 0, &Rt, &Eta, NULL);
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

    TotArea += ThisArea;
    SumIntensity += Average[Indices[i]] * ThisArea;
  }
  *outSumIntensity = SumIntensity;
  *outTotalArea = TotArea;
  FreeMemMatrix(EdgesIn, 50);
  FreeMemMatrix(EdgesOut, 50);
}
