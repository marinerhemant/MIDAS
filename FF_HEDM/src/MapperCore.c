//
// MapperCore.c — Green's theorem pixel→(R,Eta) bin mapping
//
// Extracted from DetectorMapper.c. See MapperCore.h for API docs.
//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#include "MapperCore.h"
#include "DetectorGeometry.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --------------------------------------------------------------------------
// mapper_inverse_transform_pixel
// --------------------------------------------------------------------------
void mapper_inverse_transform_pixel(int y_in, int z_in, int *y_out, int *z_out,
                                    int NrTransOpt, const int TransOpt[10],
                                    int NrPixelsY, int NrPixelsZ) {
  int y = y_in, z = z_in;
  /* Apply inverse transformations in reverse order.
     For flips the inverse is the same operation, so we just
     iterate in reverse. */
  for (int t = NrTransOpt - 1; t >= 0; t--) {
    if (TransOpt[t] == 1) { /* Flip LR (invert Y) */
      y = NrPixelsY - 1 - y;
    } else if (TransOpt[t] == 2) { /* Flip TB (invert Z) */
      z = NrPixelsZ - 1 - z;
    } else if (TransOpt[t] == 3) { /* Transpose */
      int tmp = y;
      y = z;
      z = tmp;
    }
  }
  *y_out = y;
  *z_out = z;
}

// --------------------------------------------------------------------------
// mapper_build_map — Green's theorem sub-pixel area-weighted mapping
// --------------------------------------------------------------------------
long long mapper_build_map(
    double tx, double ty, double tz,
    int NrPixelsY, int NrPixelsZ,
    double pxY, double pxZ,
    double Ycen, double Zcen, double Lsd, double RhoD,
    double p0, double p1, double p2, double p3, double p4, double p5, double p6,
    double p7, double p8, double p9, double p10,
    double *EtaBinsLow, double *EtaBinsHigh,
    double *RBinsLow, double *RBinsHigh,
    int nRBins, int nEtaBins,
    struct MapPixelData ***pxList, int **nPxList, int **maxnPx,
    double *mask,
    int *binMaskFlag,
    int NrTransOpt, const int TransOpt[10],
    omp_lock_t *binLocks,
    int SubPixelLevel, double SubPixelCardinalWidth,
    double parallax,
    int solidAngleCorr, int polarizationCorr, double polFraction,
    const double *distortionMapY, const double *distortionMapZ,
    const Panel *mapPanels, int mapNPanels,
    const DGResidualCorr *residualCorr)
{
  double TRs[3][3];
  dg_build_tilt_matrix(tx, ty, tz, TRs);
  int i;
  long long int TotNrOfBins = 0;
  long long int sumNrBins = 0;
  long long int nrContinued1 = 0;
  long long int nrContinued2 = 0;
  long long int nrContinued3 = 0;
#pragma omp parallel for schedule(dynamic, 16) reduction(                      \
        + : TotNrOfBins, sumNrBins, nrContinued1, nrContinued2, nrContinued3)
  for (i = 0; i < NrPixelsY; i++) {
    // Thread-private allocations
    double RetVals[2], RetVals2[2];
    double **Edges = dg_alloc_matrix(50, 2);
    double **EdgesOut = dg_alloc_matrix(50, 2);
    double boxEdge[4][2];
    double YZ[2];
    double cornerYZ[4][2];
    int *RChosen = malloc(nRBins * sizeof(int));
    int *EtaChosen = malloc(nEtaBins * sizeof(int));
    int j, k, l, m;
    for (j = 0; j < NrPixelsZ; j++) {
      long long int testPos = j;
      testPos *= NrPixelsY;
      testPos += i;
      int pixelIsMasked = (mask != NULL && mask[testPos] == 1.0);
      double pdY = 0, pdZ = 0;
      double dLsd = 0, dP2 = 0;
      int pIdx = GetPanelIndex((double)i, (double)j, mapNPanels, (Panel *)mapPanels);
      if (pIdx >= 0) {
        ApplyPanelCorrection((double)i, (double)j, &mapPanels[pIdx], &pdY, &pdZ);
        pdY -= (double)i;
        pdZ -= (double)j;
        dLsd = mapPanels[pIdx].dLsd;
        dP2 = mapPanels[pIdx].dP2;
      }
      double ypr = (double)i + distortionMapY[testPos] + pdY;
      double zpr = (double)j + distortionMapZ[testPos] + pdZ;

      // Determine sub-pixel level for this pixel.
      int spLevel = 1; // default: no splitting
      if (SubPixelLevel > 1) {
        double Rt_cen, Eta_cen;
        dg_pixel_to_REta_corr(ypr, zpr, Ycen, Zcen, TRs, Lsd, RhoD, p0, p1, p2,
                         p3, p4, p5, p6, p7, p8, p9, p10, pxY, dLsd, dP2, parallax,
                         residualCorr, &Rt_cen, &Eta_cen,
                         NULL);
        double absEta = fabs(Eta_cen);
        if (absEta <= SubPixelCardinalWidth ||
            fabs(absEta - 90.0) <= SubPixelCardinalWidth ||
            fabs(absEta - 180.0) <= SubPixelCardinalWidth) {
          spLevel = SubPixelLevel;
        }
      }

      // Sub-pixel loop
      for (int si = 0; si < spLevel; si++) {
        for (int sj = 0; sj < spLevel; sj++) {
          double sp_dy_lo = (double)si / spLevel - 0.5;
          double sp_dy_hi = (double)(si + 1) / spLevel - 0.5;
          double sp_dz_lo = (double)sj / spLevel - 0.5;
          double sp_dz_hi = (double)(sj + 1) / spLevel - 0.5;
          double sp_cy = ((double)(2 * si + 1) / (2.0 * spLevel)) - 0.5;
          double sp_cz = ((double)(2 * sj + 1) / (2.0 * spLevel)) - 0.5;

          double sp_dy[2] = {sp_dy_lo, sp_dy_hi};
          double sp_dz[2] = {sp_dz_lo, sp_dz_hi};

          double EtaMi = 1800;
          double EtaMa = -1800;
          double RMi = 1E8;
          double RMa = -1000;
          double Eta, Rt;
          for (k = 0; k < 2; k++) {
            for (l = 0; l < 2; l++) {
              double Y = ypr + sp_dy[k];
              double Z = zpr + sp_dz[l];
              dg_pixel_to_REta_corr(Y, Z, Ycen, Zcen, TRs, Lsd, RhoD, p0, p1, p2,
                               p3, p4, p5, p6, p7, p8, p9, p10, pxY, dLsd, dP2, parallax,
                               residualCorr, &Rt, &Eta,
                               NULL);
              RetVals[0] = Eta;
              RetVals[1] = Rt;
              if (Eta < EtaMi) EtaMi = Eta;
              if (Eta > EtaMa) EtaMa = Eta;
              if (Rt < RMi) RMi = Rt;
              if (Rt > RMa) RMa = Rt;
              dg_REta_to_YZ(Rt, Eta, &cornerYZ[k * 2 + l][0],
                            &cornerYZ[k * 2 + l][1]);
            }
          }
          // Jacobian for this sub-pixel quad
          double pixelJacobian = fabs(
              0.5 * ((cornerYZ[0][0] * cornerYZ[1][1] - cornerYZ[1][0] * cornerYZ[0][1]) +
                     (cornerYZ[1][0] * cornerYZ[3][1] - cornerYZ[3][0] * cornerYZ[1][1]) +
                     (cornerYZ[3][0] * cornerYZ[2][1] - cornerYZ[2][0] * cornerYZ[3][1]) +
                     (cornerYZ[2][0] * cornerYZ[0][1] - cornerYZ[0][0] * cornerYZ[2][1])));
          // Sub-pixel center in R-Eta space
          double ypr_sub = ypr + sp_cy;
          double zpr_sub = zpr + sp_cz;
          dg_pixel_to_REta_corr(ypr_sub, zpr_sub, Ycen, Zcen, TRs, Lsd, RhoD, p0,
                           p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, pxY, dLsd, dP2, parallax,
                           residualCorr, &Rt, &Eta,
                           NULL);
          double YZ_local[2];
          dg_REta_to_YZ(Rt, Eta, &YZ_local[0], &YZ_local[1]);
          // Find overlapping R-bins and Eta-bins
          int nrRChosen = 0;
          int nrEtaChosen = 0;
          for (k = 0; k < nRBins; k++) {
            if (RBinsHigh[k] >= RMi && RBinsLow[k] <= RMa) {
              RChosen[nrRChosen] = k;
              nrRChosen++;
            }
          }
          double etaLo = EtaMi, etaHi = EtaMa;
          int wrapsAround = (etaHi - etaLo > 180);
          if (wrapsAround) {
            double tmp = etaHi;
            etaHi = 360 + etaLo;
            etaLo = tmp;
          }
          for (k = 0; k < nEtaBins; k++) {
            if (EtaBinsHigh[k] >= etaLo && EtaBinsLow[k] <= etaHi) {
              EtaChosen[nrEtaChosen] = k;
              nrEtaChosen++;
              continue;
            }
            if (EtaBinsHigh[k] >= etaLo + 360 && EtaBinsLow[k] <= etaHi + 360) {
              EtaChosen[nrEtaChosen] = k;
              nrEtaChosen++;
              continue;
            }
            if (EtaBinsHigh[k] >= etaLo - 360 && EtaBinsLow[k] <= etaHi - 360) {
              EtaChosen[nrEtaChosen] = k;
              nrEtaChosen++;
              continue;
            }
          }
          // Masked pixel handling (only flag once, on first sub-pixel)
          if (pixelIsMasked) {
            if (si == 0 && sj == 0) {
              for (k = 0; k < nrRChosen; k++) {
                for (l = 0; l < nrEtaChosen; l++) {
                  long long int fPos =
                      (long long int)RChosen[k] * nEtaBins + EtaChosen[l];
#pragma omp atomic write
                  binMaskFlag[fPos] = 1;
                }
              }
            }
            continue;
          }
          sumNrBins += nrRChosen * nrEtaChosen;
          double totPxArea = 0;
          for (k = 0; k < nrRChosen; k++) {
            double RMin = RBinsLow[RChosen[k]];
            double RMax = RBinsHigh[RChosen[k]];
            for (l = 0; l < nrEtaChosen; l++) {
              double EtaMin = EtaBinsLow[EtaChosen[l]];
              double EtaMax = EtaBinsHigh[EtaChosen[l]];
              dg_REta_to_YZ(RMin, EtaMin, &boxEdge[0][0], &boxEdge[0][1]);
              dg_REta_to_YZ(RMin, EtaMax, &boxEdge[1][0], &boxEdge[1][1]);
              dg_REta_to_YZ(RMax, EtaMin, &boxEdge[2][0], &boxEdge[2][1]);
              dg_REta_to_YZ(RMax, EtaMax, &boxEdge[3][0], &boxEdge[3][1]);
              int nEdges = 0;
              for (m = 0; m < 4; m++) {
                double RThis = sqrt(
                    cornerYZ[m][0] * cornerYZ[m][0] +
                    cornerYZ[m][1] * cornerYZ[m][1]);
                double EtaThis =
                    dg_calc_eta_angle(cornerYZ[m][0], cornerYZ[m][1]);
                if (EtaMin < -180 && dg_sign(EtaThis) != dg_sign(EtaMin))
                  EtaThis -= 360;
                if (EtaMax > 180 && dg_sign(EtaThis) != dg_sign(EtaMax))
                  EtaThis += 360;
                if (RThis >= RMin && RThis <= RMax && EtaThis >= EtaMin &&
                    EtaThis <= EtaMax) {
                  Edges[nEdges][0] = cornerYZ[m][0];
                  Edges[nEdges][1] = cornerYZ[m][1];
                  nEdges++;
                }
              }
              for (m = 0; m < 4; m++) {
                if (dg_point_in_quad(boxEdge[m][0], boxEdge[m][1], cornerYZ)) {
                  Edges[nEdges][0] = boxEdge[m][0];
                  Edges[nEdges][1] = boxEdge[m][1];
                  nEdges++;
                }
              }
              if (nEdges < 4) {
                for (int e = 0; e < 4; e++) {
                  int i0 = DG_QUAD_ORDER[e], i1 = DG_QUAD_ORDER[(e + 1) % 4];
                  double py1 = cornerYZ[i0][0], pz1 = cornerYZ[i0][1];
                  double py2 = cornerYZ[i1][0], pz2 = cornerYZ[i1][1];
                  double hits[2][2];
                  int nhits, h;
                  nhits =
                      dg_circle_seg_intersect(py1, pz1, py2, pz2, RMin, hits);
                  for (h = 0; h < nhits; h++) {
                    double EtaH =
                        dg_calc_eta_angle(hits[h][0], hits[h][1]);
                    if (EtaMin < -180 && dg_sign(EtaH) != dg_sign(EtaMin))
                      EtaH -= 360;
                    if (EtaMax > 180 && dg_sign(EtaH) != dg_sign(EtaMax))
                      EtaH += 360;
                    if (EtaH >= EtaMin - DG_EPS && EtaH <= EtaMax + DG_EPS) {
                      Edges[nEdges][0] = hits[h][0];
                      Edges[nEdges][1] = hits[h][1];
                      nEdges++;
                    }
                  }
                  nhits =
                      dg_circle_seg_intersect(py1, pz1, py2, pz2, RMax, hits);
                  for (h = 0; h < nhits; h++) {
                    double EtaH =
                        dg_calc_eta_angle(hits[h][0], hits[h][1]);
                    if (EtaMin < -180 && dg_sign(EtaH) != dg_sign(EtaMin))
                      EtaH -= 360;
                    if (EtaMax > 180 && dg_sign(EtaH) != dg_sign(EtaMax))
                      EtaH += 360;
                    if (EtaH >= EtaMin - DG_EPS && EtaH <= EtaMax + DG_EPS) {
                      Edges[nEdges][0] = hits[h][0];
                      Edges[nEdges][1] = hits[h][1];
                      nEdges++;
                    }
                  }
                  double hy, hz;
                  if (dg_ray_seg_intersect(py1, pz1, py2, pz2, EtaMin, &hy,
                                           &hz)) {
                    double RH = sqrt(hy * hy + hz * hz);
                    if (RH >= RMin - DG_EPS && RH <= RMax + DG_EPS) {
                      Edges[nEdges][0] = hy;
                      Edges[nEdges][1] = hz;
                      nEdges++;
                    }
                  }
                  if (dg_ray_seg_intersect(py1, pz1, py2, pz2, EtaMax, &hy,
                                           &hz)) {
                    double RH = sqrt(hy * hy + hz * hz);
                    if (RH >= RMin - DG_EPS && RH <= RMax + DG_EPS) {
                      Edges[nEdges][0] = hy;
                      Edges[nEdges][1] = hz;
                      nEdges++;
                    }
                  }
                }
              }
              if (nEdges < 3) {
                nrContinued1++;
                continue;
              }
              nEdges = dg_find_unique_vertices(Edges, EdgesOut, nEdges, RMin,
                                               RMax, EtaMin, EtaMax);
              if (nEdges < 3) {
                nrContinued2++;
                continue;
              }
              double Area = dg_polygon_area(EdgesOut, nEdges, RMin, RMax);
              if (Area < 1E-5) {
                nrContinued3++;
                continue;
              }
              // Physical corrections: pyFAI convention.
              double correctedArea = Area;
              if (solidAngleCorr || polarizationCorr) {
                double twoTheta = atan(Rt * pxY / Lsd);
                if (solidAngleCorr) {
                  double c2t = cos(twoTheta);
                  correctedArea /= (c2t * c2t * c2t);
                }
                if (polarizationCorr) {
                  double s2t = sin(twoTheta);
                  double ce = cos(Eta * DG_DEG2RAD);
                  double polFactor = 1.0 - polFraction * s2t * s2t * ce * ce;
                  if (polFactor > 1e-6) correctedArea /= polFactor;
                }
              }
              // Store entry
              {
                int lockIdx =
                    ((long long)RChosen[k] * nEtaBins + EtaChosen[l]) %
                    MAPPER_N_BIN_LOCKS;
                omp_set_lock(&binLocks[lockIdx]);
                int maxnVal = maxnPx[RChosen[k]][EtaChosen[l]];
                int nVal = nPxList[RChosen[k]][EtaChosen[l]];
                if (nVal >= maxnVal) {
                  maxnVal += 2;
                  struct MapPixelData *oldarr = pxList[RChosen[k]][EtaChosen[l]];
                  struct MapPixelData *newarr =
                      realloc(oldarr, maxnVal * sizeof(*newarr));
                  if (newarr == NULL) {
                    fprintf(stderr, "realloc failed in mapper_build_map\n");
                  } else {
                    pxList[RChosen[k]][EtaChosen[l]] = newarr;
                    maxnPx[RChosen[k]][EtaChosen[l]] = maxnVal;
                  }
                }
                int raw_y, raw_z;
                mapper_inverse_transform_pixel(i, j, &raw_y, &raw_z, NrTransOpt,
                                               TransOpt, NrPixelsY, NrPixelsZ);
                pxList[RChosen[k]][EtaChosen[l]][nVal].y =
                    (float)raw_y + (float)sp_cy;
                pxList[RChosen[k]][EtaChosen[l]][nVal].z =
                    (float)raw_z + (float)sp_cz;
                pxList[RChosen[k]][EtaChosen[l]][nVal].frac = correctedArea;
                {
                  double R_bin_center =
                      (RBinsLow[RChosen[k]] + RBinsHigh[RChosen[k]]) * 0.5;
                  pxList[RChosen[k]][EtaChosen[l]][nVal].deltaR =
                      (float)(Rt - R_bin_center);
                }
                pxList[RChosen[k]][EtaChosen[l]][nVal].areaWeight = (float)Area;
                (nPxList[RChosen[k]][EtaChosen[l]])++;
                omp_unset_lock(&binLocks[lockIdx]);
              }
              totPxArea += Area;
              TotNrOfBins++;
            }
          }
        } // end sj
      }   // end si
    }
    dg_free_matrix(Edges, 50);
    dg_free_matrix(EdgesOut, 50);
    free(RChosen);
    free(EtaChosen);
  }
  printf("%lld %lld %lld\n", nrContinued1, nrContinued2, nrContinued3);

  // Debug: print R and Eta for sample pixels to diagnose empty maps
  if (TotNrOfBins == 0) {
    double TRsDbg[3][3];
    dg_build_tilt_matrix(tx, ty, tz, TRsDbg);
    int sampleY[] = {0, NrPixelsY / 2, NrPixelsY - 1, NrPixelsY / 4};
    int sampleZ[] = {0, NrPixelsZ / 2, NrPixelsZ - 1, NrPixelsZ / 4};
    printf("DEBUG: Empty map — R/Eta for sample pixels:\n");
    printf("  Eta bin range: [%.1f, %.1f]\n", EtaBinsLow[0],
           EtaBinsHigh[nEtaBins - 1]);
    printf("  R bin range:   [%.1f, %.1f]\n", RBinsLow[0],
           RBinsHigh[nRBins - 1]);
    for (int si = 0; si < 4; si++) {
      double Rdbg, Etadbg;
      dg_pixel_to_REta_corr((double)sampleY[si], (double)sampleZ[si], Ycen, Zcen,
                       TRsDbg, Lsd, RhoD, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, pxY, 0, 0, parallax,
                       residualCorr, &Rdbg, &Etadbg, NULL);
      printf("  pixel(%4d,%4d): R=%10.2f  Eta=%8.2f\n", sampleY[si],
             sampleZ[si], Rdbg, Etadbg);
    }
  }

  return TotNrOfBins;
}

// --------------------------------------------------------------------------
// mapper_free_map
// --------------------------------------------------------------------------
void mapper_free_map(struct MapPixelData ***pxList, int **nPxList, int **maxnPx,
                     int nRBins, int nEtaBins) {
  for (int i = 0; i < nRBins; i++) {
    for (int j = 0; j < nEtaBins; j++) {
      free(pxList[i][j]);
    }
    free(pxList[i]);
    free(nPxList + i); // nPxList[i] is part of contiguous alloc — only free if separately allocated
  }
  // Note: actual freeing depends on allocation pattern in caller.
  // This function frees the per-bin MapPixelData arrays.
  // The caller is responsible for freeing the top-level arrays if they
  // were allocated contiguously.
  (void)maxnPx; // maxnPx freed by caller
}
