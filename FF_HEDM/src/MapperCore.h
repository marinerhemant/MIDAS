//
// MapperCore.h — Shared Green's theorem pixel→(R,Eta) bin mapping
//
// Extracted from DetectorMapper.c to share across DetectorMapper,
// IntegratorZarrOMP, and CalibrantIntegratorOMP.
//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

#ifndef MAPPER_CORE_H
#define MAPPER_CORE_H

#include <omp.h>
#include "Panel.h"
#include "DetectorGeometry.h"

// Per-pixel bin entry. Binary layout must match Map.bin format.
struct MapPixelData {
  float y;
  float z;
  double frac;       /* corrected weight: Area / C  (C = solid-angle × polarization) */
  float deltaR;      /* R_sub_centroid - R_bin_center (gradient correction) */
  float areaWeight;  /* uncorrected geometric area weight */
};

// For backward compatibility with code that uses "struct data"
typedef struct MapPixelData MapPixelData;

// Number of OMP lock stripes for bin writes
#define MAPPER_N_BIN_LOCKS 4096

// Invert ImTransOpt for raw pixel coordinate storage in Map.bin.
// Applies the inverse of each transformation in reverse order.
void mapper_inverse_transform_pixel(int y_in, int z_in, int *y_out, int *z_out,
                                    int NrTransOpt, const int TransOpt[10],
                                    int NrPixelsY, int NrPixelsZ);

// Green's theorem sub-pixel area-weighted pixel→(R,Eta) bin mapping.
// Returns total number of bin entries created.
//
// Global-equivalent parameters (were globals in DetectorMapper.c):
//   distortionMapY, distortionMapZ — per-pixel distortion offsets [NrPixelsZ * NrPixelsY]
//   mapPanels, mapNPanels         — panel array and count
//
// Output:
//   pxList[rBin][etaBin]  — dynamic array of MapPixelData entries
//   nPxList[rBin][etaBin] — count of entries per bin
//   maxnPx[rBin][etaBin]  — allocated capacity per bin
//   binMaskFlag[rBin * nEtaBins + etaBin] — 1 if any masked pixel touched this bin
//
long long mapper_build_map(
    /* geometry */
    double tx, double ty, double tz,
    int NrPixelsY, int NrPixelsZ,
    double pxY, double pxZ,
    double Ycen, double Zcen, double Lsd, double RhoD,
    double p0, double p1, double p2, double p3, double p4, double p5, double p6,
    double p7, double p8, double p9, double p10,
    /* bins */
    double *EtaBinsLow, double *EtaBinsHigh,
    double *RBinsLow, double *RBinsHigh,
    int nRBins, int nEtaBins,
    /* output */
    struct MapPixelData ***pxList, int **nPxList, int **maxnPx,
    /* mask */
    double *mask,
    int *binMaskFlag,
    /* transforms */
    int NrTransOpt, const int TransOpt[10],
    /* threading */
    omp_lock_t *binLocks,
    /* options */
    int SubPixelLevel, double SubPixelCardinalWidth,
    double parallax,
    int solidAngleCorr, int polarizationCorr, double polFraction,
    /* distortion maps (were globals) */
    const double *distortionMapY, const double *distortionMapZ,
    /* panels (were globals) */
    const Panel *mapPanels, int mapNPanels,
    /* residual correction map (NULL = disabled) */
    const DGResidualCorr *residualCorr
);

// Free all mapping data allocated by mapper_build_map.
void mapper_free_map(struct MapPixelData ***pxList, int **nPxList, int **maxnPx,
                     int nRBins, int nEtaBins);

#endif /* MAPPER_CORE_H */
