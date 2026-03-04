//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// DetectorGeometry.c — Shared pixel↔(R,η) coordinate transforms and
// area-weighted binning for MIDAS detector mapping.
//
// Extracted from DetectorMapper.c to provide a single source of truth
// for DetectorMapper, DetectorMapperZarr, and CalibrantPanelShiftsOMP.
//

#include "DetectorGeometry.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ── Constants ───────────────────────────────────────────────────────

const double dg_dy[2] = {-0.5, +0.5};
const double dg_dz[2] = {-0.5, +0.5};
const double dg_PosMatrix[4][2] = {
    {-0.5, -0.5}, {-0.5, 0.5}, {0.5, 0.5}, {0.5, -0.5}};

// ── Small helpers ───────────────────────────────────────────────────

double dg_calc_eta_angle(double y, double z) {
  double alpha = DG_RAD2DEG * acos(z / sqrt(y * y + z * z));
  if (y > 0)
    alpha = -alpha;
  return alpha;
}

int dg_between(double val, double min, double max) {
  return ((val - DG_EPS <= max && val + DG_EPS >= min) ? 1 : 0);
}

double dg_sign(double x) {
  if (x == 0)
    return 1.0;
  else
    return x / fabs(x);
}

// ── Matrix utilities ────────────────────────────────────────────────

static inline void dg_mat_mult_33v(double m[3][3], double v[3], double r[3]) {
  int i;
  for (i = 0; i < 3; i++) {
    r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
}

static inline void dg_mat_mult_33x33(double m[3][3], double n[3][3],
                                     double res[3][3]) {
  int r;
  for (r = 0; r < 3; r++) {
    res[r][0] = m[r][0] * n[0][0] + m[r][1] * n[1][0] + m[r][2] * n[2][0];
    res[r][1] = m[r][0] * n[0][1] + m[r][1] * n[1][1] + m[r][2] * n[2][1];
    res[r][2] = m[r][0] * n[0][2] + m[r][1] * n[1][2] + m[r][2] * n[2][2];
  }
}

// ── Coordinate transforms ───────────────────────────────────────────

void dg_build_tilt_matrix(double tx_deg, double ty_deg, double tz_deg,
                          double TRs[3][3]) {
  double txr = DG_DEG2RAD * tx_deg;
  double tyr = DG_DEG2RAD * ty_deg;
  double tzr = DG_DEG2RAD * tz_deg;
  double Rx[3][3] = {
      {1, 0, 0}, {0, cos(txr), -sin(txr)}, {0, sin(txr), cos(txr)}};
  double Ry[3][3] = {
      {cos(tyr), 0, sin(tyr)}, {0, 1, 0}, {-sin(tyr), 0, cos(tyr)}};
  double Rz[3][3] = {
      {cos(tzr), -sin(tzr), 0}, {sin(tzr), cos(tzr), 0}, {0, 0, 1}};
  double TRint[3][3];
  dg_mat_mult_33x33(Ry, Rz, TRint);
  dg_mat_mult_33x33(Rx, TRint, TRs);
}

void dg_pixel_to_REta(double Y, double Z, double Ycen, double Zcen,
                      double TRs[3][3], double Lsd, double RhoD, double p0,
                      double p1, double p2, double p3, double p4, double px,
                      double dLsd, double dP2, double *R_out, double *Eta_out) {
  double panelLsd = Lsd + dLsd;
  double panelP2 = p2 + dP2;
  double Yc = (-Y + Ycen) * px;
  double Zc = (Z - Zcen) * px;
  double ABC[3] = {0, Yc, Zc};
  double ABCPr[3], XYZ[3];
  dg_mat_mult_33v(TRs, ABC, ABCPr);
  XYZ[0] = panelLsd + ABCPr[0];
  XYZ[1] = ABCPr[1];
  XYZ[2] = ABCPr[2];
  double Rad = (panelLsd / XYZ[0]) * sqrt(XYZ[1] * XYZ[1] + XYZ[2] * XYZ[2]);
  double Eta = dg_calc_eta_angle(XYZ[1], XYZ[2]);
  double RNorm = Rad / RhoD;
  double EtaT = 90 - Eta;
  double n0 = 2.0, n1 = 4.0, n2 = 2.0;
  double DistortFunc =
      (p0 * pow(RNorm, n0) * cos(DG_DEG2RAD * (2 * EtaT))) +
      (p1 * pow(RNorm, n1) * cos(DG_DEG2RAD * (4 * EtaT + p3))) +
      (panelP2 * pow(RNorm, n2));
  DistortFunc += p4 * pow(RNorm, 6.0);
  DistortFunc += 1;
  double Rt = Rad * DistortFunc / px; // in pixels
  Rt = Rt * (Lsd / panelLsd);         // re-project to global Lsd plane
  *R_out = Rt;
  *Eta_out = Eta;
}

void dg_REta_to_YZ(double R, double Eta_deg, double *Y_out, double *Z_out) {
  *Y_out = -R * sin(Eta_deg * DG_DEG2RAD);
  *Z_out = R * cos(Eta_deg * DG_DEG2RAD);
}

// ── Bin construction ────────────────────────────────────────────────

void dg_build_bin_edges(double RMin, double EtaMin, int nRBins, int nEtaBins,
                        double RBinSize, double EtaBinSize, double *RBinsLow,
                        double *RBinsHigh, double *EtaBinsLow,
                        double *EtaBinsHigh) {
  int i;
  for (i = 0; i < nEtaBins; i++) {
    EtaBinsLow[i] = EtaBinSize * i + EtaMin;
    EtaBinsHigh[i] = EtaBinSize * (i + 1) + EtaMin;
  }
  for (i = 0; i < nRBins; i++) {
    RBinsLow[i] = RBinSize * i + RMin;
    RBinsHigh[i] = RBinSize * (i + 1) + RMin;
  }
}

// ── Matrix allocation ───────────────────────────────────────────────

double **dg_alloc_matrix(int nrows, int ncols) {
  double **arr;
  int i;
  arr = malloc(nrows * sizeof(*arr));
  if (arr == NULL)
    return NULL;
  for (i = 0; i < nrows; i++) {
    arr[i] = malloc(ncols * sizeof(*arr[i]));
    if (arr[i] == NULL)
      return NULL;
  }
  return arr;
}

void dg_free_matrix(double **mat, int nrows) {
  int r;
  for (r = 0; r < nrows; r++) {
    free(mat[r]);
  }
  free(mat);
}

// ── Polygon area (thread-safe) ──────────────────────────────────────

// Local comparator context for thread-safe angular sorting.
typedef struct {
  double cx, cy; // centroid
} dg_sort_ctx;

typedef struct {
  double x, y;
} dg_point;

// Thread-safe comparison via qsort_r / __compar_d_fn_t
#if defined(__APPLE__)
// macOS qsort_r: thunk is first arg to comparator
static int dg_cmp_angle(void *ctx, const void *ia, const void *ib) {
#else
// Linux/glibc qsort_r: thunk is last arg to comparator
static int dg_cmp_angle(const void *ia, const void *ib, void *ctx) {
#endif
  dg_sort_ctx *c = (dg_sort_ctx *)ctx;
  const dg_point *a = (const dg_point *)ia;
  const dg_point *b = (const dg_point *)ib;
  double ax = a->x - c->cx, ay = a->y - c->cy;
  double bx = b->x - c->cx, by = b->y - c->cy;

  if (ax >= 0 && bx < 0)
    return 1;
  if (ax < 0 && bx >= 0)
    return -1;
  if (ax == 0 && bx == 0) {
    if (ay >= 0 || by >= 0)
      return a->y > b->y ? 1 : -1;
    return b->y > a->y ? 1 : -1;
  }
  double det = ax * by - bx * ay;
  if (det < 0)
    return 1;
  if (det > 0)
    return -1;
  double d1 = ax * ax + ay * ay;
  double d2 = bx * bx + by * by;
  return d1 > d2 ? 1 : -1;
}

double dg_polygon_area(double **Edges, int nEdges) {
  int i;
  dg_point *pts = malloc(nEdges * sizeof(*pts));
  dg_sort_ctx ctx = {0, 0};

  for (i = 0; i < nEdges; i++) {
    ctx.cx += Edges[i][0];
    ctx.cy += Edges[i][1];
    pts[i].x = Edges[i][0];
    pts[i].y = Edges[i][1];
  }
  ctx.cx /= nEdges;
  ctx.cy /= nEdges;

  // Thread-safe sort (no global state)
#if defined(__APPLE__)
  qsort_r(pts, nEdges, sizeof(dg_point), &ctx, dg_cmp_angle);
#else
  qsort_r(pts, nEdges, sizeof(dg_point), dg_cmp_angle, &ctx);
#endif

  // Shoelace formula
  double Area = 0;
  for (i = 0; i < nEdges; i++) {
    int next = (i + 1) % nEdges;
    Area += 0.5 * (pts[i].x * pts[next].y - pts[next].x * pts[i].y);
  }

  free(pts);
  return Area;
}

// ── Vertex deduplication & clipping ─────────────────────────────────

int dg_find_unique_vertices(double **EdgesIn, double **EdgesOut, int nEdgesIn,
                            double RMin, double RMax, double EtaMin,
                            double EtaMax) {
  int i, j, nEdgesOut = 0, duplicate;
  double Len, RT, ET;
  for (i = 0; i < nEdgesIn; i++) {
    duplicate = 0;
    for (j = i + 1; j < nEdgesIn; j++) {
      Len = sqrt(
          (EdgesIn[i][0] - EdgesIn[j][0]) * (EdgesIn[i][0] - EdgesIn[j][0]) +
          (EdgesIn[i][1] - EdgesIn[j][1]) * (EdgesIn[i][1] - EdgesIn[j][1]));
      if (Len == 0) {
        duplicate = 1;
      }
    }
    RT = sqrt(EdgesIn[i][0] * EdgesIn[i][0] + EdgesIn[i][1] * EdgesIn[i][1]);
    ET = dg_calc_eta_angle(EdgesIn[i][0], EdgesIn[i][1]);
    if (!dg_between(ET, EtaMin, EtaMax)) {
      if (dg_between(ET + 360, EtaMin, EtaMax)) {
        ET += 360;
      } else if (dg_between(ET - 360, EtaMin, EtaMax)) {
        ET -= 360;
      }
    }
    if (dg_between(RT, RMin, RMax) == 0) {
      duplicate = 1;
    }
    if (dg_between(ET, EtaMin, EtaMax) == 0) {
      duplicate = 1;
    }
    if (duplicate == 0) {
      EdgesOut[nEdgesOut][0] = EdgesIn[i][0];
      EdgesOut[nEdgesOut][1] = EdgesIn[i][1];
      nEdgesOut++;
    }
  }
  return nEdgesOut;
}
