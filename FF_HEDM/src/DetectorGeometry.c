//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// DetectorGeometry.c — Shared pixel↔(R,η) coordinate transforms and
// area-weighted binning for MIDAS detector mapping.
//
// Extracted from DetectorMapper.c to provide a single source of truth
// for DetectorMapper and CalibrantPanelShiftsOMP.
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
  return DG_RAD2DEG * atan2(-y, z);
}

int dg_between(double val, double min, double max) {
  return ((val - DG_EPS <= max && val + DG_EPS >= min) ? 1 : 0);
}

double dg_sign(double x) {
  if (x > 0)
    return 1.0;
  else if (x < 0)
    return -1.0;
  else
    return 0.0;
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
                      double p1, double p2, double p3, double p4, double p5,
                      double p6,
                      double px, double dLsd, double dP2, double parallax,
                      double *R_out, double *Eta_out,
                      double *Eta_untilted_out) {
  double panelLsd = Lsd + dLsd;
  double panelP2 = p2 + dP2;
  double Yc = (-Y + Ycen) * px;
  double Zc = (Z - Zcen) * px;
  double ABC[3] = {0, Yc, Zc};
  // Untilted Eta: from raw pixel-centered coords, before tilt matrix.
  // Required by CalibrantPanelShiftsOMP's box construction and boundary check.
  double EtaUntilted = dg_calc_eta_angle(ABC[1], ABC[2]);
  double ABCPr[3], XYZ[3];
  dg_mat_mult_33v(TRs, ABC, ABCPr);
  XYZ[0] = panelLsd + ABCPr[0];
  XYZ[1] = ABCPr[1];
  XYZ[2] = ABCPr[2];
  double Rad = (panelLsd / XYZ[0]) * sqrt(XYZ[1] * XYZ[1] + XYZ[2] * XYZ[2]);
  // Tilted Eta: from post-tilt coordinates, used for distortion and binning
  double EtaTilted = dg_calc_eta_angle(XYZ[1], XYZ[2]);
  double RNorm = Rad / RhoD;
  double EtaT = 90 - EtaTilted;
  double n0 = 2.0, n1 = 4.0, n2 = 2.0;
  double DistortFunc =
      (p0 * pow(RNorm, n0) * cos(DG_DEG2RAD * (2 * EtaT + p6))) +
      (p1 * pow(RNorm, n1) * cos(DG_DEG2RAD * (4 * EtaT + p3))) +
      (panelP2 * pow(RNorm, n2));
  DistortFunc += p4 * pow(RNorm, 6.0);
  DistortFunc += p5 * pow(RNorm, 4.0);
  DistortFunc += 1;
  double Rt = Rad * DistortFunc / px; // in pixels
  Rt = Rt * (Lsd / panelLsd);         // re-project to global Lsd plane
  // Parallax/absorption correction: X-rays penetrate the sensor at an
  // angle-dependent depth, shifting the apparent radial position.
  // parallax is in µm; convert to pixels via /px.
  if (parallax != 0.0) {
    double twoTheta = atan(Rad / panelLsd);
    Rt += parallax * sin(twoTheta) / px;
  }
  *R_out = Rt;
  *Eta_out = EtaTilted;
  if (Eta_untilted_out != NULL)
    *Eta_untilted_out = EtaUntilted;
}

void dg_REta_to_YZ(double R, double Eta_deg, double *Y_out, double *Z_out) {
  *Y_out = -R * sin(Eta_deg * DG_DEG2RAD);
  *Z_out = R * cos(Eta_deg * DG_DEG2RAD);
}

// Numerical inversion: given target (R_px, Eta_deg) in corrected space,
// find the raw pixel (Y, Z) such that dg_pixel_to_REta(Y, Z, ...) = (R, Eta).
// Uses Newton-Raphson with the flat-detector formula as initial guess.
void dg_invert_REta_to_pixel(
    double R_target, double Eta_target,
    double Ycen, double Zcen, double TRs[3][3],
    double Lsd, double RhoD,
    double p0, double p1, double p2, double p3, double p4, double p5,
    double p6,
    double px, double dLsd, double dP2, double parallax,
    double *Y_out, double *Z_out) {

  // Initial guess: flat-detector polar formula
  double Y = Ycen + R_target * sin(Eta_target * DG_DEG2RAD);
  double Z = Zcen + R_target * cos(Eta_target * DG_DEG2RAD);

  const int MAX_ITER = 10;
  const double TOL_R = 1e-8;   // pixels
  const double TOL_ETA = 1e-8; // degrees
  const double h = 0.01;       // finite-difference step (pixels)

  for (int iter = 0; iter < MAX_ITER; iter++) {
    // Evaluate forward function at current (Y, Z)
    double R_eval, Eta_eval;
    dg_pixel_to_REta(Y, Z, Ycen, Zcen, TRs, Lsd, RhoD,
                     p0, p1, p2, p3, p4, p5, p6, px, dLsd, dP2, parallax,
                     &R_eval, &Eta_eval, NULL);

    double dR = R_target - R_eval;
    double dEta = Eta_target - Eta_eval;
    // Handle η wraparound near ±180°
    if (dEta > 180.0) dEta -= 360.0;
    if (dEta < -180.0) dEta += 360.0;

    if (fabs(dR) < TOL_R && fabs(dEta) < TOL_ETA)
      break;

    // Numerical Jacobian: ∂(R,η)/∂(Y,Z)
    double R_dY, Eta_dY, R_dZ, Eta_dZ;
    dg_pixel_to_REta(Y + h, Z, Ycen, Zcen, TRs, Lsd, RhoD,
                     p0, p1, p2, p3, p4, p5, p6, px, dLsd, dP2, parallax,
                     &R_dY, &Eta_dY, NULL);
    dg_pixel_to_REta(Y, Z + h, Ycen, Zcen, TRs, Lsd, RhoD,
                     p0, p1, p2, p3, p4, p5, p6, px, dLsd, dP2, parallax,
                     &R_dZ, &Eta_dZ, NULL);

    double dRdY = (R_dY - R_eval) / h;
    double dRdZ = (R_dZ - R_eval) / h;
    double dEdY = (Eta_dY - Eta_eval) / h;
    double dEdZ = (Eta_dZ - Eta_eval) / h;

    // Solve 2×2 system: J · [ΔY, ΔZ]ᵀ = [dR, dEta]ᵀ
    double det = dRdY * dEdZ - dRdZ * dEdY;
    if (fabs(det) < 1e-30) break; // singular Jacobian — bail out

    double deltaY = (dEdZ * dR - dRdZ * dEta) / det;
    double deltaZ = (dRdY * dEta - dEdY * dR) / det;

    Y += deltaY;
    Z += deltaZ;
  }

  *Y_out = Y;
  *Z_out = Z;
}

// ── Panel-aware numerical inversion ─────────────────────────────────
// Given target (R_px, Eta_deg) from the forward model (which included
// per-panel dLsd/dP2 and panel dY/dZ/dTheta corrections), find the
// RAW pixel (Y, Z) such that the full forward pipeline reproduces the
// same (R, Eta).
//
// Strategy:
//   1. Newton-Raphson using dg_pixel_to_REta with panel's dLsd/dP2
//      in the forward step.  The initial guess and iteration variables
//      are in panel-CORRECTED pixel space (i.e., after ApplyPanelCorrection).
//   2. After convergence the corrected pixel is passed through
//      UnApplyPanelCorrection to recover the raw pixel position.
//
// If panel == NULL, falls back to the non-panel version with dLsd=dP2=0.

void dg_invert_REta_to_pixel_panel(
    double R_target, double Eta_target,
    double Ycen, double Zcen, double TRs[3][3],
    double Lsd, double RhoD,
    double p0, double p1, double p2, double p3, double p4, double p5,
    double p6,
    double px, double parallax,
    const Panel *panel,
    double *Y_out, double *Z_out) {

  if (panel == NULL) {
    dg_invert_REta_to_pixel(R_target, Eta_target, Ycen, Zcen, TRs,
                            Lsd, RhoD, p0, p1, p2, p3, p4, p5, p6,
                            px, 0, 0, parallax, Y_out, Z_out);
    return;
  }

  double dLsd = panel->dLsd;
  double dP2  = panel->dP2;

  // Initial guess: flat-detector polar formula (in panel-corrected space)
  double Y = Ycen + R_target * sin(Eta_target * DG_DEG2RAD);
  double Z = Zcen + R_target * cos(Eta_target * DG_DEG2RAD);

  const int MAX_ITER = 10;
  const double TOL_R = 1e-8;
  const double TOL_ETA = 1e-8;
  const double h = 0.01;

  for (int iter = 0; iter < MAX_ITER; iter++) {
    // Forward: panel-corrected pixel → (R, Eta) with panel dLsd/dP2
    double R_eval, Eta_eval;
    dg_pixel_to_REta(Y, Z, Ycen, Zcen, TRs, Lsd, RhoD,
                     p0, p1, p2, p3, p4, p5, p6, px, dLsd, dP2, parallax,
                     &R_eval, &Eta_eval, NULL);

    double dR = R_target - R_eval;
    double dEta = Eta_target - Eta_eval;
    if (dEta >  180.0) dEta -= 360.0;
    if (dEta < -180.0) dEta += 360.0;

    if (fabs(dR) < TOL_R && fabs(dEta) < TOL_ETA)
      break;

    // Numerical Jacobian
    double R_dY, Eta_dY, R_dZ, Eta_dZ;
    dg_pixel_to_REta(Y + h, Z, Ycen, Zcen, TRs, Lsd, RhoD,
                     p0, p1, p2, p3, p4, p5, p6, px, dLsd, dP2, parallax,
                     &R_dY, &Eta_dY, NULL);
    dg_pixel_to_REta(Y, Z + h, Ycen, Zcen, TRs, Lsd, RhoD,
                     p0, p1, p2, p3, p4, p5, p6, px, dLsd, dP2, parallax,
                     &R_dZ, &Eta_dZ, NULL);

    double dRdY = (R_dY - R_eval) / h;
    double dRdZ = (R_dZ - R_eval) / h;
    double dEdY = (Eta_dY - Eta_eval) / h;
    double dEdZ = (Eta_dZ - Eta_eval) / h;

    double det = dRdY * dEdZ - dRdZ * dEdY;
    if (fabs(det) < 1e-30) break;

    double deltaY = (dEdZ * dR - dRdZ * dEta) / det;
    double deltaZ = (dRdY * dEta - dEdY * dR) / det;
    Y += deltaY;
    Z += deltaZ;
  }

  // Y,Z are now panel-corrected pixel coords.  Undo panel dY/dZ/dTheta
  // to get the raw pixel position that the M-step expects.
  double rawY, rawZ;
  UnApplyPanelCorrection(Y, Z, panel, &rawY, &rawZ);
  *Y_out = rawY;
  *Z_out = rawZ;
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

double dg_polygon_area(double **Edges, int nEdges, double RMin, double RMax) {
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

  // Exact area via Green's theorem with mixed boundary:
  //
  //   A = (1/2) ∮ (x dy − y dx)
  //
  // For straight edges: standard Shoelace term = (x₁y₂ − x₂y₁)/2
  // For arc edges (both endpoints on same R-circle):
  //   ∫ along arc of radius R from angle α to β = (R²/2)(β − α)
  //   where angles are measured with atan2(y, x).
  //
  // The polygon winding is counterclockwise (positive area),
  // so arc edges go counterclockwise (increasing angle) on RMax
  // and clockwise (decreasing angle) on RMin.

  double RMin2 = RMin * RMin, RMax2 = RMax * RMax;
  double tol = 1e-6;
  double Area = 0;

  for (i = 0; i < nEdges; i++) {
    int next = (i + 1) % nEdges;
    double r1sq = pts[i].x * pts[i].x + pts[i].y * pts[i].y;
    double r2sq = pts[next].x * pts[next].x + pts[next].y * pts[next].y;

    int onRMin = (fabs(r1sq - RMin2) < tol * RMin2 &&
                  fabs(r2sq - RMin2) < tol * RMin2);
    int onRMax = (fabs(r1sq - RMax2) < tol * RMax2 &&
                  fabs(r2sq - RMax2) < tol * RMax2);

    if (onRMin || onRMax) {
      // Arc edge: use exact integral (R²/2)(β − α)
      double R = onRMin ? RMin : RMax;
      double a1 = atan2(pts[i].y, pts[i].x);
      double a2 = atan2(pts[next].y, pts[next].x);
      double dAngle = a2 - a1;
      // Normalize to (-π, π] to take the short arc
      if (dAngle > M_PI) dAngle -= 2.0 * M_PI;
      if (dAngle < -M_PI) dAngle += 2.0 * M_PI;
      Area += (R * R / 2.0) * dAngle;
    } else {
      // Straight edge: standard Shoelace term
      Area += 0.5 * (pts[i].x * pts[next].y - pts[next].x * pts[i].y);
    }
  }

  free(pts);
  return Area;
}

// ── General-quadrilateral pixel helpers ──────────────────────────────

// Vertex ordering for pixel quad traversal.
// Corner indices: 0=(dy-,dz-), 1=(dy-,dz+), 2=(dy+,dz-), 3=(dy+,dz+).
// Order 0→1→3→2 traces the perimeter.
const int DG_QUAD_ORDER[4] = {0, 1, 3, 2};

int dg_circle_seg_intersect(double y1, double z1, double y2, double z2,
                            double R, double hits[2][2]) {
  double dy = y2 - y1, dz = z2 - z1;
  double a = dy * dy + dz * dz;
  if (a < 1e-30) return 0;  // degenerate segment
  double b = 2.0 * (y1 * dy + z1 * dz);
  double c = y1 * y1 + z1 * z1 - R * R;
  double disc = b * b - 4.0 * a * c;
  if (disc < 0) return 0;
  double sqrtDisc = sqrt(disc);
  double inv2a = 0.5 / a;
  int n = 0;
  double t1 = (-b - sqrtDisc) * inv2a;
  if (t1 >= -DG_EPS && t1 <= 1.0 + DG_EPS) {
    double tc = fmax(0.0, fmin(1.0, t1));
    hits[n][0] = y1 + tc * dy;
    hits[n][1] = z1 + tc * dz;
    n++;
  }
  double t2 = (-b + sqrtDisc) * inv2a;
  if (t2 >= -DG_EPS && t2 <= 1.0 + DG_EPS && fabs(t2 - t1) > 1e-12) {
    double tc = fmax(0.0, fmin(1.0, t2));
    hits[n][0] = y1 + tc * dy;
    hits[n][1] = z1 + tc * dz;
    n++;
  }
  return n;
}

int dg_ray_seg_intersect(double y1, double z1, double y2, double z2,
                         double eta_deg, double *hy, double *hz) {
  // Eta-ray from origin: y*cos(η) + z*sin(η) = 0  (in radians)
  double eta_rad = eta_deg * DG_DEG2RAD;
  double ce = cos(eta_rad), se = sin(eta_rad);
  double dy = y2 - y1, dz = z2 - z1;
  double denom = dy * ce + dz * se;
  if (fabs(denom) < 1e-30) return 0;  // parallel
  double t = -(y1 * ce + z1 * se) / denom;
  if (t < -DG_EPS || t > 1.0 + DG_EPS) return 0;
  t = fmax(0.0, fmin(1.0, t));
  *hy = y1 + t * dy;
  *hz = z1 + t * dz;
  // Check the point is on the POSITIVE ray (correct half-plane)
  // Ray direction: (-sin(η), cos(η)).  Dot with (hy, hz) must be > 0.
  if ((-se) * (*hy) + ce * (*hz) < 0) return 0;
  return 1;
}

int dg_point_in_quad(double py, double pz, double quad[4][2]) {
  // Cross-product sign test for convex polygon.
  // Check that the point is on the same side of all 4 edges.
  int pos = 0, neg = 0;
  for (int e = 0; e < 4; e++) {
    int i0 = DG_QUAD_ORDER[e], i1 = DG_QUAD_ORDER[(e + 1) % 4];
    double ey = quad[i1][0] - quad[i0][0];
    double ez = quad[i1][1] - quad[i0][1];
    double cross = ey * (pz - quad[i0][1]) - ez * (py - quad[i0][0]);
    if (cross > 0) pos++;
    else if (cross < 0) neg++;
  }
  return (pos == 0 || neg == 0) ? 1 : 0;
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

// ── Pixel–bin area (mapperfcn-style) ────────────────────────────────

double dg_calc_pixel_bin_area(double pixY, double pixZ, double RMin,
                              double RMax, double EtaMin, double EtaMax,
                              double **Edges, double **EdgesOut) {
  // Pixel bounding box (unit pixel centered at pixY, pixZ)
  double yMin = pixY - 0.5;
  double yMax = pixY + 0.5;
  double zMin = pixZ - 0.5;
  double zMax = pixZ + 0.5;

  int nEdges = 0;
  int m;

  // (1) Check which pixel corners lie inside the (R,Eta) bin
  for (m = 0; m < 4; m++) {
    double cy = pixY + dg_PosMatrix[m][0];
    double cz = pixZ + dg_PosMatrix[m][1];
    double RThis = sqrt(cy * cy + cz * cz);
    double EtaThis = dg_calc_eta_angle(cy, cz);
    if (EtaMin < -180 && dg_sign(EtaThis) != dg_sign(EtaMin))
      EtaThis -= 360;
    if (EtaMax > 180 && dg_sign(EtaThis) != dg_sign(EtaMax))
      EtaThis += 360;
    if (RThis >= RMin && RThis <= RMax && EtaThis >= EtaMin &&
        EtaThis <= EtaMax) {
      Edges[nEdges][0] = cy;
      Edges[nEdges][1] = cz;
      nEdges++;
    }
  }

  // (2) Check which bin corners lie inside the pixel
  double boxEdge[4][2];
  dg_REta_to_YZ(RMin, EtaMin, &boxEdge[0][0], &boxEdge[0][1]);
  dg_REta_to_YZ(RMin, EtaMax, &boxEdge[1][0], &boxEdge[1][1]);
  dg_REta_to_YZ(RMax, EtaMin, &boxEdge[2][0], &boxEdge[2][1]);
  dg_REta_to_YZ(RMax, EtaMax, &boxEdge[3][0], &boxEdge[3][1]);
  for (m = 0; m < 4; m++) {
    if (boxEdge[m][0] >= yMin && boxEdge[m][0] <= yMax &&
        boxEdge[m][1] >= zMin && boxEdge[m][1] <= zMax) {
      Edges[nEdges][0] = boxEdge[m][0];
      Edges[nEdges][1] = boxEdge[m][1];
      nEdges++;
    }
  }

  if (nEdges < 4) {
    // (3) R-arc × pixel-edge intercepts — check BOTH ±sqrt roots
    double disc, zTemp, yTemp;

    // RMin arc vs y = yMin, y = yMax
    if (RMin >= fabs(yMin)) {
      disc = sqrt(RMin * RMin - yMin * yMin);
      if (dg_between(disc, zMin, zMax) == 1) {
        Edges[nEdges][0] = yMin; Edges[nEdges][1] = disc; nEdges++;
      }
      if (dg_between(-disc, zMin, zMax) == 1) {
        Edges[nEdges][0] = yMin; Edges[nEdges][1] = -disc; nEdges++;
      }
    }
    if (RMin >= fabs(yMax)) {
      disc = sqrt(RMin * RMin - yMax * yMax);
      if (dg_between(disc, zMin, zMax) == 1) {
        Edges[nEdges][0] = yMax; Edges[nEdges][1] = disc; nEdges++;
      }
      if (dg_between(-disc, zMin, zMax) == 1) {
        Edges[nEdges][0] = yMax; Edges[nEdges][1] = -disc; nEdges++;
      }
    }
    if (RMax >= fabs(yMin)) {
      disc = sqrt(RMax * RMax - yMin * yMin);
      if (dg_between(disc, zMin, zMax) == 1) {
        Edges[nEdges][0] = yMin; Edges[nEdges][1] = disc; nEdges++;
      }
      if (dg_between(-disc, zMin, zMax) == 1) {
        Edges[nEdges][0] = yMin; Edges[nEdges][1] = -disc; nEdges++;
      }
    }
    if (RMax >= fabs(yMax)) {
      disc = sqrt(RMax * RMax - yMax * yMax);
      if (dg_between(disc, zMin, zMax) == 1) {
        Edges[nEdges][0] = yMax; Edges[nEdges][1] = disc; nEdges++;
      }
      if (dg_between(-disc, zMin, zMax) == 1) {
        Edges[nEdges][0] = yMax; Edges[nEdges][1] = -disc; nEdges++;
      }
    }

    // RMin/RMax arc vs z = zMin, z = zMax
    if (RMin >= fabs(zMin)) {
      disc = sqrt(RMin * RMin - zMin * zMin);
      if (dg_between(disc, yMin, yMax) == 1) {
        Edges[nEdges][0] = disc; Edges[nEdges][1] = zMin; nEdges++;
      }
      if (dg_between(-disc, yMin, yMax) == 1) {
        Edges[nEdges][0] = -disc; Edges[nEdges][1] = zMin; nEdges++;
      }
    }
    if (RMin >= fabs(zMax)) {
      disc = sqrt(RMin * RMin - zMax * zMax);
      if (dg_between(disc, yMin, yMax) == 1) {
        Edges[nEdges][0] = disc; Edges[nEdges][1] = zMax; nEdges++;
      }
      if (dg_between(-disc, yMin, yMax) == 1) {
        Edges[nEdges][0] = -disc; Edges[nEdges][1] = zMax; nEdges++;
      }
    }
    if (RMax >= fabs(zMin)) {
      disc = sqrt(RMax * RMax - zMin * zMin);
      if (dg_between(disc, yMin, yMax) == 1) {
        Edges[nEdges][0] = disc; Edges[nEdges][1] = zMin; nEdges++;
      }
      if (dg_between(-disc, yMin, yMax) == 1) {
        Edges[nEdges][0] = -disc; Edges[nEdges][1] = zMin; nEdges++;
      }
    }
    if (RMax >= fabs(zMax)) {
      disc = sqrt(RMax * RMax - zMax * zMax);
      if (dg_between(disc, yMin, yMax) == 1) {
        Edges[nEdges][0] = disc; Edges[nEdges][1] = zMax; nEdges++;
      }
      if (dg_between(-disc, yMin, yMax) == 1) {
        Edges[nEdges][0] = -disc; Edges[nEdges][1] = zMax; nEdges++;
      }
    }

    // (4) Eta-ray × pixel-edge intercepts
    double zTempMin, zTempMax, yTempMin, yTempMax;

    // EtaMin ray vs y = yMin, yMax
    // At eta ≈ 0°/180° the ray is along the z-axis (parallel to y-edges), skip
    if (!(fabs(EtaMin) < 1E-5 || fabs(fabs(EtaMin) - 180) < 1E-5)) {
      zTempMin = -yMin / tan(EtaMin * DG_DEG2RAD);
      zTempMax = -yMax / tan(EtaMin * DG_DEG2RAD);
      if (dg_between(zTempMin, zMin, zMax) == 1) {
        Edges[nEdges][0] = yMin;
        Edges[nEdges][1] = zTempMin;
        nEdges++;
      }
      if (dg_between(zTempMax, zMin, zMax) == 1) {
        Edges[nEdges][0] = yMax;
        Edges[nEdges][1] = zTempMax;
        nEdges++;
      }
    }

    // EtaMax ray vs y = yMin, yMax
    if (!(fabs(EtaMax) < 1E-5 || fabs(fabs(EtaMax) - 180) < 1E-5)) {
      zTempMin = -yMin / tan(EtaMax * DG_DEG2RAD);
      zTempMax = -yMax / tan(EtaMax * DG_DEG2RAD);
      if (dg_between(zTempMin, zMin, zMax) == 1) {
        Edges[nEdges][0] = yMin;
        Edges[nEdges][1] = zTempMin;
        nEdges++;
      }
      if (dg_between(zTempMax, zMin, zMax) == 1) {
        Edges[nEdges][0] = yMax;
        Edges[nEdges][1] = zTempMax;
        nEdges++;
      }
    }

    // EtaMin ray vs z = zMin, zMax
    // At eta ≈ ±90° the ray is along the y-axis (parallel to z-edges), skip
    if (!(fabs(fabs(EtaMin) - 90) < 1E-5)) {
      yTempMin = -zMin * tan(EtaMin * DG_DEG2RAD);
      yTempMax = -zMax * tan(EtaMin * DG_DEG2RAD);
      if (dg_between(yTempMin, yMin, yMax) == 1) {
        Edges[nEdges][0] = yTempMin;
        Edges[nEdges][1] = zMin;
        nEdges++;
      }
      if (dg_between(yTempMax, yMin, yMax) == 1) {
        Edges[nEdges][0] = yTempMax;
        Edges[nEdges][1] = zMax;
        nEdges++;
      }
    }

    // EtaMax ray vs z = zMin, zMax
    if (!(fabs(fabs(EtaMax) - 90) < 1E-5)) {
      yTempMin = -zMin * tan(EtaMax * DG_DEG2RAD);
      yTempMax = -zMax * tan(EtaMax * DG_DEG2RAD);
      if (dg_between(yTempMin, yMin, yMax) == 1) {
        Edges[nEdges][0] = yTempMin;
        Edges[nEdges][1] = zMin;
        nEdges++;
      }
      if (dg_between(yTempMax, yMin, yMax) == 1) {
        Edges[nEdges][0] = yTempMax;
        Edges[nEdges][1] = zMax;
        nEdges++;
      }
    }
  }

  if (nEdges < 3)
    return 0.0;

  nEdges = dg_find_unique_vertices(Edges, EdgesOut, nEdges, RMin, RMax, EtaMin,
                                   EtaMax);
  if (nEdges < 3)
    return 0.0;

  return dg_polygon_area(EdgesOut, nEdges, RMin, RMax);
}

// ── Quad-based pixel-bin area ──────────────────────────────────────

double dg_calc_pixel_bin_area_quad(double cornerYZ[4][2], double RMin,
                                   double RMax, double EtaMin, double EtaMax,
                                   double **Edges, double **EdgesOut) {
  int nEdges = 0;
  int m;

  // (1) Check which pixel quad corners lie inside the (R,Eta) bin
  for (m = 0; m < 4; m++) {
    double RThis = sqrt(cornerYZ[m][0] * cornerYZ[m][0] +
                        cornerYZ[m][1] * cornerYZ[m][1]);
    double EtaThis = dg_calc_eta_angle(cornerYZ[m][0], cornerYZ[m][1]);
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

  // (2) Check which bin corners lie inside the pixel quad
  double boxEdge[4][2];
  dg_REta_to_YZ(RMin, EtaMin, &boxEdge[0][0], &boxEdge[0][1]);
  dg_REta_to_YZ(RMin, EtaMax, &boxEdge[1][0], &boxEdge[1][1]);
  dg_REta_to_YZ(RMax, EtaMin, &boxEdge[2][0], &boxEdge[2][1]);
  dg_REta_to_YZ(RMax, EtaMax, &boxEdge[3][0], &boxEdge[3][1]);
  for (m = 0; m < 4; m++) {
    if (dg_point_in_quad(boxEdge[m][0], boxEdge[m][1], cornerYZ)) {
      Edges[nEdges][0] = boxEdge[m][0];
      Edges[nEdges][1] = boxEdge[m][1];
      nEdges++;
    }
  }

  // (3) Intersections of R-arcs and eta-rays with pixel quad edges
  if (nEdges < 4) {
    for (int e = 0; e < 4; e++) {
      int i0 = DG_QUAD_ORDER[e], i1 = DG_QUAD_ORDER[(e + 1) % 4];
      double py1 = cornerYZ[i0][0], pz1 = cornerYZ[i0][1];
      double py2 = cornerYZ[i1][0], pz2 = cornerYZ[i1][1];
      double hits[2][2];
      int nhits, h;

      nhits = dg_circle_seg_intersect(py1, pz1, py2, pz2, RMin, hits);
      for (h = 0; h < nhits; h++) {
        double EtaH = dg_calc_eta_angle(hits[h][0], hits[h][1]);
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

      nhits = dg_circle_seg_intersect(py1, pz1, py2, pz2, RMax, hits);
      for (h = 0; h < nhits; h++) {
        double EtaH = dg_calc_eta_angle(hits[h][0], hits[h][1]);
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
      if (dg_ray_seg_intersect(py1, pz1, py2, pz2, EtaMin, &hy, &hz)) {
        double RH = sqrt(hy * hy + hz * hz);
        if (RH >= RMin - DG_EPS && RH <= RMax + DG_EPS) {
          Edges[nEdges][0] = hy;
          Edges[nEdges][1] = hz;
          nEdges++;
        }
      }
      if (dg_ray_seg_intersect(py1, pz1, py2, pz2, EtaMax, &hy, &hz)) {
        double RH = sqrt(hy * hy + hz * hz);
        if (RH >= RMin - DG_EPS && RH <= RMax + DG_EPS) {
          Edges[nEdges][0] = hy;
          Edges[nEdges][1] = hz;
          nEdges++;
        }
      }
    }
  }

  if (nEdges < 3)
    return 0.0;

  nEdges = dg_find_unique_vertices(Edges, EdgesOut, nEdges, RMin, RMax, EtaMin,
                                   EtaMax);
  if (nEdges < 3)
    return 0.0;

  return dg_polygon_area(EdgesOut, nEdges, RMin, RMax);
}
