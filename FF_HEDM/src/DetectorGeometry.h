//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// DetectorGeometry.h — Shared pixel↔(R,η) coordinate transforms and
// area-weighted binning for MIDAS detector mapping.
//
// Used by: DetectorMapper, DetectorMapperZarr, CalibrantPanelShiftsOMP
//

#ifndef DETECTOR_GEOMETRY_H
#define DETECTOR_GEOMETRY_H

#include <math.h>

#define DG_DEG2RAD 0.0174532925199433
#define DG_RAD2DEG 57.2957795130823
#define DG_EPS 1E-6

// ── Coordinate transforms ───────────────────────────────────────────

// Build tilt rotation matrix TRs = Rx(tx) · Ry(ty) · Rz(tz)
// tx, ty, tz in degrees.
void dg_build_tilt_matrix(double tx_deg, double ty_deg, double tz_deg,
                          double TRs[3][3]);

// Canonical forward transform: pixel (Y,Z) → tilt-corrected (R_px, Eta_deg).
// Includes full distortion model (p0–p5) and per-panel Lsd/p2 corrections.
// dLsd = per-panel ΔLsd (0 for no correction), dP2 = per-panel Δp2.
// parallax = parallax correction in µm (0 for no correction).
void dg_pixel_to_REta(double Y, double Z, double Ycen, double Zcen,
                      double TRs[3][3], double Lsd, double RhoD, double p0,
                      double p1, double p2, double p3, double p4, double p5,
                      double px, double dLsd, double dP2, double parallax,
                      double *R_out, double *Eta_out,
                      double *Eta_untilted_out);

// Inverse: (R_px, Eta_deg) → centered (Y, Z) in pixel units.
// Coordinates are centered at beam center (0,0).
void dg_REta_to_YZ(double R, double Eta_deg, double *Y_out, double *Z_out);

// Numerical inversion: given target (R_px, Eta_deg) in corrected space,
// find the raw pixel (Y, Z) such that dg_pixel_to_REta(Y, Z, ...) ≈ (R, Eta).
// Uses Newton-Raphson with the flat-detector formula as initial guess.
void dg_invert_REta_to_pixel(
    double R_target, double Eta_target,
    double Ycen, double Zcen, double TRs[3][3],
    double Lsd, double RhoD,
    double p0, double p1, double p2, double p3, double p4, double p5,
    double px, double dLsd, double dP2, double parallax,
    double *Y_out, double *Z_out);

// ── Bin construction ────────────────────────────────────────────────

// Build uniform R and Eta bin edges.
void dg_build_bin_edges(double RMin, double EtaMin, int nRBins, int nEtaBins,
                        double RBinSize, double EtaBinSize, double *RBinsLow,
                        double *RBinsHigh, double *EtaBinsLow,
                        double *EtaBinsHigh);

// ── Polygon area / vertex utilities ─────────────────────────────────

// Allocate a nrows×ncols matrix of doubles.
double **dg_alloc_matrix(int nrows, int ncols);

// Free a matrix allocated by dg_alloc_matrix.
void dg_free_matrix(double **mat, int nrows);

// Compute area of a convex polygon given as nEdges (Y,Z) vertices
// (Shoelace formula with angular sort), then apply circular-segment
// corrections for adjacent vertex pairs on the same R-circle.
// Thread-safe: uses local sorting state.
double dg_polygon_area(double **Edges, int nEdges, double RMin, double RMax);

// Deduplicate polygon vertices and clip to (R, η) bin boundaries.
// Returns the number of unique, in-bounds vertices written to EdgesOut.
int dg_find_unique_vertices(double **EdgesIn, double **EdgesOut, int nEdgesIn,
                            double RMin, double RMax, double EtaMin,
                            double EtaMax);

// ── General-quadrilateral pixel helpers ─────────────────────────────

// Pixel quad vertex ordering for traversal (counterclockwise).
// Indices into cornerYZ[4][2]: edges are QUAD_ORDER[e]→QUAD_ORDER[(e+1)%4].
extern const int DG_QUAD_ORDER[4];

// Find intersections of circle y²+z²=R² with line segment P1→P2.
// Returns 0, 1, or 2.  Valid intersection points stored in hits[][2].
int dg_circle_seg_intersect(double y1, double z1, double y2, double z2,
                            double R, double hits[2][2]);

// Find intersection of eta-ray (from origin at angle eta_deg) with segment P1→P2.
// Returns 1 if found (result in *hy, *hz), 0 otherwise.
int dg_ray_seg_intersect(double y1, double z1, double y2, double z2,
                         double eta_deg, double *hy, double *hz);

// Check if point (py,pz) is inside the convex quadrilateral defined by
// cornerYZ[DG_QUAD_ORDER[0..3]].  Returns 1 if inside, 0 otherwise.
int dg_point_in_quad(double py, double pz, double quad[4][2]);

// Compute the intersection area of a unit pixel centered at (pixY, pixZ)
// with the (R, Eta) bin [RMin..RMax] × [EtaMin..EtaMax].
// Uses true R-arc and Eta-ray geometry (mapperfcn-style).
// Edges[50][2] and EdgesOut[50][2] are caller-allocated scratch arrays.
double dg_calc_pixel_bin_area(double pixY, double pixZ, double RMin,
                              double RMax, double EtaMin, double EtaMax,
                              double **Edges, double **EdgesOut);

// Compute the intersection area of a pixel quadrilateral (cornerYZ[4][2])
// with the (R, Eta) bin [RMin..RMax] × [EtaMin..EtaMax].
// Uses actual pixel corners in remapped (Y,Z) space instead of unit square.
double dg_calc_pixel_bin_area_quad(double cornerYZ[4][2], double RMin,
                                   double RMax, double EtaMin, double EtaMax,
                                   double **Edges, double **EdgesOut);

// ── Small helpers ───────────────────────────────────────────────────

double dg_calc_eta_angle(double y, double z);
int dg_between(double val, double min, double max);
double dg_sign(double x);

// ── Constants ───────────────────────────────────────────────────────

extern const double dg_dy[2];           // {-0.5, +0.5}
extern const double dg_dz[2];           // {-0.5, +0.5}
extern const double dg_PosMatrix[4][2]; // pixel corner offsets

#endif // DETECTOR_GEOMETRY_H
