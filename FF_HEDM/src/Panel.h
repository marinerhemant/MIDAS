#ifndef PANEL_H
#define PANEL_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int id;
  int yMin;       // Start Pixel Y (Inclusive)
  int yMax;       // End Pixel Y (Inclusive)
  int zMin;       // Start Pixel Z
  int zMax;       // End Pixel Z
  double dY;      // Shift Y
  double dZ;      // Shift Z
  double dTheta;  // In-plane rotation (degrees)
  double dLsd;    // Per-panel Lsd offset (microns)
  double dP2;     // Per-panel p2 distortion offset
  double centerY; // Rotation anchor Y = (yMin+yMax)/2.0
  double centerZ; // Rotation anchor Z = (zMin+zMax)/2.0
} Panel;

// Generates panels from parameters.
// Gap arrays must have sufficient elements for the number of gaps (N-1).
// nPanelsY, nPanelsZ: Number of panels in Y and Z directions.
// panelSizeY, panelSizeZ: Size of each panel in pixels.
// gapsY, gapsZ: Arrays containing gap sizes. gapsY size is (nPanelsY - 1), etc.
int GeneratePanels(int nPanelsY, int nPanelsZ, int panelSizeY, int panelSizeZ,
                   int *gapsY, int *gapsZ, Panel **panels, int *nPanels);

// Loads panel shifts from a separate file.
// Format: ID dY dZ
int LoadPanelShifts(const char *filename, int nPanels, Panel *panels);

// Save panel shifts to a file.
int SavePanelShifts(const char *filename, int nPanels, Panel *panels);

// Returns panel index for a given pixel, or -1 if not found.
int GetPanelIndex(double y, double z, int nPanels, Panel *panels);

// Apply panel correction: rotation around center, then translational shift.
// Input:  raw pixel (y, z) and panel index.
// Output: corrected (yOut, zOut).
// Requires <math.h> to be included by the caller.
static inline void ApplyPanelCorrection(double y, double z, const Panel *p,
                                        double *yOut, double *zOut) {
  double dy = y - p->centerY;
  double dz = z - p->centerZ;
  if (p->dTheta != 0.0) {
    double rad = 0.0174532925199433 * p->dTheta;
    double cosT = cos(rad);
    double sinT = sin(rad);
    *yOut = p->centerY + dy * cosT - dz * sinT + p->dY;
    *zOut = p->centerZ + dy * sinT + dz * cosT + p->dZ;
  } else {
    *yOut = y + p->dY;
    *zOut = z + p->dZ;
  }
}

// Inverse: undo panel correction (shift then inverse rotation).
// Used when converting from corrected coords back to raw pixel coords.
static inline void UnApplyPanelCorrection(double y, double z, const Panel *p,
                                          double *yOut, double *zOut) {
  double ys = y - p->dY;
  double zs = z - p->dZ;
  if (p->dTheta != 0.0) {
    double rad = -0.0174532925199433 * p->dTheta; // negate for inverse
    double cosT = cos(rad);
    double sinT = sin(rad);
    double dy = ys - p->centerY;
    double dz = zs - p->centerZ;
    *yOut = p->centerY + dy * cosT - dz * sinT;
    *zOut = p->centerZ + dy * sinT + dz * cosT;
  } else {
    *yOut = ys;
    *zOut = zs;
  }
}

#endif
