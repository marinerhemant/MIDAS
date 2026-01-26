#ifndef PANEL_H
#define PANEL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int id;
  int yMin;  // Start Pixel Y (Inclusive)
  int yMax;  // End Pixel Y (Inclusive)
  int zMin;  // Start Pixel Z
  int zMax;  // End Pixel Z
  double dY; // Shift Y
  double dZ; // Shift Z
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

#endif
