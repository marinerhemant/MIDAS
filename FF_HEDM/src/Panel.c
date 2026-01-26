#include "Panel.h"

int GeneratePanels(int nPanelsY, int nPanelsZ, int panelSizeY, int panelSizeZ,
                   int *gapsY, int *gapsZ, Panel **panels, int *nPanels) {
  int totalPanels = nPanelsY * nPanelsZ;
  *nPanels = totalPanels;
  *panels = (Panel *)malloc(totalPanels * sizeof(Panel));
  if (*panels == NULL) {
    fprintf(stderr, "Error: Memory allocation failed for panels.\n");
    return 1;
  }

  int currentY = 0;
  int currentZ = 0;
  int idx = 0;

  // Y Loop
  for (int i = 0; i < nPanelsY; i++) {
    int yStart = currentY;
    int yEnd = yStart + panelSizeY - 1; // Inclusive

    // Calculate next Y start
    if (i < nPanelsY - 1) {
      currentY = yEnd + 1 + gapsY[i];
    }

    // Z Loop
    currentZ = 0; // Reset Z for each row
    for (int j = 0; j < nPanelsZ; j++) {
      int zStart = currentZ;
      int zEnd = zStart + panelSizeZ - 1; // Inclusive

      // Calculate next Z start
      if (j < nPanelsZ - 1) {
        currentZ = zEnd + 1 + gapsZ[j];
      }

      (*panels)[idx].id = idx;
      (*panels)[idx].yMin = yStart;
      (*panels)[idx].yMax = yEnd;
      (*panels)[idx].zMin = zStart;
      (*panels)[idx].zMax = zEnd;
      (*panels)[idx].dY = 0.0;
      (*panels)[idx].dZ = 0.0;

      idx++;
    }
  }

  return 0;
}

int LoadPanelShifts(const char *filename, int nPanels, Panel *panels) {
  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    fprintf(stderr, "Error: Could not open panel shifts file %s\n", filename);
    return 1;
  }

  char line[1024];
  while (fgets(line, sizeof(line), fp)) {
    if (line[0] == '#' || line[0] == '\n')
      continue;
    int id;
    double dY, dZ;
    if (sscanf(line, "%d %lf %lf", &id, &dY, &dZ) == 3) {
      if (id >= 0 && id < nPanels) {
        panels[id].dY = dY;
        panels[id].dZ = dZ;
      }
    }
  }
  fclose(fp);
  return 0;
}

int SavePanelShifts(const char *filename, int nPanels, Panel *panels) {
  FILE *fp = fopen(filename, "w");
  if (fp == NULL) {
    fprintf(stderr, "Error: Could not open file for writing panel shifts %s\n",
            filename);
    return 1;
  }

  fprintf(fp, "# ID dY dZ\n");
  for (int i = 0; i < nPanels; i++) {
    fprintf(fp, "%d %.10f %.10f\n", panels[i].id, panels[i].dY, panels[i].dZ);
  }
  fclose(fp);
  return 0;
}

int GetPanelIndex(double y, double z, int nPanels, Panel *panels) {
  // Simple linear search. Can be optimized if needed, but nPanels is small.
  // Assuming integer coordinates for bounds check, but input is double.
  // Using simple round or floor might be risky near edges, strictly checking
  // against bounds. Usually pixels are 0.5 centered or integer indices.
  // Assuming standard 0-based indexing where a pixel (y,z) belongs to [Min,
  // Max].

  for (int i = 0; i < nPanels; i++) {
    if (y >= panels[i].yMin && y <= panels[i].yMax && z >= panels[i].zMin &&
        z <= panels[i].zMax) {
      return i;
    }
  }
  return -1;
}
