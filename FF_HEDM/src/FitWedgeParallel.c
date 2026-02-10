//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//

//
//  FitWedgeParallel.c
//
//  By Hemant Sharma
//  Dated: 02-10-2026
//  Parallelized and adapted for SpotMatrix.csv input
//

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <nlopt.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823

// --- Helper Functions from FitWedge.c ---

int compareDoubles(const void *a, const void *b) {
  double da = *(const double *)a;
  double db = *(const double *)b;
  return (da > db) - (da < db);
}

typedef struct {
  int pairIndex;
  double wedge;
  double minf;
} Result;

int compareResults(const void *a, const void *b) {
  double ma = ((const Result *)a)->minf;
  double mb = ((const Result *)b)->minf;
  return (ma > mb) - (ma < mb);
}

static inline void MatrixMultF33(double m[3][3], double n[3][3],
                                 double res[3][3]) {
  int r;
  for (r = 0; r < 3; r++) {
    res[r][0] = m[r][0] * n[0][0] + m[r][1] * n[1][0] + m[r][2] * n[2][0];
    res[r][1] = m[r][0] * n[0][1] + m[r][1] * n[1][1] + m[r][2] * n[2][1];
    res[r][2] = m[r][0] * n[0][2] + m[r][1] * n[1][2] + m[r][2] * n[2][2];
  }
}

static inline void MatrixMult(double m[3][3], double v[3], double r[3]) {
  int i;
  for (i = 0; i < 3; i++) {
    r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
}

static inline double sind(double x) { return sin(deg2rad * x); }
static inline double cosd(double x) { return cos(deg2rad * x); }
static inline double tand(double x) { return tan(deg2rad * x); }
static inline double asind(double x) { return rad2deg * (asin(x)); }
static inline double acosd(double x) { return rad2deg * (acos(x)); }
static inline double atand(double x) { return rad2deg * (atan(x)); }

static inline double CalcEtaAngle(double y, double z) {
  double alpha = rad2deg * acos(z / sqrt(y * y + z * z));
  if (y > 0)
    alpha = -alpha;
  return alpha;
}

struct my_func_data {
  double Lsd;
  double Ys;
  double Zs;
  double Ome1;
  double Ome2;
  double Wavelength;
};

void YsZsCalc(double Lsd, double Ycen, double Zcen, double p0, double p1,
              double p2, double MaxRad, double tx, double ty, double tz,
              double px, double Ys, double Zs, double *YOut, double *ZOut) {
  double txr = deg2rad * tx;
  double tyr = deg2rad * ty;
  double tzr = deg2rad * tz;
  double Rx[3][3] = {
      {1, 0, 0}, {0, cos(txr), -sin(txr)}, {0, sin(txr), cos(txr)}};
  double Ry[3][3] = {
      {cos(tyr), 0, sin(tyr)}, {0, 1, 0}, {-sin(tyr), 0, cos(tyr)}};
  double Rz[3][3] = {
      {cos(tzr), -sin(tzr), 0}, {sin(tzr), cos(tzr), 0}, {0, 0, 1}};
  double TRint[3][3], TRs[3][3];
  MatrixMultF33(Ry, Rz, TRint);
  MatrixMultF33(Rx, TRint, TRs);
  double n0 = 2, n1 = 4, n2 = 2, Yc, Zc;
  double Eta, RNorm, Rad, EtaT, DistortFunc, Rcorr;
  Yc = -(Ys - Ycen) * px;
  Zc = (Zs - Zcen) * px;
  double ABC[3] = {0, Yc, Zc};
  double ABCPr[3];
  MatrixMult(TRs, ABC, ABCPr);
  double XYZ[3] = {Lsd + ABCPr[0], ABCPr[1], ABCPr[2]};
  Rad = (Lsd / (XYZ[0])) * (sqrt(XYZ[1] * XYZ[1] + XYZ[2] * XYZ[2]));
  Eta = CalcEtaAngle(XYZ[1], XYZ[2]);
  RNorm = Rad / MaxRad;
  EtaT = 90 - Eta;
  DistortFunc = (p0 * (pow(RNorm, n0)) * (cos(deg2rad * (2 * EtaT)))) +
                (p1 * (pow(RNorm, n1)) * (cos(deg2rad * (4 * EtaT)))) +
                (p2 * (pow(RNorm, n2))) + 1;
  Rcorr = Rad * DistortFunc;
  *YOut = -Rcorr * sin(deg2rad * Eta);
  *ZOut = Rcorr * cos(deg2rad * Eta);
}

static double problem_function(unsigned n, const double *x, double *grad,
                               void *f_data_trial) {
  double Error;
  struct my_func_data *f_data = (struct my_func_data *)f_data_trial;
  double Lsd = f_data->Lsd;
  double Omega1 = f_data->Ome1;
  double Omega2 = f_data->Ome2;
  double ysi = f_data->Ys;
  double zsi = f_data->Zs;
  double wl = f_data->Wavelength;
  double wedge = x[0];
  double CosOme = cosd(Omega1);
  double SinOme = sind(Omega1);
  double eta = rad2deg * (atan2(-ysi, zsi));
  double Ring_radius = sqrt((ysi * ysi) + (zsi * zsi));
  double tth = atand(Ring_radius / Lsd);
  double theta = tth / 2;
  double SinTheta = sind(theta);
  double CosTheta = cosd(theta);
  double ds = 2 * SinTheta / wl;
  double CosW = cosd(wedge);
  double SinW = sind(wedge);
  double SinEta = sind(eta);
  double CosEta = cosd(eta);
  double k1 = -ds * SinTheta;
  double k2 = -ds * CosTheta * SinEta;
  double k3 = ds * CosTheta * CosEta;
  if (eta == 90) {
    k3 = 0;
    k2 = -CosTheta;
  } else if (eta == -90) {
    k3 = 0;
    k2 = CosTheta;
  }
  double k1f = (k1 * CosW) + (k3 * SinW);
  double k3f = (k3 * CosW) - (k1 * SinW);
  double k2f = k2;
  double G1a = k1f * CosOme + k2f * SinOme;
  double G2a = k2f * CosOme - k1f * SinOme;
  double G3a = k3f;
  double normGa = sqrt((G1a * G1a) + (G2a * G2a) + (G3a * G3a));
  double g1 = G1a * ds / normGa;
  double g2 = G2a * ds / normGa;
  double g3 = G3a * ds / normGa;
  g1 = -g1;
  g2 = -g2;
  g3 = -g3;
  double Length_G = sqrt((g1 * g1) + (g2 * g2) + (g3 * g3));
  double k1i = -(Length_G * Length_G) * (wl / 2);
  double A = (k1i + (g3 * SinW)) / (CosW);
  double a_Sin = (g1 * g1) + (g2 * g2);
  double b_Sin = 2 * A * g2;
  double c_Sin = (A * A) - (g1 * g1);
  double a_Cos = a_Sin;
  double b_Cos = -2 * A * g1;
  double c_Cos = (A * A) - (g2 * g2);
  double Par_Sin = (b_Sin * b_Sin) - (4 * a_Sin * c_Sin);
  double Par_Cos = (b_Cos * b_Cos) - (4 * a_Cos * c_Cos);
  double P_check_Sin = 0;
  double P_check_Cos = 0;
  double P_Sin, P_Cos;
  if (Par_Sin >= 0) {
    P_Sin = sqrt(Par_Sin);
  } else {
    P_Sin = 0;
    P_check_Sin = 1;
  }
  if (Par_Cos >= 0) {
    P_Cos = sqrt(Par_Cos);
  } else {
    P_Cos = 0;
    P_check_Cos = 1;
  }

  double Sin_Omega1 = ((-b_Sin) - (P_Sin)) / (2 * a_Sin);
  double Sin_Omega2 = ((-b_Sin) + (P_Sin)) / (2 * a_Sin);
  double Cos_Omega1 = ((-b_Cos) - (P_Cos)) / (2 * a_Cos);
  double Cos_Omega2 = ((-b_Cos) + (P_Cos)) / (2 * a_Cos);

  if (Sin_Omega1 < -1)
    Sin_Omega1 = 0;
  else if (Sin_Omega1 > 1)
    Sin_Omega1 = 0;
  else if (Sin_Omega2 > 1)
    Sin_Omega2 = 0;
  else if (Sin_Omega2 < -1)
    Sin_Omega2 = 0;
  if (Cos_Omega1 < -1)
    Cos_Omega1 = 0;
  else if (Cos_Omega1 > 1)
    Cos_Omega1 = 0;
  else if (Cos_Omega2 > 1)
    Cos_Omega2 = 0;
  else if (Cos_Omega2 < -1)
    Cos_Omega2 = 0;
  if (P_check_Sin == 1) {
    Sin_Omega1 = 0;
    Sin_Omega2 = 0;
  }
  if (P_check_Cos == 1) {
    Cos_Omega1 = 0;
    Cos_Omega2 = 0;
  }
  double Option_1 =
      fabs((Sin_Omega1 * Sin_Omega1) + (Cos_Omega1 * Cos_Omega1) - 1);
  double Option_2 =
      fabs((Sin_Omega1 * Sin_Omega1) + (Cos_Omega2 * Cos_Omega2) - 1);
  double Omega_1, Omega_2;
  if (Option_1 < Option_2) {
    Omega_1 = rad2deg * (atan2(Sin_Omega1, Cos_Omega1));
    Omega_2 = rad2deg * (atan2(Sin_Omega2, Cos_Omega2));
  } else {
    Omega_1 = rad2deg * (atan2(Sin_Omega1, Cos_Omega2));
    Omega_2 = rad2deg * (atan2(Sin_Omega2, Cos_Omega1));
  }
  double Omega_diff1 = fabs(Omega_1 - Omega2);
  double Omega_diff2 = fabs(Omega_2 - Omega2);
  double OmegaMin = 10000;
  if (Omega_diff1 < OmegaMin) {
    OmegaMin = Omega_diff1;
  }
  if (Omega_diff2 < OmegaMin) {
    OmegaMin = Omega_diff2;
  }
  Error = OmegaMin * OmegaMin;
  return Error;
}

// Thread-safe FitWedge function
// Returns 0 if success, 1 if hit bounds
int FitWedgeThreadSafe(double Lsd, double Ycen, double Zcen, double p0,
                       double p1, double p2, double MaxRad, double tx,
                       double ty, double tz, double px, double Ys, double Zs,
                       double MinOme, double MaxOme, double WedgeIn,
                       double *WedgeFit, double *MinFOut, double Wavelength) {
  struct my_func_data f_data;
  f_data.Lsd = Lsd;
  f_data.Ome1 = MinOme;
  f_data.Ome2 = MaxOme;
  f_data.Wavelength = Wavelength;
  double Y, Z;
  YsZsCalc(Lsd, Ycen, Zcen, p0, p1, p2, MaxRad, tx, ty, tz, px, Ys, Zs, &Y, &Z);
  f_data.Ys = Y;
  f_data.Zs = Z;
  unsigned n = 1;
  double xl[n], xu[n], x[n];
  xl[0] = WedgeIn - 5.0; // Increasing range slightly for robustness
  xu[0] = WedgeIn + 5.0;
  x[0] = WedgeIn;
  struct my_func_data *f_datat;
  f_datat = &f_data;
  void *trp = (struct my_func_data *)f_datat;

  // nlopt_create is thread safe if used locally
  nlopt_opt opt;
  opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
  nlopt_set_lower_bounds(opt, xl);
  nlopt_set_upper_bounds(opt, xu);
  nlopt_set_min_objective(opt, problem_function, trp);
  double minf;
  nlopt_optimize(opt, x, &minf);
  nlopt_destroy(opt);
  *WedgeFit = x[0];
  *MinFOut = minf;

  // Check bounds
  if (fabs(x[0] - xl[0]) < 1e-4 || fabs(x[0] - xu[0]) < 1e-4) {
    return 1;
  }
  return 0;
}

// --- Data Structures ---

typedef struct {
  int GrainID;
  int SpotID;
  double Omega;
  double DetectorHor;  // Y
  double DetectorVert; // Z
  double OmeRaw;
  double Eta;
  int RingNr;
  // Other fields ignored for now
} Spot;

typedef struct {
  int idx1;
  int idx2;
  double OmeDiff; // Precompute for debugging if needed
  double EtaSum;
} Pair;

// --- Main ---

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <ParamFile> <nCPUs>\n", argv[0]);
    return 1;
  }

  char *ParamFN = argv[1];
  int nCPUs = atoi(argv[2]);
  omp_set_num_threads(nCPUs);

  // --- Read Parameters ---
  FILE *fileParam = fopen(ParamFN, "r");
  if (!fileParam) {
    printf("Error opening param file: %s\n", ParamFN);
    return 1;
  }

  char aline[1024];
  char dummy[1024];
  char *str;
  int LowNr;
  double Ycen, Zcen, px, Lsd, ty, tz, p0, p1, p2, MaxRad, tx;
  double Wavelength, Wedge = 0.0;
  double OmegaStart = 0.0;
  double OmegaStep = 0.0;

  // Initialize defaults
  Ycen = Zcen = 1024.0;
  px = 0.1;
  Lsd = 1000.0;
  p0 = p1 = p2 = 0.0;

  while (fgets(aline, 1024, fileParam) != NULL) {
    if (strncmp(aline, "BC ", 3) == 0)
      sscanf(aline, "%s %lf %lf", dummy, &Ycen, &Zcen);
    else if (strncmp(aline, "px ", 3) == 0)
      sscanf(aline, "%s %lf", dummy, &px);
    else if (strncmp(aline, "Lsd ", 4) == 0)
      sscanf(aline, "%s %lf", dummy, &Lsd);
    else if (strncmp(aline, "tx ", 3) == 0)
      sscanf(aline, "%s %lf", dummy, &tx);
    else if (strncmp(aline, "ty ", 3) == 0)
      sscanf(aline, "%s %lf", dummy, &ty);
    else if (strncmp(aline, "tz ", 3) == 0)
      sscanf(aline, "%s %lf", dummy, &tz);
    else if (strncmp(aline, "Wedge ", 6) == 0)
      sscanf(aline, "%s %lf", dummy, &Wedge);
    else if (strncmp(aline, "p0 ", 3) == 0)
      sscanf(aline, "%s %lf", dummy, &p0);
    else if (strncmp(aline, "p1 ", 3) == 0)
      sscanf(aline, "%s %lf", dummy, &p1);
    else if (strncmp(aline, "p2 ", 3) == 0)
      sscanf(aline, "%s %lf", dummy, &p2);
    else if (strncmp(aline, "RhoD ", 5) == 0)
      sscanf(aline, "%s %lf", dummy, &MaxRad);
    else if (strncmp(aline, "Wavelength ", 11) == 0)
      sscanf(aline, "%s %lf", dummy, &Wavelength);
    else if (strncmp(aline, "OmegaStart ", 11) == 0)
      sscanf(aline, "%s %lf", dummy, &OmegaStart);
    else if (strncmp(aline, "OmegaStep ", 10) == 0)
      sscanf(aline, "%s %lf", dummy, &OmegaStep);
  }
  fclose(fileParam);

  printf("Parameters read. Lsd: %f, Wavelength: %f, Wedge: %f\n", Lsd,
         Wavelength, Wedge);

  // --- Read SpotMatrix.csv ---
  char *SpotMatrixFN = "SpotMatrix.csv";
  FILE *fSpots = fopen(SpotMatrixFN, "r");
  if (!fSpots) {
    printf("Error opening %s. Ensure it is in the current directory.\n",
           SpotMatrixFN);
    return 1;
  }

  // First pass: count lines
  int nLines = 0;
  while (fgets(aline, 1024, fSpots) != NULL) {
    if (aline[0] != '%')
      nLines++;
  }
  rewind(fSpots);

  Spot *spots = (Spot *)malloc(nLines * sizeof(Spot));
  int nSpots = 0;

  // Skip header if present (lines starting with %)
  while (fgets(aline, 1024, fSpots) != NULL) {
    if (aline[0] == '%')
      continue;

    // Parse line
    // Format: %multimap key not used by sscanf, so we use dummy for first few
    // tokens if needed
    // Example line:
    // 74778 17224 43.781480 553.083540 656.679760 43.781480 -119.414750 1
    // Columns: GrainID SpotID Omega DetectorHor DetectorVert OmeRaw Eta RingNr

    Spot s;
    int r = sscanf(aline, "%d %d %lf %lf %lf %lf %lf %d", &s.GrainID, &s.SpotID,
                   &s.Omega, &s.DetectorHor, &s.DetectorVert, &s.OmeRaw, &s.Eta,
                   &s.RingNr);

    if (r >= 8) {
      spots[nSpots++] = s;
    }
  }
  fclose(fSpots);
  printf("Read %d spots.\n", nSpots);

  // --- Find Pairs Efficiently ---
  // Data is assumed PRE-SORTED by GrainID, then RingNr.
  // We can process chunk by chunk.

  size_t pairsCap = 10000; // Initial capacity
  Pair *pairs = (Pair *)malloc(pairsCap * sizeof(Pair));
  int nPairs = 0;

  int i = 0;
  while (i < nSpots) {
    int j = i + 1;
    // Find end of current chunk (same GrainID and RingNr)
    while (j < nSpots && spots[j].GrainID == spots[i].GrainID &&
           spots[j].RingNr == spots[i].RingNr) {
      j++;
    }

    // Process chunk [i, j)
    // Iterate all pairs within this chunk
    for (int p1 = i; p1 < j; p1++) {
      for (int p2 = p1 + 1; p2 < j; p2++) {

        double OmeDiff = fabs(spots[p1].Omega - spots[p2].Omega);
        if (fabs(OmeDiff - 180.0) > 1.0)
          continue;

        double EtaSum = spots[p1].Eta + spots[p2].Eta;
        // Check if sum is close to 180 or -180.
        if (fabs(fabs(EtaSum) - 180.0) > 1.0)
          continue;

        // Check Omega Filtering
        if (OmegaStep > 1e-6) { // Only check if OmegaStep is provided/non-zero
          double k1 = (spots[p1].Omega - OmegaStart) / OmegaStep;
          double k2 = (spots[p2].Omega - OmegaStart) / OmegaStep;

          // If close to integer, exclude
          if (fabs(k1 - round(k1)) < 1e-4 || fabs(k2 - round(k2)) < 1e-4) {
            continue;
          }
        }

        // Found a valid pair
        if (nPairs >= pairsCap) {
          pairsCap *= 2;
          pairs = (Pair *)realloc(pairs, pairsCap * sizeof(Pair));
          if (!pairs) {
            fprintf(stderr, "Memory allocation failed for pairs array.\n");
            free(spots);
            return 1;
          }
        }

        pairs[nPairs].idx1 = p1;
        pairs[nPairs].idx2 = p2;
        pairs[nPairs].OmeDiff = OmeDiff;
        pairs[nPairs].EtaSum = EtaSum;
        nPairs++;
      }
    }

    // Move to next chunk
    i = j;
  }

  printf("Found %d pairs.\n", nPairs);

  // --- Parallel Fitting ---
  Result *results = (Result *)malloc(nPairs * sizeof(Result));
  int *validResults =
      (int *)calloc(nPairs, sizeof(int)); // 0: invalid, 1: valid

  if ((!results || !validResults) && nPairs > 0) {
    fprintf(stderr, "Memory allocation failed for results array.\n");
    if (results)
      free(results);
    if (validResults)
      free(validResults);
    free(pairs);
    free(spots);
    return 1;
  }

  double startTime = omp_get_wtime();

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < nPairs; i++) {
    int idx1 = pairs[i].idx1;
    int idx2 = pairs[i].idx2;

    double Ome1 = spots[idx1].Omega;
    double Ome2 = spots[idx2].Omega;
    double Y1 = spots[idx1].DetectorHor;
    double Z1 = spots[idx1].DetectorVert;
    double Y2 = spots[idx2].DetectorHor;
    double Z2 = spots[idx2].DetectorVert;

    double MinOme, MaxOme;
    double YsUse, ZsUse;

    if (Ome1 < Ome2) {
      MinOme = Ome1;
      MaxOme = Ome2;
      YsUse = Y1;
      ZsUse = Z1;
    } else {
      MinOme = Ome2;
      MaxOme = Ome1;
      YsUse = Y2;
      ZsUse = Z2;
    }

    double resWedge = 0;
    double resMinF = 0;
    int status = FitWedgeThreadSafe(Lsd, Ycen, Zcen, p0, p1, p2, MaxRad, tx, ty,
                                    tz, px, YsUse, ZsUse, MinOme, MaxOme, Wedge,
                                    &resWedge, &resMinF, Wavelength);

    if (status == 0) {
      validResults[i] = 1;
      results[i].pairIndex = i;
      results[i].wedge = resWedge;
      results[i].minf = resMinF;
    } else {
      validResults[i] = 0;
    }
  }

  double endTime = omp_get_wtime();

  // --- Initial Filtering (Status 0) ---
  int nValidInit = 0;
  for (int i = 0; i < nPairs; i++) {
    if (validResults[i])
      nValidInit++;
  }

  Result *tempResults = (Result *)malloc(nValidInit * sizeof(Result));
  int k = 0;
  // Use a temp array to store minf values for stats
  double *minfVals = (double *)malloc(nValidInit * sizeof(double));
  if (!tempResults || !minfVals) {
    // Handle allocation fail if needed, though unlikely given nPairs allocated
  }

  for (int i = 0; i < nPairs; i++) {
    if (validResults[i]) {
      tempResults[k] = results[i];
      minfVals[k] = results[i].minf;
      k++;
    }
  }

  // --- Calculate Minf Stats for Filtering ---
  double minfThreshold = 1e9; // Default high
  if (nValidInit > 0) {
    // Sort minfVals for Median
    qsort(minfVals, nValidInit, sizeof(double), compareDoubles);

    double medianMinf;
    if (nValidInit % 2 == 0) {
      medianMinf =
          (minfVals[nValidInit / 2 - 1] + minfVals[nValidInit / 2]) / 2.0;
    } else {
      medianMinf = minfVals[nValidInit / 2];
    }

    // Calculate Mean and StdDev of minf
    double sumMinf = 0.0;
    for (int i = 0; i < nValidInit; i++)
      sumMinf += minfVals[i];
    double meanMinf = sumMinf / nValidInit;

    double sumSqDiffMinf = 0.0;
    for (int i = 0; i < nValidInit; i++) {
      sumSqDiffMinf += (minfVals[i] - meanMinf) * (minfVals[i] - meanMinf);
    }
    double stdDevMinf = sqrt(sumSqDiffMinf / nValidInit);

    minfThreshold = medianMinf + 2.0 * stdDevMinf;

    printf("Minf Filter Stats: Median=%e, StdDev=%e, Threshold=%e\n",
           medianMinf, stdDevMinf, minfThreshold);
  }
  free(minfVals);

  // --- Final Filtering based on Minf Threshold ---
  int nValid = 0;
  for (int i = 0; i < nValidInit; i++) {
    if (tempResults[i].minf <= minfThreshold) {
      nValid++;
    }
  }

  Result *finalResults = (Result *)malloc(nValid * sizeof(Result));
  k = 0;
  for (int i = 0; i < nValidInit; i++) {
    if (tempResults[i].minf <= minfThreshold) {
      finalResults[k++] = tempResults[i];
    }
  }

  free(tempResults);
  free(results);
  free(validResults); // pairs and spots needed for indices

  // Sort results based on minf
  qsort(finalResults, nValid, sizeof(Result), compareResults);

  // --- Output Results ---
  FILE *fOut = fopen("WedgeResults.txt", "w");
  if (!fOut) {
    printf("Error opening output file.\n");
    free(finalResults);
    free(pairs);
    free(spots);
    return 1;
  }

  fprintf(fOut, "GrainID SpotID1 SpotID2 RingNr Omega1 Omega2 Y1 Z1 Eta1 Y2 Z2 "
                "Eta2 Wedge minf\n");
  for (int i = 0; i < nValid; i++) {
    int pairIdx = finalResults[i].pairIndex;
    int idx1 = pairs[pairIdx].idx1;
    int idx2 = pairs[pairIdx].idx2;
    fprintf(fOut, "%d %d %d %d %f %f %f %f %f %f %f %f %f %e\n",
            spots[idx1].GrainID, spots[idx1].SpotID, spots[idx2].SpotID,
            spots[idx1].RingNr, spots[idx1].Omega, spots[idx2].Omega,
            spots[idx1].DetectorHor, spots[idx1].DetectorVert, spots[idx1].Eta,
            spots[idx2].DetectorHor, spots[idx2].DetectorVert, spots[idx2].Eta,
            finalResults[i].wedge, finalResults[i].minf);
  }
  fclose(fOut);
  printf("Results written to WedgeResults.txt\n");

  // --- Statistics ---
  if (nValid > 0) {
    // Calculate Mean, StdDev, Range for Wedge
    double sum = 0.0;
    double minW = finalResults[0].wedge;
    double maxW = finalResults[0].wedge;

    // We need a temp array for median calculation since 'results' is sorted by
    // minf
    double *wedgeVals = (double *)malloc(nValid * sizeof(double));

    for (int i = 0; i < nValid; i++) {
      double w = finalResults[i].wedge;
      wedgeVals[i] = w;
      sum += w;
      if (w < minW)
        minW = w;
      if (w > maxW)
        maxW = w;
    }
    double mean = sum / nValid;

    double sumSqDiff = 0.0;
    for (int i = 0; i < nValid; i++) {
      sumSqDiff += (wedgeVals[i] - mean) * (wedgeVals[i] - mean);
    }
    double stdDev = sqrt(sumSqDiff / nValid);

    // Calculate Median (sort temp array)
    qsort(wedgeVals, nValid, sizeof(double), compareDoubles);
    double median;
    if (nValid % 2 == 0) {
      median = (wedgeVals[nValid / 2 - 1] + wedgeVals[nValid / 2]) / 2.0;
    } else {
      median = wedgeVals[nValid / 2];
    }

    printf("\n--- Wedge Statistics ---\n");
    printf("Count:      %d\n", nValid);
    printf("Mean:       %f\n", mean);
    printf("Median:     %f\n", median);
    printf("Std Dev:    %f\n", stdDev);
    printf("Range:      [%f, %f]\n", minW, maxW);
    printf("------------------------\n");
    printf("Total Time: %f seconds (Wall Clock)\n", endTime - startTime);
    printf("------------------------\n");

    free(wedgeVals);
  } else {
    printf("\nNo pairs found for statistics (after filtering).\n");
    printf("Total Time: %f seconds (Wall Clock)\n", endTime - startTime);
    printf("------------------------\n");
  }

  free(spots);
  free(pairs);
  free(finalResults);

  return 0;
}
