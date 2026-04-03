//
// Copyright (c) 2014, UChicago Argonne, LLC
// See LICENSE file.
//
// MIDAS_ParamParser.c — Implementation of centralized parameter file parser
//

#include "MIDAS_ParamParser.h"

void midas_config_defaults(MIDASConfig *cfg) {
  memset(cfg, 0, sizeof(*cfg));
  cfg->Padding = 6;
  cfg->HeadSize = 0;
  cfg->DataType = 1;
  cfg->RBinWidth = 4;
  cfg->tolP3 = 45;          // Historical default
  cfg->tolP6 = 90;          // cos(2η) has period 180°, need ±90°
  cfg->tolShifts = 1.0;
  cfg->MinIndicesForFit = 1;
  cfg->nIterations = 1;
  cfg->iterOffset = 0;
  cfg->OutlierIterations = 1;
  cfg->TrimmedMeanFraction = 1.0;
  cfg->tolLsdPanel = 100;
  cfg->tolP2Panel = 0.0001;
  cfg->tolWavelength = 0.001;
  cfg->lineoutRBinSize = 0.25;
  cfg->lineoutRMin = 10.0;
  sprintf(cfg->darkDatasetName, "exchange/dark");
  sprintf(cfg->dataDatasetName, "exchange/data");
  cfg->MargABC = 0.3;
  cfg->MargABG = 0.3;
  cfg->NrFilesPerDistance = 1;
  cfg->NumPhases = 1;
  cfg->MinFracAccept = 0.5;
  cfg->MinNrSpots = 1;
  cfg->SpaceGroup = 225;
  cfg->WriteSpots = 1;
  cfg->WriteImage = 1;
  cfg->UpdatedOrientations = 1;
  cfg->PeakIntensity = 2000.0;
  cfg->MaxOutputIntensity = 65000.0;
  cfg->num_lambda_samples = 1;
  cfg->SubPixelLevel = 1;
  cfg->SubPixelCardinalWidth = 5.0;
  cfg->Width = 1500;  // ring half-width in µm (Width/px = pixels)
  cfg->ResidualCorrMapFN[0] = '\0';
}

void midas_apply_tol_defaults(MIDASConfig *cfg) {
  if (!cfg->tolP0Set && cfg->tolP0 == 0)
    cfg->tolP0 = cfg->tolP;
  if (!cfg->tolP1Set && cfg->tolP1 == 0)
    cfg->tolP1 = cfg->tolP;
  if (!cfg->tolP2Set && cfg->tolP2 == 0)
    cfg->tolP2 = cfg->tolP;
  if (!cfg->tolP3Set && cfg->tolP3 == 45)
    cfg->tolP3 = cfg->tolP;
  if (!cfg->tolP4Set && cfg->tolP4 == 0)
    cfg->tolP4 = cfg->tolP;
  if (!cfg->tolP5Set && cfg->tolP5 == 0)
    cfg->tolP5 = cfg->tolP;
  if (!cfg->tolP6Set && cfg->tolP6 == 90)
    cfg->tolP6 = cfg->tolP;
  if (!cfg->tolP7Set && cfg->tolP7 == 0)
    cfg->tolP7 = cfg->tolP;
  if (!cfg->tolP8Set && cfg->tolP8 == 180)
    cfg->tolP8 = cfg->tolP;
  if (!cfg->tolP9Set && cfg->tolP9 == 0)
    cfg->tolP9 = cfg->tolP;
  if (!cfg->tolP10Set && cfg->tolP10 == 180)
    cfg->tolP10 = cfg->tolP;
  if (!cfg->tolP11Set && cfg->tolP11 == 0)
    cfg->tolP11 = cfg->tolP;
  if (!cfg->tolP12Set && cfg->tolP12 == 180)
    cfg->tolP12 = cfg->tolP;
  if (!cfg->tolP13Set && cfg->tolP13 == 0)
    cfg->tolP13 = cfg->tolP;
  if (!cfg->tolP14Set && cfg->tolP14 == 180)
    cfg->tolP14 = cfg->tolP;
}

int midas_parse_params(const char *filename, MIDASConfig *cfg) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "Error: cannot open parameter file %s\n", filename);
    return -1;
  }

  midas_config_defaults(cfg);

  char aline[MAX_LINE_LENGTH];
  while (fgets(aline, MAX_LINE_LENGTH, f) != NULL) {
    if (param_skip_line(aline)) continue;

    // ── File I/O ──
    if (param_str(aline, "FileStem", cfg->FileStem, sizeof(cfg->FileStem))) continue;
    if (param_str(aline, "Folder", cfg->Folder, sizeof(cfg->Folder))) continue;
    if (param_str(aline, "Ext", cfg->Ext, sizeof(cfg->Ext))) continue;
    if (param_str(aline, "Dark", cfg->Dark, sizeof(cfg->Dark))) continue;
    if (param_int(aline, "StartNr", &cfg->StartNr)) continue;
    if (param_int(aline, "EndNr", &cfg->EndNr)) continue;
    if (param_int(aline, "Padding", &cfg->Padding)) continue;
    if (param_int(aline, "HeadSize", &cfg->HeadSize)) continue;
    if (param_int(aline, "DataType", &cfg->DataType)) continue;
    if (param_int(aline, "SkipFrame", &cfg->SkipFrame)) continue;

    // ── HDF5/Zarr dataset names ──
    if (param_str(aline, "darkDataset", cfg->darkDatasetName, sizeof(cfg->darkDatasetName))) continue;
    if (param_str(aline, "darkLoc", cfg->darkDatasetName, sizeof(cfg->darkDatasetName))) continue;
    if (param_str(aline, "dataDataset", cfg->dataDatasetName, sizeof(cfg->dataDatasetName))) continue;
    if (param_str(aline, "dataLoc", cfg->dataDatasetName, sizeof(cfg->dataDatasetName))) continue;

    // ── Output / Input paths ──
    if (param_str(aline, "OutputFolder", cfg->OutputFolder, sizeof(cfg->OutputFolder))) continue;
    if (param_str(aline, "ResultFolder", cfg->ResultFolder, sizeof(cfg->ResultFolder))) continue;
    if (param_str(aline, "RefinementFileName", cfg->InputFileName, sizeof(cfg->InputFileName))) continue;
    if (param_str(aline, "InFileName", cfg->InputFileName, sizeof(cfg->InputFileName))) continue;
    if (param_str(aline, "GrainsFile", cfg->GrainsFile, sizeof(cfg->GrainsFile))) continue;
    if (param_str(aline, "SeedOrientations", cfg->SeedOrientations, sizeof(cfg->SeedOrientations))) continue;
    if (param_str(aline, "DataDirectory", cfg->DataDirectory, sizeof(cfg->DataDirectory))) continue;
    if (param_str(aline, "GridFileName", cfg->GridFileName, sizeof(cfg->GridFileName))) continue;

    // ── Detector Geometry ──
    if (key_match(aline, "NrPixels")) {
      sscanf(aline, "%*s %d", &cfg->NrPixelsY);
      cfg->NrPixelsZ = cfg->NrPixelsY;
      continue;
    }
    if (param_int(aline, "NrPixelsY", &cfg->NrPixelsY)) continue;
    if (param_int(aline, "NrPixelsZ", &cfg->NrPixelsZ)) continue;
    if (param_double(aline, "px", &cfg->px)) continue;
    if (param_double(aline, "Lsd", &cfg->Lsd)) continue;
    if (param_double(aline, "Distance", &cfg->Lsd)) continue;  // alias
    if (param_double2(aline, "BC", &cfg->ybc, &cfg->zbc)) continue;
    if (param_double(aline, "tx", &cfg->tx)) continue;
    if (param_double(aline, "ty", &cfg->ty)) continue;
    if (param_double(aline, "tz", &cfg->tz)) continue;
    if (param_double(aline, "p0", &cfg->p0)) continue;
    if (param_double(aline, "p1", &cfg->p1)) continue;
    if (param_double(aline, "p2", &cfg->p2)) continue;
    if (param_double(aline, "p3", &cfg->p3)) continue;
    if (param_double(aline, "p4", &cfg->p4)) continue;
    if (param_double(aline, "p5", &cfg->p5)) continue;
    if (param_double(aline, "p6", &cfg->p6)) continue;
    if (param_double(aline, "p7", &cfg->p7)) continue;
    if (param_double(aline, "p8", &cfg->p8)) continue;
    if (param_double(aline, "p9", &cfg->p9)) continue;
    if (param_double(aline, "p10", &cfg->p10)) continue;
    if (param_double(aline, "p11", &cfg->p11)) continue;
    if (param_double(aline, "p12", &cfg->p12)) continue;
    if (param_double(aline, "p13", &cfg->p13)) continue;
    if (param_double(aline, "p14", &cfg->p14)) continue;
    if (param_double(aline, "Wedge", &cfg->Wedge)) continue;
    if (param_double(aline, "RhoD", &cfg->RhoD)) continue;

    // ── Multi-detector geometry ──
    if (key_match(aline, "DetParams")) {
      if (cfg->nDetParams < 10) {
        sscanf(aline, "%*s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
               &cfg->DetParams[cfg->nDetParams][0], &cfg->DetParams[cfg->nDetParams][1],
               &cfg->DetParams[cfg->nDetParams][2], &cfg->DetParams[cfg->nDetParams][3],
               &cfg->DetParams[cfg->nDetParams][4], &cfg->DetParams[cfg->nDetParams][5],
               &cfg->DetParams[cfg->nDetParams][6], &cfg->DetParams[cfg->nDetParams][7],
               &cfg->DetParams[cfg->nDetParams][8], &cfg->DetParams[cfg->nDetParams][9]);
        cfg->nDetParams++;
      }
      continue;
    }
    if (param_int(aline, "BigDetSize", &cfg->BigDetSize)) continue;

    // ── Crystallography ──
    if (param_int(aline, "SpaceGroup", &cfg->SpaceGroup)) continue;
    if (param_double6(aline, "LatticeConstant", cfg->LatticeConstant)) continue;
    if (param_double6(aline, "LatticeParameter", cfg->LatticeConstant)) continue;
    if (param_double(aline, "Wavelength", &cfg->Wavelength)) continue;

    // ── Image Transforms ──
    if (key_match(aline, "ImTransOpt")) {
      if (cfg->NrTransOpt < 10)
        sscanf(aline, "%*s %d", &cfg->TransOpt[cfg->NrTransOpt++]);
      continue;
    }

    // ── Masking ──
    if (param_str(aline, "ResidualCorrectionMap", cfg->ResidualCorrMapFN, sizeof(cfg->ResidualCorrMapFN))) continue;
    if (key_match(aline, "MaskFile")) {
      sscanf(aline, "%*s %s", cfg->MaskFN);
      cfg->makeMap = 3;
      // Also populate ForwardSim MaskFile + useMask
      strncpy(cfg->MaskFile, cfg->MaskFN, sizeof(cfg->MaskFile) - 1);
      cfg->useMask = 1;
      continue;
    }
    if (key_match(aline, "GapFile")) {
      sscanf(aline, "%*s %s", cfg->GapFN);
      cfg->makeMap = 2;
      continue;
    }
    if (key_match(aline, "BadPxFile")) {
      sscanf(aline, "%*s %s", cfg->BadPxFN);
      cfg->makeMap = 2;
      continue;
    }
    if (key_match(aline, "GapIntensity")) {
      sscanf(aline, "%*s %lld", &cfg->GapIntensity);
      cfg->makeMap = 1;
      continue;
    }
    if (key_match(aline, "BadPxIntensity")) {
      sscanf(aline, "%*s %lld", &cfg->BadPxIntensity);
      cfg->makeMap = 1;
      continue;
    }

    // ── Omega/Box ranges ──
    if (key_match(aline, "OmegaRange")) {
      if (cfg->nOmeRanges < MAX_N_OMEGA_RANGES) {
        sscanf(aline, "%*s %lf %lf",
               &cfg->OmegaRanges[cfg->nOmeRanges][0],
               &cfg->OmegaRanges[cfg->nOmeRanges][1]);
        cfg->nOmeRanges++;
      }
      continue;
    }
    if (key_match(aline, "BoxSize")) {
      if (cfg->nBoxSizes < MAX_N_OMEGA_RANGES) {
        sscanf(aline, "%*s %lf %lf %lf %lf",
               &cfg->BoxSizes[cfg->nBoxSizes][0],
               &cfg->BoxSizes[cfg->nBoxSizes][1],
               &cfg->BoxSizes[cfg->nBoxSizes][2],
               &cfg->BoxSizes[cfg->nBoxSizes][3]);
        cfg->nBoxSizes++;
      }
      continue;
    }
    if (param_double(aline, "OmegaStart", &cfg->OmegaStart)) continue;
    if (param_double(aline, "OmegaEnd", &cfg->OmegaEnd)) continue;
    if (param_double(aline, "OmegaStep", &cfg->OmegaStep)) continue;

    // ── Ring selection ──
    if (key_match(aline, "RingsToExclude")) {
      if (cfg->nRingsExclude < MAX_N_RINGS)
        sscanf(aline, "%*s %d", &cfg->RingsExclude[cfg->nRingsExclude++]);
      continue;
    }
    if (key_match(aline, "RingsToReject")) {  // alias for RingsToExclude
      if (cfg->nRingsExclude < MAX_N_RINGS)
        sscanf(aline, "%*s %d", &cfg->RingsExclude[cfg->nRingsExclude++]);
      continue;
    }
    if (key_match(aline, "RingThresh")) {
      if (cfg->nRingThresh < MAX_N_RINGS) {
        int ringVal;
        sscanf(aline, "%*s %d", &ringVal);
        cfg->RingThresh[cfg->nRingThresh++] = ringVal;
        // Also populate RingsToUse for ForwardSim compatibility
        if (cfg->nRingsToUse < 500)
          cfg->RingsToUse[cfg->nRingsToUse++] = ringVal;
      }
      continue;
    }
    if (param_int(aline, "MaxRingNumber", &cfg->MaxRingNumber)) continue;
    if (param_double(aline, "MaxRingRad", &cfg->RhoD)) continue;
    if (key_match(aline, "RingNumbers")) {
      if (cfg->nRingNumbers < MAX_N_RINGS)
        sscanf(aline, "%*s %d", &cfg->RingNumbers[cfg->nRingNumbers++]);
      continue;
    }
    if (key_match(aline, "RingRadii")) {
      if (cfg->nRingRadii < MAX_N_RINGS)
        sscanf(aline, "%*s %lf", &cfg->RingRadii[cfg->nRingRadii++]);
      continue;
    }

    // ── Optimization Tolerances ──
    if (param_double(aline, "tolTilts", &cfg->tolTilts)) continue;
    if (key_match(aline, "tolTiltX")) {
      sscanf(aline, "%*s %lf", &cfg->tolTiltX);
      cfg->tolTiltXSet = 1;
      continue;
    }
    if (param_double(aline, "tolBC", &cfg->tolBC)) continue;
    if (param_double(aline, "tolLsd", &cfg->tolLsd)) continue;
    if (param_double(aline, "tolP", &cfg->tolP)) continue;
    if (key_match(aline, "tolP0")) {
      sscanf(aline, "%*s %lf", &cfg->tolP0);
      cfg->tolP0Set = 1;
      continue;
    }
    if (key_match(aline, "tolP1")) {
      sscanf(aline, "%*s %lf", &cfg->tolP1);
      cfg->tolP1Set = 1;
      continue;
    }
    if (key_match(aline, "tolP2")) {
      sscanf(aline, "%*s %lf", &cfg->tolP2);
      cfg->tolP2Set = 1;
      continue;
    }
    if (key_match(aline, "tolP3")) {
      sscanf(aline, "%*s %lf", &cfg->tolP3);
      cfg->tolP3Set = 1;
      continue;
    }
    if (key_match(aline, "tolP4")) {
      sscanf(aline, "%*s %lf", &cfg->tolP4);
      cfg->tolP4Set = 1;
      continue;
    }
    if (key_match(aline, "tolP5")) {
      sscanf(aline, "%*s %lf", &cfg->tolP5);
      cfg->tolP5Set = 1;
      continue;
    }
    if (key_match(aline, "tolP6")) {
      sscanf(aline, "%*s %lf", &cfg->tolP6);
      cfg->tolP6Set = 1;
      continue;
    }
    if (key_match(aline, "tolP7")) {
      sscanf(aline, "%*s %lf", &cfg->tolP7);
      cfg->tolP7Set = 1;
      continue;
    }
    if (key_match(aline, "tolP8")) {
      sscanf(aline, "%*s %lf", &cfg->tolP8);
      cfg->tolP8Set = 1;
      continue;
    }
    if (key_match(aline, "tolP9")) {
      sscanf(aline, "%*s %lf", &cfg->tolP9);
      cfg->tolP9Set = 1;
      continue;
    }
    if (key_match(aline, "tolP10")) {
      sscanf(aline, "%*s %lf", &cfg->tolP10);
      cfg->tolP10Set = 1;
      continue;
    }
    if (key_match(aline, "tolP11")) {
      sscanf(aline, "%*s %lf", &cfg->tolP11);
      cfg->tolP11Set = 1;
      continue;
    }
    if (key_match(aline, "tolP12")) {
      sscanf(aline, "%*s %lf", &cfg->tolP12);
      cfg->tolP12Set = 1;
      continue;
    }
    if (key_match(aline, "tolP13")) {
      sscanf(aline, "%*s %lf", &cfg->tolP13);
      cfg->tolP13Set = 1;
      continue;
    }
    if (key_match(aline, "tolP14")) {
      sscanf(aline, "%*s %lf", &cfg->tolP14);
      cfg->tolP14Set = 1;
      continue;
    }
    if (param_double(aline, "tolShifts", &cfg->tolShifts)) continue;
    if (param_double(aline, "tolRotation", &cfg->tolRotation)) continue;
    if (param_double(aline, "tolLsdPanel", &cfg->tolLsdPanel)) continue;
    if (param_double(aline, "tolP2Panel", &cfg->tolP2Panel)) continue;
    if (param_double(aline, "tolWavelength", &cfg->tolWavelength)) continue;
    if (param_double(aline, "tolParallax", &cfg->tolParallax)) continue;

    // ── Calibration control ──
    if (param_double(aline, "Width", &cfg->Width)) continue;
    if (param_double(aline, "EtaBinSize", &cfg->EtaBinSize)) continue;
    if (param_int(aline, "RBinDivisions", &cfg->RBinWidth)) continue;
    if (param_double(aline, "MultFactor", &cfg->outlierFactor)) continue;
    if (param_int(aline, "MinIndicesForFit", &cfg->MinIndicesForFit)) continue;
    if (param_int(aline, "FitOrWeightedMean", &cfg->FitWeightMean)) continue;
    if (param_int(aline, "nIterations", &cfg->nIterations)) continue;
    if (param_int(aline, "IterOffset", &cfg->iterOffset)) continue;
    if (param_double(aline, "DoubletSeparation", &cfg->DoubletSeparation)) continue;
    if (param_int(aline, "NormalizeRingWeights", &cfg->NormalizeRingWeights)) continue;
    if (param_int(aline, "OutlierIterations", &cfg->OutlierIterations)) continue;
    if (param_int(aline, "RemoveOutliersBetweenIters", &cfg->RemoveOutliersBetweenIters)) continue;
    if (param_int(aline, "ReFitPeaks", &cfg->ReFitPeaks)) continue;
    if (param_double(aline, "TrimmedMeanFraction", &cfg->TrimmedMeanFraction)) continue;
    if (param_int(aline, "WeightByRadius", &cfg->WeightByRadius)) continue;
    if (param_int(aline, "WeightByFitSNR", &cfg->WeightByFitSNR)) continue;
    if (param_int(aline, "WeightByPositionUncertainty", &cfg->WeightByPositionUncertainty)) continue;
    if (param_int(aline, "AdaptiveEtaBins", &cfg->AdaptiveEtaBins)) continue;
    if (param_int(aline, "L2Objective", &cfg->L2Objective)) continue;
    if (param_int(aline, "FitWavelength", &cfg->FitWavelength)) continue;
    if (param_int(aline, "FitParallax", &cfg->FitParallax)) continue;
    if (param_double(aline, "Parallax", &cfg->parallaxIn)) continue;
    if (param_int(aline, "PerPanelLsd", &cfg->PerPanelLsd)) continue;
    if (param_int(aline, "PerPanelDistortion", &cfg->PerPanelDistortion)) continue;
    if (param_int(aline, "GradientCorrection", &cfg->GradientCorrection)) continue;
    if (param_int(aline, "PeakFitMode", &cfg->PeakFitMode)) continue;
    if (param_int(aline, "SubPixelLevel", &cfg->SubPixelLevel)) continue;
    if (param_double(aline, "SubPixelCardinalWidth", &cfg->SubPixelCardinalWidth)) continue;
    if (param_double(aline, "ConvergenceThresholdPPM", &cfg->ConvergenceThresholdPPM)) continue;
    if (param_int(aline, "SkipVerification", &cfg->SkipVerification)) continue;
    if (param_str(aline, "RingDiagnosticsCSV", cfg->RingDiagnosticsCSV, sizeof(cfg->RingDiagnosticsCSV))) continue;
    if (param_int(aline, "ResumeFromCheckpoint", &cfg->ResumeFromCheckpoint)) continue;
    if (param_int(aline, "FixPanelID", &cfg->FixPanelID)) continue;

    // ── FitPos/Strain control ──
    if (param_int(aline, "TopLayer", &cfg->TopLayer)) continue;
    if (param_int(aline, "TakeGrainMax", &cfg->TakeGrainMax)) continue;
    if (param_int(aline, "LocalMaximaOnly", &cfg->LocalMaximaOnly)) continue;
    if (param_double(aline, "MargABC", &cfg->MargABC)) continue;
    if (param_double(aline, "MargABG", &cfg->MargABG)) continue;
    if (param_int(aline, "DebugMode", &cfg->DebugMode)) continue;
    if (param_double(aline, "OmeBinSize", &cfg->OmeBinSize)) continue;
    if (param_double(aline, "WeightMask", &cfg->WeightMask)) continue;
    if (param_double(aline, "WeightFitRMSE", &cfg->WeightFitRMSE)) continue;
    if (param_int(aline, "DoDynamicReassignment", &cfg->DoDynamicReassignment)) continue;

    // ── NF/Sample parameters ──
    if (param_double(aline, "ExcludePoleAngle", &cfg->MinEta)) continue;
    if (param_double(aline, "MinEta", &cfg->MinEta)) continue;
    if (param_double(aline, "Hbeam", &cfg->Hbeam)) continue;
    if (param_double(aline, "Rsample", &cfg->Rsample)) continue;
    if (param_double(aline, "BeamSize", &cfg->BeamSize)) continue;
    if (param_double(aline, "BeamThickness", &cfg->BeamThickness)) continue;

    // ── NF image processing ──
    if (param_int(aline, "Deblur", &cfg->Deblur)) continue;
    if (param_int(aline, "DoLoGFilter", &cfg->DoLoGFilter)) continue;
    if (param_int(aline, "GaussFiltRadius", &cfg->GaussFiltRadius)) continue;
    if (param_double(aline, "GaussWidth", &cfg->GaussWidth)) continue;
    if (param_int(aline, "LoGMaskRadius", &cfg->LoGMaskRadius)) continue;
    if (param_int(aline, "MedFiltRadius", &cfg->MedFiltRadius)) continue;
    if (param_int(aline, "BlanketSubtraction", &cfg->BlanketSubtraction)) continue;
    if (param_int(aline, "SkipImageBinning", &cfg->SkipImageBinning)) continue;

    // ── NF orientation/grid ──
    if (param_double(aline, "StepSizeOrient", &cfg->StepSizeOrient)) continue;
    if (param_double(aline, "StepsizeOrient", &cfg->StepSizeOrient)) continue;  // alias
    if (param_int(aline, "NrOrientations", &cfg->NrOrientations)) continue;
    // EResolution is handled below via key_match to also read num_lambda_samples
    if (param_int(aline, "nDistances", &cfg->nDistances)) continue;
    if (param_int(aline, "NrFilesPerDistance", &cfg->NrFilesPerDistance)) continue;
    if (param_double(aline, "LsdMean", &cfg->LsdMean)) continue;
    if (param_double(aline, "OrientTol", &cfg->OrientTol)) continue;
    if (param_int(aline, "Twins", &cfg->Twins)) continue;
    if (param_double(aline, "GBAngle", &cfg->GBAngle)) continue;
    if (param_double(aline, "MarginOme", &cfg->MarginOme)) continue;
    if (param_double(aline, "MarginEta", &cfg->MarginEta)) continue;
    if (param_double(aline, "MinConfidence", &cfg->MinConfidence)) continue;
    if (param_double(aline, "MinFracAccept", &cfg->MinFracAccept)) continue;

    // ── ProcessGrains / Forward ──
    if (param_int(aline, "nScans", &cfg->nScans)) continue;
    if (param_int(aline, "PhaseNr", &cfg->PhaseNr)) continue;
    if (param_int(aline, "NumPhases", &cfg->NumPhases)) continue;
    if (param_int(aline, "MinNrSpots", &cfg->MinNrSpots)) continue;
    if (param_double(aline, "GlobalPosition", &cfg->GlobalPosition)) continue;
    if (param_str(aline, "OutDirPath", cfg->OutDirPath, sizeof(cfg->OutDirPath))) continue;
    if (param_int(aline, "NoSaveAll", &cfg->NoSaveAll)) continue;
    if (param_double(aline, "GridSize", &cfg->GridSize)) continue;
    if (param_double(aline, "EdgeLength", &cfg->EdgeLength)) continue;
    if (param_int(aline, "GridPoints", &cfg->GridPoints)) continue;
    if (param_int(aline, "WriteLegacyBin", &cfg->WriteLegacyBin)) continue;
    if (param_int(aline, "SimulationBatches", &cfg->SimulationBatches)) continue;

    // ── ForwardSimulation ──
    // InFileName is already handled at line ~94 as InputFileName
    if (param_str(aline, "OutFileName", cfg->OutFileName, sizeof(cfg->OutFileName))) continue;
    if (param_str(aline, "IntensitiesFile", cfg->IntensitiesFile, sizeof(cfg->IntensitiesFile))) continue;
    // MaskFile is handled above in the Masking section (line ~153)
    if (param_int(aline, "WriteSpots", &cfg->WriteSpots)) continue;
    if (param_int(aline, "WriteImage", &cfg->WriteImage)) continue;
    if (param_int(aline, "IsBinary", &cfg->IsBinary)) continue;
    if (param_int(aline, "LoadNr", &cfg->LoadNr)) continue;
    if (param_int(aline, "UpdatedOrientations", &cfg->UpdatedOrientations)) continue;
    if (param_int(aline, "NFOutput", &cfg->NFOutput)) continue;
    if (param_double(aline, "PeakIntensity", &cfg->PeakIntensity)) continue;
    if (param_double(aline, "MaxOutputIntensity", &cfg->MaxOutputIntensity)) continue;
    if (key_match(aline, "RingsToUse")) {
      if (cfg->nRingsToUse < 500) {
        char tmpd[256]; int tmpv;
        if (sscanf(aline, "%s %d", tmpd, &tmpv) == 2)
          cfg->RingsToUse[cfg->nRingsToUse++] = tmpv;
      }
      continue;
    }
    if (key_match(aline, "EResolution")) {
      char tmpd[256];
      sscanf(aline, "%s %lf %d", tmpd, &cfg->EResolution, &cfg->num_lambda_samples);
      continue;
    }

    // ── Panel config ──
    if (param_int(aline, "NPanelsY", &cfg->NPanelsY)) continue;
    if (param_int(aline, "NPanelsZ", &cfg->NPanelsZ)) continue;
    if (param_int(aline, "PanelSizeY", &cfg->PanelSizeY)) continue;
    if (param_int(aline, "PanelSizeZ", &cfg->PanelSizeZ)) continue;
    if (key_match(aline, "PanelGapsY")) {
      char *ptr = (char *)aline + strlen("PanelGapsY");
      while (*ptr && isspace((unsigned char)*ptr)) ptr++;
      int gi = 0;
      int val;
      while (gi < 10 && sscanf(ptr, "%d", &val) == 1) {
        cfg->PanelGapsY[gi++] = val;
        while (*ptr && !isspace((unsigned char)*ptr)) ptr++;
        while (*ptr && isspace((unsigned char)*ptr)) ptr++;
      }
      continue;
    }
    if (key_match(aline, "PanelGapsZ")) {
      char *ptr = (char *)aline + strlen("PanelGapsZ");
      while (*ptr && isspace((unsigned char)*ptr)) ptr++;
      int gi = 0;
      int val;
      while (gi < 10 && sscanf(ptr, "%d", &val) == 1) {
        cfg->PanelGapsZ[gi++] = val;
        while (*ptr && !isspace((unsigned char)*ptr)) ptr++;
        while (*ptr && isspace((unsigned char)*ptr)) ptr++;
      }
      continue;
    }
    if (param_str(aline, "PanelShiftsFile", cfg->PanelShiftsFile, sizeof(cfg->PanelShiftsFile))) continue;

    // ── Lineout parameters ──
    if (param_double(aline, "RBinSize", &cfg->lineoutRBinSize)) continue;
    if (param_double(aline, "RMin", &cfg->lineoutRMin)) continue;
    if (param_double(aline, "RMax", &cfg->lineoutRMax)) continue;

    // ── Silently consumed (no-op) keys ──
    if (key_match(aline, "DistortionOrder")) continue;

    // Unrecognized keys are silently skipped.
  }

  fclose(f);

  // Derive NrPixels
  cfg->NrPixels = (cfg->NrPixelsY > cfg->NrPixelsZ) ? cfg->NrPixelsY : cfg->NrPixelsZ;

  // Apply tolerance fallbacks
  midas_apply_tol_defaults(cfg);

  // GE binary (DataType 1) has an 8192-byte header that must be skipped.
  // Auto-default if user didn't set HeadSize explicitly.
  if (cfg->DataType == 1 && cfg->HeadSize == 0)
    cfg->HeadSize = 8192;

  return 0;
}
