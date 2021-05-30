cd $1
~/opt/MIDAS/FF_HEDM/bin/MergeOverlappingPeaksAll $2
~/opt/MIDAS/FF_HEDM/bin/CalcRadiusAll $2
~/opt/MIDAS/FF_HEDM/bin/FitSetup $2
~/opt/MIDAS/FF_HEDM/bin/SaveBinData
