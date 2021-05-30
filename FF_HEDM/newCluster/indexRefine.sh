cd $1
nSpots=$( wc -l < SpotsToIndex.csv )
~/opt/MIDAS/FF_HEDM/bin/IndexerOMP paramstest.txt $2 $3 ${nSpots} $4
~/opt/MIDAS/FF_HEDM/bin/FitPosOrStrainsOMP paramstest.txt $2 $3 ${nSpots} $4
