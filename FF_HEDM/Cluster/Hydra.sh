#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

source ${HOME}/.MIDAS/paths

cmdname=$(basename $0)

echo "Analysis code for Hydra:"
echo "Version: 1, 2017/03/30, in case of problems contact hsharma@anl.gov"

if [[ ${#*} != 7 ]]
then
  echo "Provide ParametersFile StartLayerNr EndLayerNr DoPeakSearch Number of NODES to use MachineName EmailAddress!"
  echo "EG. ${cmdname} parameters.txt 1 1 1 (or 0) 6 orthros(orthrosextra,local,rice) hsharma@anl.gov"
  echo "the source parameter file should not have ring numbers and layer numbers in it."
  echo "If DoPeakSearch is 0, the parameter file must have a folder name for each layer (in order) to look into and redo analysis."
  echo "For example FolderName Ruby_scan2_Layer1_Analysis_Time_2016_09_19_17_11_07"
  echo "For example FolderName Ruby_scan2_Layer2_Analysis_Time_2016_09_19_17_13_23"
  echo "If these are not provided, it will check the parent folder and if multiple"
  echo "analyses are present for a layer, will take the latest one."
  echo "If DoPeakSearch is 0, it will overwrite the results in the directory it works."
  exit 1
fi

${PFDIR}/prepareFilesHydra.py $1
nDetectors=4
for (( detnr=1; detnr<$nDetectors; detnr++ ))
do
	cd Detector$detnr
	echo "Processing Detector number ${detnr}"
	${PFDIR}/PeakSearchHydra.sh ${1}_det${detnr} $2 $3 $4 $5 $6 $7
done
