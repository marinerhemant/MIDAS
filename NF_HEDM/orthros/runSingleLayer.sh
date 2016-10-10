#!/bin/bash -eu

source ${HOME}/.MIDAS/pathsNF
cmdname=$(basename $0)

if [[ ${#*} != 5 ]];
then
  echo "Usage: ${cmdname} top_parameter_file(full_path) nCPUS FFSeedOrientations ProcessImages EmailAddress"
  echo "Eg. ${cmdname} ParametersFile.txt 384 1(or 0) 1(or 0) hsharma@anl.gov"
  echo "This will produce the output in the run folder, called Microstructure.mic."
  echo "FFSeedOrientations is when either Orientations exist already (0) or when you provide a FF Orientation file (1)."
  echo "ProcessImages is whether you want to process the diffraction images (1) or if they were processed earlier (0)."
  echo "NOTE: run from the folder with the Key.txt, DiffractionSpots.txt, OrientMat.txt and ParametersFile.txt"
  echo "At least the parameters file should be in the folder from where the command is executed."
  exit 1
fi

if [[ $1 == /* ]]; then TOP_PARAM_FILE=$1; else TOP_PARAM_FILE=$(pwd)/$1; fi
NCPUS=$2
FFSeedOrientations=$3
ProcessImages=$4

# Go to the right folder
DataDirectory=$( awk '$1 ~ /^DataDirectory/ { print $2 }' ${TOP_PARAM_FILE} )
cd ${DataDirectory}

STEM=$( awk '$1 ~ /^ReducedFileName/ { print $2 }' ${TOP_PARAM_FILE} )
CHART=/
FOLDER=${STEM%$CHART*}
mkdir -p ${FOLDER}

# Make hkls.csv
${BINFOLDER}/GetHKLList ${TOP_PARAM_FILE}

GrainsFile=$( awk '$1 ~ /^GrainsFile/ { print $2 }' ${TOP_PARAM_FILE} )
SeedOrientations=$( awk '$1 ~ /^SeedOrientations/ { print $2 }' ${TOP_PARAM_FILE} )
Micf=$(awk '$1 ~ /^MicFileBinary/ { print $2 }' ${TOP_PARAM_FILE})
MICFN=${DataDirectory}/${Micf}

if [[ -n $MICFN ]];
then 
  rm -rf $MICFN
fi

# Generate SeedOrientations
if [[ ${FFSeedOrientations} == 1 ]];
then
  ${BINFOLDER}/GenSeedOrientationsFF2NFHEDM $GrainsFile $SeedOrientations
fi

NrOrientations=$( wc -l ${SeedOrientations} | awk '{print $1}' )
echo "NrOrientations ${NrOrientations}" >> ${TOP_PARAM_FILE}

# Make HexGrid and DiffractionSpots
${BINFOLDER}/MakeHexGrid $TOP_PARAM_FILE

# Filter HexGrid
if [[ -n $( awk '$1 ~ /^GridMask/ { print }' ${TOP_PARAM_FILE} ) ]];
then
  Xmin=$( awk '$1 ~ /^GridMask/ { print $2 }' ${TOP_PARAM_FILE} )
  Xmax=$( awk '$1 ~ /^GridMask/ { print $3 }' ${TOP_PARAM_FILE} )
  Ymin=$( awk '$1 ~ /^GridMask/ { print $4 }' ${TOP_PARAM_FILE} )
  Ymax=$( awk '$1 ~ /^GridMask/ { print $5 }' ${TOP_PARAM_FILE} )
  awk ' { if (($3 >= '$Xmin') && ($3 <= '$Xmax') && ($4 >= '$Ymin') && ($4 <= '$Ymax'))  print }' < grid.txt >grid_new.txt
  mv grid.txt grid_old.txt
  wc -l <grid_new.txt >grid.txt
  cat grid_new.txt >>grid.txt
fi
${BINFOLDER}/MakeDiffrSpots $TOP_PARAM_FILE

# Calculate nGridPoints
nGridPoints=$( wc -l grid.txt | awk '{print $1}' )
(( nGridPoints = nGridPoints - 1 ))
STARTNR=1
ENDNR=${nGridPoints}
echo "StartNr: ${STARTNR}"
echo "EndNr: ${ENDNR}"

# Proess Images
NDISTANCES=$( awk '$1 ~ /^nDistances/ { print $2 }' ${TOP_PARAM_FILE} )
NRFILESPERDISTANCE=$( awk '$1 ~ /^NrFilesPerDistance/ { print $2 }' ${TOP_PARAM_FILE} )
NRPIXELS=$( awk '$1 ~ /^NrPixels/ { print $2 }' ${TOP_PARAM_FILE} )
if [[ ${ProcessImages} == 1 ]];
then
  echo "Median"
  ${SWIFTDIR}/swift -sites.file ${PFDIR}sites${NCPUS}.xml -tc.file ${PFDIR}tc.data -config ${PFDIR}cf ${PFDIR}ProcessMedianParallel.swift \
    -paramfile=${TOP_PARAM_FILE} -NrLayers=${NDISTANCES} -NrFilesPerLayer=${NRFILESPERDISTANCE} -NrPixels=${NRPIXELS}
  echo "Image"
  ${SWIFTDIR}/swift -sites.file ${PFDIR}sites${NCPUS}.xml -tc.file ${PFDIR}tc.data -config ${PFDIR}cf ${PFDIR}ProcessImagesParallel.swift \
    -paramfile=${TOP_PARAM_FILE} -NrLayers=${NDISTANCES} -NrFilesPerLayer=${NRFILESPERDISTANCE} -NrPixels=${NRPIXELS}
fi

# MMapImageInfo and scp all the bin files to /dev/shm of each node
${BINFOLDER}/MMapImageInfo ${TOP_PARAM_FILE}
FileList="${DataDirectory}/SpotsInfo.bin ${DataDirectory}/DiffractionSpots.bin ${DataDirectory}/Key.bin ${DataDirectory}/OrientMat.bin"
if [[ ${NCPUS} == 128 ]]
then
	ssh puppy21 cp -v $FileList /dev/shm
	ssh puppy22 cp -v $FileList /dev/shm
	ssh puppy37 cp -v $FileList /dev/shm
	ssh puppy39 cp -v $FileList /dev/shm
	ssh puppy41 cp -v $FileList /dev/shm
	ssh puppy43 cp -v $FileList /dev/shm
	ssh puppy44 cp -v $FileList /dev/shm
elif [[ ${NCPUS} == 64 ]]
then
	ssh pup0100 cp -v $FileList /dev/shm
elif [[ ${NCPUS} == 320 ]]
then
	ssh pup0101 cp -v $FileList /dev/shm/
	ssh pup0102 cp -v $FileList /dev/shm/
	ssh pup0103 cp -v $FileList /dev/shm/
	ssh pup0104 cp -v $FileList /dev/shm/
	ssh pup0105 cp -v $FileList /dev/shm/
elif [[ ${NCPUS} == 384 ]]
then
	ssh pup0100 cp -v $FileList /dev/shm/
	ssh pup0101 cp -v $FileList /dev/shm/
	ssh pup0102 cp -v $FileList /dev/shm/
	ssh pup0103 cp -v $FileList /dev/shm/
	ssh pup0104 cp -v $FileList /dev/shm/
	ssh pup0105 cp -v $FileList /dev/shm/
fi

# Process data
${SWIFTDIR}/swift -sites.file ${PFDIR}/sites${NCPUS}.xml -tc.file ${PFDIR}/tc.data -config ${PFDIR}/cf ${PFDIR}/FitOrientation.swift \
  -startnr=${STARTNR} -endnr=${ENDNR} -paramfile=${TOP_PARAM_FILE} -micfn=${MICFN}

# Parse Mic file
${BINFOLDER}/ParseMic ${TOP_PARAM_FILE}
RC=${?}
if [[ RC != 0 ]]
then
  echo "RC == 0."
fi

EmailAdd=$5
echo "The run started with ${cmdname} $@ has finished, please check." | mail -s "MIDAS run finished" ${EmailAdd}
exit 0
