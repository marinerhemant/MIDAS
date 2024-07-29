#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

LOCAL_DIR=$( pwd )/Cluster
CHART=/
FF_MIDAS_DIR=${LOCAL_DIR%$CHART*}
BINFOLDER=${FF_MIDAS_DIR}/bin

##### Put correct folder paths
configdir=${HOME}/.MIDAS
configfile=${configdir}/paths
BLOSCDIR=${configdir}/BLOSC/lib64
BLOSC1DIR=${configdir}/BLOSC1/lib64
FFTWDIR=${configdir}/FFTW/lib
HDFDIR=${configdir}/HDF5/lib
LIBTIFFDIR=${configdir}/LIBTIFF/lib
LIBZIPDIR=${configdir}/LIBZIP/lib64
NLOPTDIR=${configdir}/NLOPT/lib
ZLIBDIR=${configdir}/ZLIB/lib
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
echo "export LD_LIBRARY_PATH=${BLOSC1DIR}:${BLOSCDIR}:${FFTWDIR}:${HDFDIR}:${LIBTIFFDIR}:${LIBZIPDIR}:${NLOPTDIR}:${ZLIBDIR}:${LD_LIBRARY_PATH}" > ${configfile}
echo "BINFOLDER=${BINFOLDER}" >> ${configfile}
echo "PFDIR=${LOCAL_DIR}" >> ${configfile}
echo "SWIFTDIR=${HOME}/.MIDAS/swift/bin" >> ${configfile}
ln -s ${LOCAL_DIR}/RealTimeMultipleLayersSingleSwiftJob.sh ${configdir}/MIDAS_V5_FarField_Layers
ln -s ${LOCAL_DIR}/TrackGrains.py ${configdir}/MIDAS_V5_FarField_TrackGrains.py
ln -s ${BINFOLDER}/Calibrant ${configdir}/MIDAS_V5_FarField_Calibration
ln -s ${BINFOLDER}/FitWedge ${configdir}/MIDAS_V5_FarField_Wedge
ln -s ${BINFOLDER}/FitTiltX ${configdir}/MIDAS_V5_FarField_TiltX
ln -s ${BINFOLDER}/ProcessGrains ${configdir}/MIDAS_V5_FarField_ProcessGrains
ln -s ${FF_MIDAS_DIR}/CalibrationParametersExample.txt ${configdir}/MIDAS_V5_FarField_Calibration_ExampleFile.txt
ln -s ${BINFOLDER}/FitGrain ${configdir}/MIDAS_V5_FarField_FitGrain

echo "Congratulations, you can now use MIDAS to run FarField analysis"
echo "Go to ${HOME}/.MIDAS folder, there are MIDAS_V... files for running analysis"
