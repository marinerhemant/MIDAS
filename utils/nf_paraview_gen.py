#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

##
# HDF5 (compressed output) file arrangement:
#            Group                  Dataset
#		|--- FFGrainID           -- FFGrainNr          -- Nonsense value: -15, always uint
#		|--- GrainNrs            -- GrainNr            -- Nonsense value: -15, always uint
#		|--- KernelAverageMiso   -- KAM                -- Nonsense value: -15, range 0.180
#		|--- EulerAngles        |-- EulerAngle1        -- Nonsense value: -15, range +-2pi
#		|                       |-- EulerAngle2        -- Nonsense value: -15, range +-2pi
#		|                       |-- EulerAngle3        -- Nonsense value: -15, range +-2pi
#		|--- Confidence          -- ConfidenceValue    -- Nonsense value: -15, range 0...1
#		|--- PhaseNumber         -- PhaseNr            -- Nonsense value: -15, range 1...n
##

fillVal = -15 # this value can be used to filter out nonsense values.

### Only modify the following arguments:
### Also look at where variable FileName is defined to see if the file arrangement is different (Line266)
####
sampleName = 'kgt1147_def'
filestem = 'MicrostructureTxt_kgt1147_def_nf_layer'
outfn = 'MicOut_kgt1147_def'
spaceGroup = 229 # This is used for misorientation calculation
startnr = 1
endnr = 101
minConfidence = 0.6
orientTol = 10.0 # In degrees, used to define grains
zspacing = -5
xyspacing = 5  # X and Y spacing are equal
xExtent = 600 # Maximum Extent of xValues in um
			   # (this should be a bit larger than your sample diameter or edge length)
yExtent = 1400 # Maximum Extent of yValues in um
			   # (this should be a bit larger than your sample diameter or edge length)
####

import math
import sys, os
import numpy as np
import h5py
import time
from numba import jit
from math import cos, sin, tan, sqrt, asin, acos, atan
import ctypes
rad2deg = 57.2957795130823

Dims = np.array([0,0,0])
Dims = Dims.astype(int)
Dims[1] = int(xExtent/abs(xyspacing))
Dims[2] = int(yExtent/abs(xyspacing))
Dims[0] = (endnr - startnr + 1)
print('Dimensions of final array:')
print(Dims)
startPos = 0
grainIDs = np.zeros((Dims))
Euler1 = np.zeros((Dims))
Euler2 = np.zeros((Dims))
Euler3 = np.zeros((Dims))
Confidence = np.zeros((Dims))
PhaseNr = np.zeros((Dims))
Euler1.fill(fillVal)
Euler2.fill(fillVal)
Euler3.fill(fillVal)
Confidence.fill(fillVal)
PhaseNr.fill(fillVal)
grainIDs.fill(fillVal)
dataNr = 0;
outarr = np.zeros((Dims[1],Dims[2],7))
outarr = outarr.astype(float)
dimarr = np.array([Dims[1],Dims[2],abs(xyspacing)])
dimarr = dimarr.astype(int)

def writeHDF5File(grID,eul1,eul2,eul3,conf,phNr,kam,grNr,fileID):
	f = h5py.File(fileID,'w')
	grainIDs = f.create_group('FFGrainID')
	Euls = f.create_group('EulerAngles')
	Conf = f.create_group('Confidence')
	PhaseNr = f.create_group('PhaseNumber')
	GrainNrs = f.create_group('GrainNrs')
	KAMs = f.create_group('KernelAverageMiso')
	GrainID = grainIDs.create_dataset('FFGrainNr',data=grID,compression="gzip")
	Euler1 = Euls.create_dataset('EulerAngle1',data=eul1,compression="gzip")
	Euler2 = Euls.create_dataset('EulerAngle2',data=eul2,compression="gzip")
	Euler3 = Euls.create_dataset('EulerAngle3',data=eul3,compression="gzip")
	confidence = Conf.create_dataset('ConfidenceValue',data=conf,compression="gzip")
	PhaseNrs = PhaseNr.create_dataset('PhaseNr',data=phNr,compression="gzip")
	GrainNr = GrainNrs.create_dataset('GrainNr',data=grNr,compression="gzip")
	KAM = KAMs.create_dataset('KAM',data=kam,compression="gzip")
	f.close()

def writeXMLXdmf(dims,deltas,fn,h5fn,sample_name):
	f = open(fn,'w')
	# Header
	f.write('<?xml version="1.0" ?>\n')
	f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
	f.write('<Xdmf xmlns:xi="http:#www.w3.org/2003/XInclude" Version="2.2">\n')
	f.write('<Information Name="%s" Value="0"/>\n'%sample_name)
	f.write('<Domain>\n')
	# Info about topology and Geometry
	f.write('<Grid Name="Structured Grid" GridType="Uniform">\n')
	f.write('<Topology TopologyType="3DCoRectMesh" Dimensions="%d %d %d">\n'%(dims[0]+1,dims[1]+1,dims[2]+1))
	f.write('</Topology>\n')
	f.write('<Geometry Type="ORIGIN_DXDYDZ">\n')
	f.write('<!-- Origin -->\n')
	f.write('<DataItem Dimensions="3" NumberType="Float" Format="XML">\n')
	f.write('%lf %lf %lf\n'%(-dims[0]/2*(abs(deltas[2])/deltas[2]),-dims[1]/2,-dims[2]/2))
	f.write('</DataItem>\n')
	f.write('<!-- DXDYDZ -->\n')
	f.write('<DataItem Dimensions="3" NumberType="Float" Format="XML">\n')
	f.write('%lf %lf %lf\n'%(deltas[2],deltas[0],deltas[1]))
	f.write('</DataItem>\n')
	f.write('</Geometry>\n')
	# Data: FFGrainID, EulerAngles, Confidence, GrainNr, KernelAverageMisorientation, PhaseNr
	f.write('<Attribute Name="FFGrainID" AttributeType="Scalar" Center="Cell">\n')
	f.write('<DataItem Format="HDF" Dimensions="%d %d %d" NumberType="Int">\n'%(dims[0],dims[1],dims[2]))
	f.write('%s.h5:/FFGrainID/FFGrainNr\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('<Attribute Name="EulerAngles" AttributeType="Vector" Center="Cell">\n')
	f.write('<DataItem ItemType="Function" Dimensions="%d %d %d 3"\n'%(dims[0],dims[1],dims[2]))
	f.write('	Function=" JOIN( $0,$1,$2 ) ">\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/EulerAngles/EulerAngle1\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/EulerAngles/EulerAngle2\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/EulerAngles/EulerAngle3\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('<Attribute Name="Confidence" AttributeType="Scalar" Center="Cell">\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/Confidence/ConfidenceValue\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('<Attribute Name="GrainNr" AttributeType="Scalar" Center="Cell">\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Int" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/GrainNrs/GrainNr\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('<Attribute Name="KernelAverageMisorientation" AttributeType="Scalar" Center="Cell">\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/KernelAverageMiso/KAM\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('<Attribute Name="PhaseNumber" AttributeType="Scalar" Center="Cell">\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/PhaseNumber/PhaseNr\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('</Grid>\n')
	f.write('</Domain>\n')
	f.write('</Xdmf>\n')
	# Close the file
	f.close()

@jit('void(float64[:,:],int64[:],float64[:,:,:])',nopython=True,nogil=True)
def mapData(data,dims,outArr):
	spacing = dims[2]
	nrRows,nrCols = data.shape
	outArr.fill(fillVal)
	gridSpacing = data[0,5]
	extent = int(math.ceil(gridSpacing/spacing))
	outArr[:,:,6] = 10000
	for i in range(nrRows):
		xPos = data[i,4]
		yPos = data[i,3]
		xBinNr = int(xPos/spacing + dims[0]/2)
		yBinNr = int(yPos/spacing + dims[1]/2)
		xT = spacing*(xBinNr - dims[0]/2)
		yT = spacing*(yBinNr - dims[1]/2)
		distt = math.sqrt((xT-xPos)*(xT-xPos)+(yT-yPos)*(yT-yPos))
		if (xBinNr < 0) or (xBinNr > dims[0]-1) or (yBinNr < 0) or (yBinNr > dims[1]-1):
			continue
		else:
			if (outArr[xBinNr,yBinNr,6] > distt):
				outArr[xBinNr,yBinNr,0] = data[i,0]
				outArr[xBinNr,yBinNr,1] = data[i,7]
				outArr[xBinNr,yBinNr,2] = data[i,8]
				outArr[xBinNr,yBinNr,3] = data[i,9]
				outArr[xBinNr,yBinNr,4] = data[i,10]
				outArr[xBinNr,yBinNr,5] = data[i,11]
				outArr[xBinNr,yBinNr,6] = distt
		for j in range(-extent,extent+1):
			for k in range(-extent,extent+1):
				xBinT = xBinNr + j
				yBinT = yBinNr + k
				if  (xBinT < 0) or (xBinT > dims[0]-1) or (yBinT < 0) or (yBinT > dims[1]-1):
					continue
				xT2 = spacing*(xBinT - dims[0]/2)
				yT2 = spacing*(yBinT - dims[1]/2)
				distt2 = math.sqrt(((xT2-xPos)*(xT2-xPos))+((yT2-yPos)*(yT2-yPos)))
				if (outArr[xBinT,yBinT,6] > distt2):
					outArr[xBinNr,yBinNr,0] = data[i,0]
					outArr[xBinNr,yBinNr,1] = data[i,7]
					outArr[xBinNr,yBinNr,2] = data[i,8]
					outArr[xBinNr,yBinNr,3] = data[i,9]
					outArr[xBinNr,yBinNr,4] = data[i,10]
					outArr[xBinNr,yBinNr,5] = data[i,11]
					# ~ outArr[xBinT,yBinT,0:6] = data[i,[0,7,8,9,10,11]]
					outArr[xBinT,yBinT,6] = distt2

for fnr in range(startnr,endnr+1):
	print('LayerNr: '+ str(fnr))
	#FileName = sampleName + 'Layer' + str(fnr) + '/' + filestem + str(fnr) + '.mic'
	FileName = filestem + str(fnr) + '.mic'
	t1 = time.time()
	micfiledata = np.genfromtxt(FileName,skip_header=4)
	data = micfiledata[micfiledata[:,10] > minConfidence,:]
	data = data.astype(float)
	mapData(data,dimarr,outarr)
	print(time.time() - t1)
	grainIDs[dataNr,:,:] = outarr[:,:,0]
	Euler1[dataNr,:,:] = outarr[:,:,1]
	Euler2[dataNr,:,:] = outarr[:,:,2]
	Euler3[dataNr,:,:] = outarr[:,:,3]
	Confidence[dataNr,:,:] = outarr[:,:,4]
	PhaseNr[dataNr,:,:] = outarr[:,:,5]
	dataNr += 1

Euler1.astype(np.float64).tofile('EulerAngles1.bin')
Euler2.astype(np.float64).tofile('EulerAngles2.bin')
Euler3.astype(np.float64).tofile('EulerAngles3.bin')
KamArr = np.zeros((Dims))

# We need to provide the following:
# orientTol, dims[0], dims[1], dims[2], fillVal, spaceGroup.
home = os.path.expanduser("~")
grainsCalc = ctypes.CDLL(home + "/opt/MIDAS/NF_HEDM/bin/NFGrainsCalc.so")
grainsCalc.calcGrainNrs.argtypes = (ctypes.c_double,
										ctypes.c_int,
										ctypes.c_int,
										ctypes.c_int,
										ctypes.c_double,
										ctypes.c_int,
									)
grainsCalc.calcGrainNrs.restype = None
grainsCalc.calcGrainNrs(orientTol,Dims[0],Dims[1],Dims[2],fillVal,spaceGroup)
grains = np.fromfile('GrainNrs.bin',dtype=np.int32)
grains = grains.reshape((Dims))

# write files
writeHDF5File(grainIDs.astype(np.int32),Euler1.astype(np.float32),Euler2.astype(np.float32),Euler3.astype(np.float32),Confidence.astype(np.float32),PhaseNr.astype(np.float32),KamArr.astype(np.float32),grains.astype(np.int32),outfn+'.h5')
writeXMLXdmf(Dims,[xyspacing,xyspacing,zspacing],outfn+'.xmf',outfn,sampleName)
