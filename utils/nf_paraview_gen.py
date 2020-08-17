#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

import math
import sys, os
import numpy as np
import h5py
import time
from numba import jit
from math import cos, sin, tan, sqrt, asin, acos, atan
import ctypes
import matplotlib.pyplot as plt
rad2deg = 57.2957795130823

##
# HDF5 (compressed output) file arrangement:
#            Group                  Dataset
#		|--- FFGrainID           -- FFGrainNr          -- Nonsense value: -15, always uint
#		|--- Grains             |-- GrainNr            -- Nonsense value: -15, always uint
#		|                       |-- GrainSize          -- Nonsense value: -15, always uint
#		|--- KernelAverageMiso   -- KAM                -- Nonsense value: -15, range 0.180
#		|--- EulerAngles        |-- EulerAngle1        -- Nonsense value: -15, range +-2pi
#		|                       |-- EulerAngle2        -- Nonsense value: -15, range +-2pi
#		|                       |-- EulerAngle3        -- Nonsense value: -15, range +-2pi
#		|--- Confidence          -- ConfidenceValue    -- Nonsense value: -15, range 0...1
#		|--- PhaseNumber         -- PhaseNr            -- Nonsense value: -15, range 1...n
#		|--- Positions          |-- X                  --                      [microns]
#		|                       |-- Y                  --                      [microns]
#		|                       |-- Z                  --                      [microns]
##

# Before running this, please execute the following command in bash:
#		ulimit -S -s 1310720

fillVal = -15 # this value can be used to filter out nonsense values.

### Only modify the following arguments:
### Also look at where variable FileName is defined to see if the file arrangement is different (Line364)
####
sampleName = 'HeatHTNS9_crack_NF'
filestem = 'MicrostructureText_Layer' #### CHECK LINE 364
outfn = 'MicHeatHTNS9'
formula = 'NiTi7'
materialName = 'NS9'
sample = 'HeatHTNS9'
scanN = 'Begin'
spaceGroup = 194 # This is used for misorientation calculation
startnr = 0
endnr = 36
thisPhaseNr = 1
LatC = np.array([2.9243, 2.9243, 4.6726 ,90.0, 90.0, 120.0],dtype=np.float32)
minConfidence = 0.3
orientTol = 10.0 # In degrees, used to define grains
zspacing = -2
xyspacing = 2  # X and Y spacing are equal, should be equal to the edge_size used during reconstruction
startZ = 0 # Starting Z position
xExtent = 1400 # Maximum Extent of xValues in um
			   # (this should be larger than 2x the sample radius or 2x the distance between the farther edge of the sample and the rotation axis)
yExtent = 1400 # Maximum Extent of yValues in um
			   # (this should be larger than 2x the sample radius or 2x the distance between the farther edge of the sample and the rotation axis)
####

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
xVals = np.zeros((Dims))
yVals = np.zeros((Dims))
zVals = np.zeros((Dims))
Euler1.fill(fillVal)
Euler2.fill(fillVal)
Euler3.fill(fillVal)
Confidence.fill(fillVal)
PhaseNr.fill(fillVal)
grainIDs.fill(fillVal)
dataNr = 0
outarr = np.zeros((Dims[1],Dims[2],9))
outarr = outarr.astype(float)
dimarr = np.array([Dims[1],Dims[2],abs(xyspacing)])
dimarr = dimarr.astype(int)

origHead = '''# TEM_PIXperUM          1.000000
           # x-star                1
           # y-star                1
           # z-star                1
           # WorkingDistance       1.000000
           #
           # Phase 1
           # MaterialName
           # Formula
           # Info
           # Symmetry              62
           # LatticeConstants
           # NumberFamilies
           # hklFamilies
           # hklFamilies
           # hklFamilies
           # hklFamilies
           # Categories 0 0 0 0 0
           #
           # GRID: SqrGrid
           # XSTEP:
           # YSTEP:
           # NCOLS_ODD:
           # NCOLS_EVEN:
           # NROWS:
           #
           # OPERATOR:
           #
           # SAMPLEID:
           #
           # SCANID:
           #'''

compound_dt = np.dtype({'names':['H','K','L','Solution 1','Diffraction Intensity','Solution 2'],'formats':['<i4','<i4','<i4','<i1','<f4','<i1']})

def writeH5EBSDFile(eul1,eul2,eul3,conf,phNr,grID,fileID):
	f = h5py.File(fileID,'w')
	f.attrs['FileVersion'] = np.array([5],dtype=np.int32)
	ETA = np.array([0.],dtype=np.float32)
	f.create_dataset('EulerTransformationAngle',data=ETA)
	ETAx = np.array([0.,0.,1.],dtype=np.float32)
	f.create_dataset('EulerTransformationAxis',data=ETAx)
	STA = np.array([0.],dtype=np.float32)
	f.create_dataset('SampleTransformationAngle',data=STA)
	STAx = np.array([0.,0.,1.],dtype=np.float32)
	f.create_dataset('SampleTransformationAxis',data=STAx)
	Idx = np.arange(1,Dims[0]+1).astype(np.int32)
	f.create_dataset('Index',data=Idx)
	XP = np.array(Dims[1],dtype=np.int64)
	f.create_dataset('Max X Points',data=XP)
	YP = np.array(Dims[2],dtype=np.int64)
	f.create_dataset('Max Y Points',data=YP)
	htl = 1 if zspacing < 0 else 0
	STO = np.array([htl],dtype=np.uint32)
	f.create_dataset('Stacking Order',data=STO)
	xyR = np.array([xyspacing],dtype=np.float32)
	zR = np.array([abs(zspacing)],dtype=np.float32)
	f.create_dataset('X Resolution',data=xyR)
	f.create_dataset('Y Resolution',data=xyR)
	f.create_dataset('Z Resolution',data=zR)
	f.create_dataset('Manufacturer',(1,),data=np.string_("TSL"))
	zSI = np.array(1,dtype=np.int64)
	zEI = np.array(Dims[0],dtype=np.int64)
	f.create_dataset('ZStartIndex',data=zSI)
	f.create_dataset('ZEndIndex',data=zEI)
	x = np.arange(0,Dims[1]*xyspacing,xyspacing) # We need to now map each voxel
	y = np.arange(0,Dims[2]*xyspacing,xyspacing) # We need to now map each voxel
	xv,yv = np.meshgrid(x,y)
	xPosArr = xv.reshape(Dims[1]*Dims[2]).astype(np.float32)
	yPosArr = yv.reshape(Dims[1]*Dims[2]).astype(np.float32)
	for i in range(1,Dims[0]+1):
		gpL = f.create_group(str(i))
		dGPL = gpL.create_group('Data')
		hGPL = gpL.create_group('Header')
		hGPL.create_dataset('GRID',data=np.string_('SqrGrid'))
		hGPL.create_dataset('SAMPLEID',data=np.string_(sample))
		hGPL.create_dataset('SCANID',data=np.string_(scanN))
		nCev = np.array(Dims[1],dtype=np.uint32)
		hGPL.create_dataset('NCOLS_EVEN',data=nCev)
		hGPL.create_dataset('NCOLS_ODD',data=nCev)
		nRows = np.array(Dims[2],dtype=np.uint32)
		hGPL.create_dataset('NROWS',data=nRows)
		hGPL.create_dataset('TEM_PIXperUM',data=np.array([1],dtype=np.float32))
		hGPL.create_dataset('x-star',data=np.array([1],dtype=np.float32))
		hGPL.create_dataset('y-star',data=np.array([1],dtype=np.float32))
		hGPL.create_dataset('z-star',data=np.array([1],dtype=np.float32))
		hGPL.create_dataset('WorkingDistance',data=np.array([1],dtype=np.float32))
		hGPL.create_dataset('OPERATOR',data=np.string_(' '))
		hGPL.create_dataset('OriginalFile',data=np.string_(' '))
		hGPL.create_dataset('OriginalHeader',data=np.string_(origHead))
		pHGPL = hGPL.create_group('Phases')
		phHGPL = pHGPL.create_group(str(thisPhaseNr))
		phHGPL.create_dataset('Formula',data=np.string_(formula))
		phHGPL.create_dataset('Info',data=np.string_(' '))
		phHGPL.create_dataset('MaterialName',data=np.string_(materialName))
		phHGPL.create_dataset('LatticeConstants',data=LatC)
		phHGPL.create_dataset('Phase',data=np.array(thisPhaseNr,dtype=np.int32))
		phHGPL.create_dataset('Symmetry',data=np.array([62],dtype=np.int32))
		phHGPL.create_dataset('NumberFamilies',data=np.array([4],dtype=np.int32))
		hklGPL = phHGPL.create_group('hklFamilies')
		arrDs0 = np.array([1]).astype(compound_dt)
		hklGPL.create_dataset('0',data=arrDs0)
		hklGPL.create_dataset('1',data=arrDs0)
		hklGPL.create_dataset('2',data=arrDs0)
		hklGPL.create_dataset('3',data=arrDs0)
		hGPL.create_dataset('XSTEP',data=xyR)
		hGPL.create_dataset('YSTEP',data=xyR)
		grThis = grID[i-1,:,:].astype(np.int32).reshape((Dims[1]*Dims[2]))
		dGPL.create_dataset('FFGrainID',data=grThis)
		CIThis = conf[i-1,:,:].astype(np.float32).reshape((Dims[1]*Dims[2]))
		dGPL.create_dataset('Confidence Index',data=CIThis)
		Phi1 = eul1[i-1,:,:].astype(np.float32).reshape((Dims[1]*Dims[2]))
		Phi = eul2[i-1,:,:].astype(np.float32).reshape((Dims[1]*Dims[2]))
		Phi2 = eul3[i-1,:,:].astype(np.float32).reshape((Dims[1]*Dims[2]))
		dGPL.create_dataset('Phi1',data=Phi1)
		dGPL.create_dataset('Phi',data=Phi)
		dGPL.create_dataset('Phi2',data=Phi2)
		dGPL.create_dataset('X Position',data=xPosArr)
		dGPL.create_dataset('Y Position',data=yPosArr)
	f.close()

def writeHDF5File(grID,eul1,eul2,eul3,conf,phNr,kam,grNr,grSz,x,y,z,fileID):
	f = h5py.File(fileID,'w')
	grainIDs = f.create_group('FFGrainID')
	Euls = f.create_group('EulerAngles')
	Conf = f.create_group('Confidence')
	PhaseNr = f.create_group('PhaseNumber')
	Grains = f.create_group('Grains')
	KAMs = f.create_group('KernelAverageMiso')
	Poss = f.create_group('Positions')
	GrainID = grainIDs.create_dataset('FFGrainNr',data=grID,compression="gzip")
	Euler1 = Euls.create_dataset('EulerAngle1',data=eul1,compression="gzip")
	Euler2 = Euls.create_dataset('EulerAngle2',data=eul2,compression="gzip")
	Euler3 = Euls.create_dataset('EulerAngle3',data=eul3,compression="gzip")
	confidence = Conf.create_dataset('ConfidenceValue',data=conf,compression="gzip")
	PhaseNrs = PhaseNr.create_dataset('PhaseNr',data=phNr,compression="gzip")
	GrainNr = Grains.create_dataset('GrainNr',data=grNr,compression="gzip")
	GrainSizes = Grains.create_dataset('GrainSize',data=grSz,compression="gzip")
	KAM = KAMs.create_dataset('KAM',data=kam,compression="gzip")
	X = Poss.create_dataset('X',data=x,compression="gzip")
	Y = Poss.create_dataset('Y',data=y,compression="gzip")
	Z = Poss.create_dataset('Z',data=z,compression="gzip")
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
	f.write(' %s.h5:/Grains/GrainNr\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('<Attribute Name="GrainSizes" AttributeType="Scalar" Center="Cell">\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Int" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/Grains/GrainSize\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('<Attribute Name="XPositions" AttributeType="Scalar" Center="Cell">\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Int" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/Positions/X\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('<Attribute Name="YPositions" AttributeType="Scalar" Center="Cell">\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Int" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/Positions/Y\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('<Attribute Name="ZPositions" AttributeType="Scalar" Center="Cell">\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Int" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/Positions/Z\n'%(h5fn))
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
	extent = int(math.ceil(5*gridSpacing/spacing))
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

y = np.linspace(-dimarr[0]*dimarr[2]/2,(Dims[1]-1-dimarr[0]/2)*dimarr[2],Dims[1])
x = np.linspace(-dimarr[1]*dimarr[2]/2,(Dims[2]-1-dimarr[1]/2)*dimarr[2],Dims[2])
xx,yy = np.meshgrid(x,y)

for fnr in range(startnr,endnr+1):
	print('LayerNr: '+ str(fnr))
	xVals[dataNr,:,:] = xx
	yVals[dataNr,:,:] = yy
	zVals[dataNr,:,:] = dataNr*zspacing + startZ
	FileName = sampleName + 'Layer' + str(fnr) + '/' + filestem + str(fnr) + '.mic'
	#~ FileName = filestem + str(fnr) + '.mic'
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
grainSizes = np.fromfile('GrainSizes.bin',dtype=np.int32)
grainSizes = grainSizes.reshape((Dims))
KamArr = np.fromfile('KAMArr.bin',dtype=np.float64)
KamArr = KamArr.reshape((Dims))

# write files
writeHDF5File(grainIDs.astype(np.int32),Euler1.astype(np.float32),Euler2.astype(np.float32),Euler3.astype(np.float32),Confidence.astype(np.float32),PhaseNr.astype(np.float32),KamArr.astype(np.float32),grains.astype(np.int32),grainSizes.astype(np.int32),xVals.astype(np.float32),yVals.astype(np.float32),zVals.astype(np.float32),outfn+'.h5')
writeXMLXdmf(Dims,[xyspacing,xyspacing,zspacing],outfn+'.xmf',outfn,sampleName)
writeH5EBSDFile(Euler1,Euler2,Euler3,Confidence,PhaseNr,grainIDs,outfn+'.h5ebsd')
