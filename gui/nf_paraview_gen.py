#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

## Issues: Check the placement direction of data according to ParaView.

##
# HDF5 (compressed output) file arrangement: 
#            Group			  Dataset
#		|--- GrainID       -- GrainNr            -- Nonsense value: -10,000, always uint
#		|--- EulerAngles  |-- EulerAngle1        -- Nonsense value: -10,000, range +-2pi
#		|                 |-- EulerAngle2        -- Nonsense value: -10,000, range +-2pi
#		|                 |-- EulerAngle3        -- Nonsense value: -10,000, range +-2pi
#		|--- Confidence    -- ConfidenceValue    -- Nonsense value: -10,000, range 0...1 
#		|--- PhaseNumber   -- PhaseNr			 -- Nonsense value: -10,000, range 1...n 
##

### Only modify the following arguments:
### Also look at where filename is defined to see if the arrangement is different
#### 
sampleName = 'ss709_AR1_nf2_R1'
filestem = 'MicrostructureText_Layer'
outfn = 'MicOut'
startnr = 1
endnr = 43
minConfidence = 0.1
zspacing = -2
xyspacing = 2  # X and Y spacing are equal
xExtent = 1200 # Maximum Extent of xValues in um 
			   # (this should be a bit larger than your sample diameter or edge length)
yExtent = 1200 # Maximum Extent of yValues in um
			   # (this should be a bit larger than your sample diameter or edge length)
####

import math
import sys, os
import numpy as np
import h5py
import time

def writeHDF5File(grID,eul1,eul2,eul3,conf,phNr,fileID):
	f = h5py.File(fileID,'w')
	grainIDs = f.create_group('GrainID')
	Euls = f.create_group('EulerAngles')
	Conf = f.create_group('Confidence')
	PhaseNr = f.create_group('PhaseNumber')
	GrainID = grainIDs.create_dataset('GrainNrs',data=grID,compression="gzip")
	Euler1 = Euls.create_dataset('EulerAngle1',data=eul1,compression="gzip")
	Euler2 = Euls.create_dataset('EulerAngle2',data=eul2,compression="gzip")
	Euler3 = Euls.create_dataset('EulerAngle3',data=eul3,compression="gzip")
	confidence = Conf.create_dataset('ConfidenceValue',data=conf,compression="gzip")
	PhaseNrs = PhaseNr.create_dataset('PhaseNr',data=phNr,compression="gzip")
	f.close()

def writeBinaryFile(grID,eul1,eul2,eul3,conf,phNr,fileID):
	grID.astype(np.int32).tofile(fileID+'IDs.bin')
	eul1.astype(np.float32).tofile(fileID+'eul1.bin')
	eul2.astype(np.float32).tofile(fileID+'eul2.bin')
	eul3.astype(np.float32).tofile(fileID+'eul3.bin')
	conf.astype(np.float32).tofile(fileID+'conf.bin')
	phNr.astype(np.float32).tofile(fileID+'phNr.bin')

def writeXMLXdmf(dims,deltas,fn,h5fn,sample_name):
	f = open(fn,'w')
	# Header
	f.write('<?xml version="1.0" ?>\n')
	f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
	f.write('<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">\n')
	f.write('<Information Name="%s" Value="0"/>\n'%sample_name)
	f.write('<Domain>\n')
	# Info about topology and Geometry
	f.write('<Grid Name="Structured Grid" GridType="Uniform">\n')
	f.write('<Topology TopologyType="3DCoRectMesh" Dimensions="%d %d %d">\n'%(dims[0]+1,dims[1]+1,dims[2]+1))
	f.write('</Topology>\n')
	f.write('<Geometry Type="ORIGIN_DXDYDZ">\n')
	f.write('<!-- Origin -->\n')
	f.write('<DataItem Dimensions="3" NumberType="Float" Format="XML">\n')
	f.write('%lf %lf %lf\n'%(-dims[0]/2,-dims[1]/2,-dims[2]/2*(abs(deltas[2])/deltas[2]))))
	f.write('</DataItem>\n')
	f.write('<!-- DXDYDZ -->\n')
	f.write('<DataItem Dimensions="3" NumberType="Float" Format="XML">\n')
	f.write('%lf %lf %lf\n'%(deltas[0],deltas[1],deltas[2]))
	f.write('</DataItem>\n')
	f.write('</Geometry>\n')
	# Data: GrainID, EulerAngles, Confidence, PhaseNr
	f.write('<Attribute Name="GrainID" AttributeType="Scalar" Center="Cell">\n')
	f.write('<DataItem Format="HDF" Dimensions="%d %d %d" NumberType="Int">\n'%(dims[0],dims[1],dims[2]))
	f.write('%s.h5:/GrainID/GrainNrs\n'%(h5fn))
	# ~ f.write('<DataItem Format="Binary" Dimensions="%d %d %d" NumberType="Int">\n'%(dims[0],dims[1],dims[2]))
	# ~ f.write('%sIDs.bin\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('<Attribute Name="EulerAngles" AttributeType="Vector" Center="Cell">\n')
	f.write('<DataItem ItemType="Function" Dimensions="%d %d %d 3"\n'%(dims[0],dims[1],dims[2]))
	f.write('	Function=" JOIN( $0,$1,$2 ) ">\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/EulerAngles/EulerAngle1\n'%(h5fn))
	# ~ f.write('<DataItem Format="Binary" Dimensions="%d %d %d" NumberType="Float">\n'%(dims[0],dims[1],dims[2]))
	# ~ f.write('%seul1.bin\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/EulerAngles/EulerAngle2\n'%(h5fn))
	# ~ f.write('<DataItem Format="Binary" Dimensions="%d %d %d" NumberType="Float">\n'%(dims[0],dims[1],dims[2]))
	# ~ f.write('%seul2.bin\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/EulerAngles/EulerAngle3\n'%(h5fn))
	# ~ f.write('<DataItem Format="Binary" Dimensions="%d %d %d" NumberType="Float">\n'%(dims[0],dims[1],dims[2]))
	# ~ f.write('%seul3.bin\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('<Attribute Name="Confidence" AttributeType="Scalar" Center="Cell">\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/Confidence/ConfidenceValue\n'%(h5fn))
	# ~ f.write('<DataItem Format="Binary" Dimensions="%d %d %d" NumberType="Float">\n'%(dims[0],dims[1],dims[2]))
	# ~ f.write('%sconf.bin\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('<Attribute Name="PhaseNumber" AttributeType="Scalar" Center="Cell">\n')
	f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n'%(dims[0],dims[1],dims[2]))
	f.write(' %s.h5:/PhaseNumber/PhaseNr\n'%(h5fn))
	# ~ f.write('<DataItem Format="Binary" Dimensions="%d %d %d" NumberType="Float">\n'%(dims[0],dims[1],dims[2]))
	# ~ f.write('%sphNr.bin\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('</Grid>\n')
	f.write('</Domain>\n')
	f.write('</Xdmf>\n')
	# Close the file
	f.close()

def mapData(data,dims,spacing):
	nrRows,nrCols = data.shape
	outArr = np.zeros((dims[0],dims[1],7))
	outArr.fill(-10000)
	gridSpacing = data[0,5]
	extent = int(math.ceil(gridSpacing/spacing))
	outArr[:,:,6] = 10000
	for i in range(nrRows):
		xPos = data[i,3]
		yPos = data[i,4]
		xBinNr = int(xPos/spacing + dims[0]/2)
		yBinNr = int(yPos/spacing + dims[1]/2)
		xT = spacing*(xBinNr - dims[0]/2)
		yT = spacing*(yBinNr - dims[1]/2)
		distt = math.sqrt((xT-xPos)*(xT-xPos)+(yT-yPos)*(yT-yPos))
		if (xBinNr < 0) or (xBinNr > dims[0]-1) or (yBinNr < 0) or (yBinNr > dims[1]-1):
			continue
		else:
			if (outArr[xBinNr,yBinNr,6] > distt):
				outArr[xBinNr,yBinNr,0:6] = data[i,[0,7,8,9,10,11]]
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
					outArr[xBinT,yBinT,0:6] = data[i,[0,7,8,9,10,11]]
					outArr[xBinT,yBinT,6] = distt2
	return outArr

Dims = [0,0,0]
Dims[0] = int(xExtent/abs(xyspacing))
Dims[1] = int(yExtent/abs(xyspacing))
Dims[2] = (endnr - startnr + 1) # Maximum Extent of zValues
print 'Dimensions of final array: '
print Dims
startPos = 0
grainIDs = np.zeros((Dims))
Euler1 = np.zeros((Dims))
Euler2 = np.zeros((Dims))
Euler3 = np.zeros((Dims))
Confidence = np.zeros((Dims))
PhaseNr = np.zeros((Dims))
direction = zspacing/abs(zspacing)
dataNr = 0;

for fnr in range(startnr,endnr+1):
	print 'LayerNr: '+ str(fnr)
	fn = sampleName + 'Layer' + str(fnr) + '/' + filestem + str(fnr) + '.mic'
	micfiledata = np.genfromtxt(fn,skip_header=4)
	data = micfiledata[micfiledata[:,10] > minConfidence,:]
	t1 = time.time()
	outarr = mapData(data,[Dims[0],Dims[1]],abs(xyspacing))
	print time.time() - t1
	grainIDs[:,:,dataNr] = outarr[:,:,0]
	Euler1[:,:,dataNr] = outarr[:,:,1]
	Euler2[:,:,dataNr] = outarr[:,:,2]
	Euler3[:,:,dataNr] = outarr[:,:,3]
	Confidence[:,:,dataNr] = outarr[:,:,4]
	PhaseNr[:,:,dataNr] = outarr[:,:,5]
	dataNr += direction

writeHDF5File(grainIDs.astype(np.int32),Euler1.astype(np.float32),Euler2.astype(np.float32),Euler3.astype(np.float32),Confidence.astype(np.float32),PhaseNr.astype(np.float32),outfn+'.h5')
# ~ writeBinaryFile(grainIDs,Euler1,Euler2,Euler3,Confidence,PhaseNr,outfn)
writeXMLXdmf(Dims,[xyspacing,xyspacing,zspacing],outfn+'.xmf',outfn,sampleName)
