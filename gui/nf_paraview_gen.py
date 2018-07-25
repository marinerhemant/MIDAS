#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

##### Important: The grid spacing generated would be equal to the vertical spacing.
## Issues: Check the placement direction of data according to ParaView.

##
# HDF5 file arrangement: 
#            Group			  Dataset
#		|--- GrainID       -- GrainNr
#		|--- EulerAngles  |-- EulerAngle1
#		|                 |-- EulerAngle2
#		|                 |-- EulerAngle3
#		|--- Confidence    -- ConfidenceValue
#		|--- PhaseNumber   -- PhaseNr
##

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
	GrainID = grainIDs.create_dataset('GrainNrs',data=grID)
	Euler1 = Euls.create_dataset('EulerAngle1',data=eul1)
	Euler2 = Euls.create_dataset('EulerAngle2',data=eul2)
	Euler3 = Euls.create_dataset('EulerAngle3',data=eul3)
	confidence = Conf.create_dataset('ConfidenceValue',data=conf)
	PhaseNrs = PhaseNr.create_dataset('PhaseNr',data=phNr)
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
	f.write('<Topology TopologyType="3DCoRectMesh" Dimensions="%d %d %d"></Topology>\n'%(dims[0]+1,dims[1]+1,dims[2]+1))
	f.write('<Geometry Type="ORIGIN_DXDYDZ">\n')
	f.write('<!-- Origin -->\n')
	f.write('<DataItem Dimensions="3" NumberType="Float" Format="XML">\n')
	f.write('0 0 0\n')
	f.write('</DataItem>\n')
	f.write('<!-- DXDYDZ -->\n')
	f.write('<DataItem Dimensions="3" NumberType="Float" Format="XML">\n')
	f.write('%lf %lf %lf\n'%(deltas[0],deltas[1],deltas[2]))
	f.write('</DataItem>\n')
	f.write('</Geometry>\n')
	# Data: GrainID, EulerAngles, Confidence, PhaseNr
	f.write('<Attribute Name="GrainID" AttributeType="Scalar" Center="Cell">\n')
	# ~ f.write('<DataItem Format="HDF" Dimensions="%d %d %d" NumberType="Int">\n')
	# ~ f.write('%s:/GrainID/GrainNrs\n'%(h5fn))
	f.write('<DataItem Format="Binary" Dimensions="%d %d %d" NumberType="Int">\n')
	f.write('%sIDs.bin\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('<Attribute Name="EulerAngles" AttributeType="Vector" Center="Cell">\n')
	f.write('<DataItem ItemType="Function" Dimensions="%d %d %d 3"\n'%(dims[0],dims[1],dims[2]))
	f.write('	Function=" JOIN( $0,$1,$2 ) ">\n')
	# ~ f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n')
	# ~ f.write(' %s:/EulerAngles/EulerAngle1\n'%(h5fn))
	f.write('<DataItem Format="Binary" Dimensions="%d %d %d" NumberType="Float">\n')
	f.write('%seul1.bin\n'%(h5fn))
	f.write('</DataItem>\n')
	# ~ f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n')
	# ~ f.write(' %s:/EulerAngles/EulerAngle2\n'%(h5fn))
	f.write('<DataItem Format="Binary" Dimensions="%d %d %d" NumberType="Float">\n')
	f.write('%seul2.bin\n'%(h5fn))
	f.write('</DataItem>\n')
	# ~ f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n')
	# ~ f.write(' %s:/EulerAngles/EulerAngle3\n'%(h5fn))
	f.write('<DataItem Format="Binary" Dimensions="%d %d %d" NumberType="Float">\n')
	f.write('%seul3.bin\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('<Attribute Name="Confidence" AttributeType="Scalar" Center="Cell">\n')
	# ~ f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n')
	# ~ f.write(' %s:/Confidence/ConfidenceValue\n'%(h5fn))
	f.write('<DataItem Format="Binary" Dimensions="%d %d %d" NumberType="Float">\n')
	f.write('%sconf.bin\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('<Attribute Name="PhaseNumber" AttributeType="Scalar" Center="Cell">\n')
	# ~ f.write('<DataItem Dimensions="%d %d %d" NumberType="Float" Format="HDF">\n')
	# ~ f.write(' %s:/PhaseNumber/PhaseNr\n'%(h5fn))
	f.write('<DataItem Format="Binary" Dimensions="%d %d %d" NumberType="Float">\n')
	f.write('%sphNr.bin\n'%(h5fn))
	f.write('</DataItem>\n')
	f.write('</Attribute>\n')
	f.write('</Grid>\n')
	f.write('</Domain>\n')
	f.write('</Xdmf>\n')
	# Close the file
	f.close()

def mapData(data,dims,spacing): ## Check for neighboring pixels as well and populate those
	nrRows,nrCols = data.shape
	outArr = np.zeros((dims[0],dims[1],7))
	gridSpacing = data[0,5]
	extent = int(1 + math.ceil(gridSpacing/spacing))
	outArr[:,:,6] = 10000
	for i in range(nrRows):
		xPos = data[i,3]
		yPos = data[i,4]
		xBinNr = int(xPos/spacing + dims[0]/2)
		yBinNr = int(yPos/spacing + dims[1]/2)
		xT = spacing*(xBinNr - dims[0]/2)
		yT = spacing*(yBinNr - dims[1]/2)
		distt = np.sqrt(np.power((xT-xPos),2)+np.power((yT-yPos),2))
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
				if (xBinT < 0) or (xBinT > dims[0]-1) or (yBinT < 0) or (yBinT > dims[1]-1):
					continue
				xT2 = spacing*(xBinT - dims[0]/2)
				yT2 = spacing*(yBinT - dims[1]/2)
				distt2 = np.sqrt(np.power((xT2-xPos),2)+np.power((yT2-yPos),2))
				if (outArr[xBinT,yBinT,6] > distt2):
					outArr[xBinT,yBinT,0:6] = data[i,[0,7,8,9,10,11]]
					outArr[xBinT,yBinT,6] = distt2
	return outArr

sampleName = 'ss709_AR1_nf2_Rep1'
filestem = 'MicrostructureText_Layer'
outfn = 'MicOut'
startnr = 1
endnr = 1
minConfidence = 0.1
zspacing = -2
xExtent = 1200 # Maximum Extent of xValues in microns
yExtent = 1200 # Maximum Extent of yValues
Dims = [0,0,0]
Dims[0] = int(xExtent/abs(zspacing))
Dims[1] = int(yExtent/abs(zspacing))
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
if (direction == 1):
	dataNr = 0
else:
	dataNr = endnr-startnr

for fnr in range(startnr,endnr+1):
	print 'LayerNr: '+ str(fnr)
	fn = 'ss709_AR1_nf2_R1Layer' + str(fnr) + '/' + filestem + str(fnr) + '.mic'
	micfiledata = np.genfromtxt(fn,skip_header=4)
	data = micfiledata[micfiledata[:,10] > minConfidence,:]
	t1 = time.time()
	outarr = mapData(data,[Dims[0],Dims[1]],abs(zspacing))
	print time.time() - t1
	grainIDs[:,:,dataNr] = outarr[:,:,0]
	Euler1[:,:,dataNr] = outarr[:,:,1]
	Euler2[:,:,dataNr] = outarr[:,:,2]
	Euler3[:,:,dataNr] = outarr[:,:,3]
	Confidence[:,:,dataNr] = outarr[:,:,4]
	PhaseNr[:,:,dataNr] = outarr[:,:,5]
	dataNr += direction

from PIL import Image
im = Image.fromarray(Confidence[:,:,0])
im.save("trial.tiff")
# ~ writeHDF5File(grainIDs,Euler1,Euler2,Euler3,Confidence,PhaseNr,outfn+'.h5')
writeBinaryFile(grainIDs,Euler1,Euler2,Euler3,Confidence,PhaseNr,outfn)
writeXMLXdmf(Dims,[abs(zspacing)]*3,outfn+'.xmf',outfn,sampleName)
