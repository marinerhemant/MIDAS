#!/usr/bin/env python

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as Tk
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import tempfile
import tkinter.filedialog as tkFileDialog
import math
from math import sin, cos, sqrt
from numpy import linalg as LA
import subprocess
from multiprocessing.dummy import Pool
import ctypes
import sys

# Try to import midas_config from utils
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    utils_dir = os.path.join(os.path.dirname(current_dir), 'utils')
    if utils_dir not in sys.path:
        sys.path.append(utils_dir)
    import midas_config
except ImportError as e:
    print(f"Warning: Could not import midas_config: {e}")
    midas_config = None

# Helpers
deg2rad = 0.0174532925199433
rad2deg = 57.2957795130823

colors = ['r','g','b','c','m','y','r','g','b','c','m','y',
		  'r','g','b','c','m','y','r','g','b','c','m','y',
		  'r','g','b','c','m','y','r','g','b','c','m','y',
		  'r','g','b','c','m','y','r','g','b','c','m','y',
		  'r','g','b','c','m','y','r','g','b','c','m','y',
		  'r','g','b','c','m','y','r','g','b','c','m','y',
		  'r','g','b','c','m','y','r','g','b','c','m','y',
		  'r','g','b','c','m','y','r','g','b','c','m','y',
		  'r','g','b','c','m','y','r','g','b','c','m','y',
		  'r','g','b','c','m','y','r','g','b','c','m','y',
		  'r','g','b','c','m','y','r','g','b','c','m','y',
		  'r','g','b','c','m','y','r','g','b','c','m','y',
		  'r','g','b','c','m','y','r','g','b','c','m','y']

def _quit():
	root.quit()
	root.destroy()

def CalcEtaAngle(XYZ):
	alpha = rad2deg*np.arccos(np.divide(XYZ[2,:],LA.norm(XYZ[1:,:],axis = 0)))
	alpha[XYZ[1,:]>0] = -alpha[XYZ[1,:]>0]
	return alpha

def CalcEtaAngleRad(y,z):
	Rad = sqrt(y*y+z*z)
	alpha = rad2deg*math.acos(z/Rad)
	if y > 0:
		alpha = -alpha
	return [alpha,Rad]

def YZ4mREta(R,Eta):
	return [-R*sin(Eta*deg2rad),R*cos(Eta*deg2rad)]

def getfn(fstem,fnum,geNum):
	if sepfolderVar.get():
		fldr = folder + '/ge' + str(geNum) + '/'
	else:
		fldr = folder
	if geNum != -1:
		return fldr + fstem + '_' + str(fnum).zfill(padding) + '.ge' + str(geNum)
	else:
		return fldr + fstem + '_' + str(fnum).zfill(padding) + '.' + fnextvar.get()

def getImage(fn,bytesToSkip):
	print("Reading file: " + fn)
	global Header, BytesPerPixel
	Header = HeaderVar.get()
	BytesPerPixel = BytesVar.get()
	f = open(fn,'rb')
	f.seek(bytesToSkip,os.SEEK_SET)
	if BytesPerPixel == 2:
		data = np.fromfile(f,dtype=np.uint16,count=(NrPixelsY*NrPixelsZ))
	elif BytesPerPixel == 4:
		data = np.fromfile(f,dtype=np.int32,count=(NrPixelsY*NrPixelsZ))
	f.close()
	data = np.reshape(data,(NrPixelsY,NrPixelsZ))
	data = data.astype(float)
	if transpose.get() == 1:
		data = np.transpose(data)
	flip_h = hflip.get() == 1
	flip_v = vflip.get() == 1
	if flip_h and flip_v:
		data = data[::-1, ::-1].copy()
	elif flip_h:
		data = data[::-1, :].copy()
	elif flip_v:
		data = data[:, ::-1].copy()
	return data

def getImageMax(fn):
	print("Reading file: " + fn)
	global Header, BytesPerPixel
	Header = HeaderVar.get()
	BytesPerPixel = BytesVar.get()
	t1 = time.time()
	f = open(fn,'rb')
	f.seek(Header,os.SEEK_SET)
	dataMax = np.zeros(NrPixelsY*NrPixelsZ)
	nFramesToDo = nFramesMaxVar.get()
	startFrameNr = maxStartFrameNrVar.get()
	t1 = time.time()
	t = time.time()
	if midas_config and midas_config.MIDAS_BIN_DIR:
		imageMax = ctypes.CDLL(os.path.join(midas_config.MIDAS_BIN_DIR, 'imageMax.so'))
	else:
		home = os.path.expanduser("~")
		imageMax = ctypes.CDLL(home + "/opt/MIDAS/FF_HEDM/bin/imageMax.so")
	
	# Use safe temp file
	with tempfile.NamedTemporaryFile(suffix='.max', delete=False) as tf:
		imgMaxOutPath = tf.name
	# Close the file handle so the C library can open it
	
	imageMax.imageMax(fn.encode('ASCII'),Header,BytesPerPixel,NrPixelsY,NrPixelsZ,nFramesToDo,startFrameNr,imgMaxOutPath.encode('ASCII'))
	t2 = time.time()
	f = open(imgMaxOutPath,"rb")
	if BytesPerPixel == 2:
		dataMax = np.fromfile(f,dtype=np.uint16,count=(NrPixelsY*NrPixelsZ))
	elif BytesPerPixel == 4:
		dataMax = np.fromfile(f,dtype=np.int32,count=(NrPixelsY*NrPixelsZ))
	f.close()
	os.remove(imgMaxOutPath) # Clean up
	t3 = time.time()
	print("Time taken to calculate max: " + str(t2-t1))
	dataMax = np.reshape(dataMax,(NrPixelsY,NrPixelsZ))
	dataMax = dataMax.astype(float)
	if transpose.get() == 1:
		dataMax = np.transpose(dataMax)
	flip_h = hflip.get() == 1
	flip_v = vflip.get() == 1
	if flip_h and flip_v:
		dataMax = dataMax[::-1, ::-1].copy()
	elif flip_h:
		dataMax = dataMax[::-1, :].copy()
	elif flip_v:
		dataMax = dataMax[:, ::-1].copy()
	return dataMax

def getData(geNum,bytesToSkip):
	fn = getfn(fileStem,fileNumber,geNum)
	global getMax
	getMax = getMaxVar.get()
	if not getMax:
		data = getImage(fn,bytesToSkip)
	else:
		data = getImageMax(fn)
	doDark = var.get()
	if doDark == 1:
		darkfn = getfn(darkStem,darkNum,geNum)
		if nDetectors > 1:
			if dark[geNum-startDetNr] is None:
				dark[geNum-startDetNr] = getImage(darkfn,Header)
			thisdark = dark[geNum-startDetNr]
		else:
			thisdark = getImage(darkfn,Header)
		corrected = np.subtract(data,thisdark)
	else:
		corrected = data
	nonzerocoords = np.nonzero(corrected)
	return [corrected,nonzerocoords]

def getDataB(geNum,bytesToSkip):
	fn = getfn(fileStem,fileNumber,geNum)
	global getMax
	getMax = getMaxVar.get()
	if not getMax:
		data = getImage(fn,bytesToSkip)
	else:
		data = getImageMax(fn)
	doDark = var.get()
	if doDark == 1:
		darkfn = getfn(darkStem,darkNum,geNum)
		if nDetectors > 1:
			if dark[geNum-startDetNr] is None:
				dark[geNum-startDetNr] = getImage(darkfn,Header)
			thisdark = dark[geNum-startDetNr]
		else:
			thisdark = getImage(darkfn,Header)
		corrected = np.subtract(data,thisdark)
	else:
		corrected = data
	return corrected

def transforms(idx):
	txr = tx[idx]*deg2rad
	tyr = ty[idx]*deg2rad
	tzr = tz[idx]*deg2rad
	Rx = np.array([[1,0,0],[0,cos(txr),-sin(txr)],[0,sin(txr),cos(txr)]])
	Ry = np.array([[cos(tyr),0,sin(tyr)],[0,1,0],[-sin(tyr),0,cos(tyr)]])
	Rz = np.array([[cos(tzr),-sin(tzr),0],[sin(tzr),cos(tzr),0],[0,0,1]])
	return np.dot(Rx,np.dot(Ry,Rz))

def bcoord():
	numrows, numcols = bdata.shape
	def format_coord(x, y):
		col = int(x+0.5)
		row = int(y+0.5)
		bcx = float(bclocalvar1.get())
		bcy = float(bclocalvar2.get())
		[eta, rr] = CalcEtaAngleRad(-x+bcx,y-bcy)
		if col>=0 and col<numcols and row>=0 and row<numrows:
			z = bdata[row,col]
			return 'x=%1.4f, y=%1.4f, Intensity=%1.4f, RingRad(pixels)=%1.4f, Eta(degrees)=%1.4f'%(x,y,z,rr,eta)
		else:
			return 'x=%1.4f, y=%1.4f, RingRad(pixels)=%1.4f, Eta(degrees)=%1.4f'%(x,y,rr,eta)
	b.format_coord = format_coord

def acoord():
	numrows, numcols = mask2.shape
	def format_coord(x, y):
		col = int(x+0.5)
		row = int(y+0.5)
		xD = x - bigdetsize/2
		yD = y - bigdetsize/2
		[eta,R] = CalcEtaAngleRad(-xD,yD)
		if col>=0 and col<numcols and row>=0 and row<numrows:
			z = mask2[row,col]
			return 'x=%1.4f, y=%1.4f, Intensity=%1.4f, RingRad(pixels)=%1.4f, Eta(degrees)=%1.4f'%(x,y,z,R,eta)
		else:
			return 'x=%1.4f, y=%1.4f, RingRad(pixels)=%1.4f, Eta(degrees)=%1.4f'%(x,y,R,eta)
	a.format_coord = format_coord

def plotRingsOffset():
	global lines2
	global lsdlocal, bclocal
	global DisplRingInfo, refreshPlot
	lsdlocal = float(lsdlocalvar.get())
	bclocal[0] = float(bclocalvar1.get())
	bclocal[1] = float(bclocalvar2.get())
	Etas = np.linspace(-180,180,num=360)
	lines2 = []
	colornr = 0
	txtDisplay = 'Selected Rings (Increasing radius): '
	if bdata is not None:
		lims = [b.get_xlim(), b.get_ylim()]
	for idx, ringrad in enumerate(ringRads):
		Y = []
		Z = []
		txtDisplay += 'HKL:['
		for i in range(3):
			txtDisplay += str(hkls[idx][i]) + ','
		txtDisplay += '],RingNr:' + str(RingsToShow[idx]) + ',Rad[px]:' + str(int(ringrad/px)) + 'Color:' + colors[idx] + ', '
		for eta in Etas:
			ringrad2 = ringrad * (lsdlocal / lsdorig)
			tmp = YZ4mREta(ringrad2,eta)
			Y.append(tmp[0]/px + bclocal[0])
			Z.append(tmp[1]/px + bclocal[1])
		if bdata is not None:
			lines2.append(b.plot(Y,Z,color=colors[colornr]))
		colornr+= 1
	if bdata is not None and refreshPlot != 1:
		b.set_xlim([lims[0][0],lims[0][1]])
		b.set_ylim([lims[1][0],lims[1][1]])
	txtDisplay = txtDisplay[:-2]
	maxl = 270
	if len(txtDisplay) >maxl:
		tmpdisplay = ''
		nseps = int(len(txtDisplay)/maxl + 1)
		for i in range(nseps):
			tmpdisplay += txtDisplay[i*maxl:(i+1)*maxl] + '\n'
		txtDisplay = tmpdisplay[:-1]
	DisplRingInfo = Tk.Label(master=root,text=txtDisplay,justify=Tk.LEFT)
	DisplRingInfo.grid(row=figrowspan-1,column=0,columnspan=10)
	if bdata is not None and refreshPlot != 1:
		bcoord()
	if refreshPlot != 1:
		canvas.draw_idle()
		canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)

def plotRings():
	global lines
	Etas = np.linspace(-180,180,num=360)
	lines = []
	colornr = 0
	if mask2 is not None:
		lims = [a.get_xlim(), a.get_ylim()]
		for ringrad in ringRads:
			Y = []
			Z = []
			for eta in Etas:
				tmp = YZ4mREta(ringrad,eta)
				Y.append(tmp[0]/px + bigdetsize/2)
				Z.append(tmp[1]/px + bigdetsize/2)
			lines.append(a.plot(Y,Z,color=colors[colornr]))
			colornr+= 1
		a.set_xlim([lims[0][0],lims[0][1]])
		a.set_ylim([lims[1][0],lims[1][1]])
		acoord()

def doRings():
	global lines
	global lines2
	global DisplRingInfo
	plotYesNo = plotRingsVar.get()
	if lines is not None:
		for line in lines:
			line.pop(0).remove()
		lines = None
	if lines2 is not None:
		for line2 in lines2:
			line2.pop(0).remove()
		lines2 = None
	if DisplRingInfo is not None:
		DisplRingInfo.grid_forget()
		DisplRingInfo = None
	if plotYesNo == 1:
		if ringRads is None:
			ringSelection()
		else:
			plotRings()
			plotRingsOffset()
	else:
		canvas.draw_idle()
		canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)

def clickRings():
	global refreshPlot
	refreshPlot = 0
	doRings()

def plot_updater():
	global initplot
	global fileNumber
	global frameNr
	global threshold
	global lims
	global lines
	global mask2
	global Header, BytesPerPixel
	Header = HeaderVar.get()
	BytesPerPixel = BytesVar.get()
	if not initplot:
		lims = [a.get_xlim(), a.get_ylim()]
	frameNr = int(framenrvar.get())
	threshold = float(thresholdvar.get())
	upperthreshold = float(maxthresholdvar.get())
	#a.clear()
	# Plot mask if wanted
	if mask is None:
		readBigDet()
	## Go through each geNum, get the data, transform it, put it on the bigDet
	mask2 = np.copy(mask)
	fileNumber = int(firstFileNumber + frameNr/nFramesPerFile)
	framesToSkip = frameNr % nFramesPerFile
	bytesToSkip = Header + framesToSkip*(BytesPerPixel*NrPixelsY*NrPixelsZ)
	for i in range(startDetNr,endDetNr+1):
		[thresholded,(rows,cols)] = getData(i,bytesToSkip)
		TRs = transforms(i-startDetNr)
		Xc = np.zeros(rows.shape)
		Yc = -(cols - bcs[i-startDetNr][0])*px
		Zc = (rows - bcs[i-startDetNr][1])*px
		ABC = np.array([Xc,Yc,Zc])
		ABCPr = np.dot(TRs,ABC)
		NewYs = ABCPr[1,:]/px
		NewZs = ABCPr[2,:]/px
		NewYs = NewYs.astype(int)
		NewZs = NewZs.astype(int)
		mask2[bigdetsize/2 - NewZs,bigdetsize/2 - NewYs] = thresholded[rows,cols]
	#lines = None
	doRings()
	global _a_artist, _a_logmode
	use_log = dolog.get() != 0
	if use_log:
		if threshold == 0:
			threshold = 1
		if upperthreshold == 0:
			upperthreshold = 1
		mask3 = np.copy(mask2)
		mask3 [ mask3 == 1 ] = 10
		mask3 [ mask3 == 0 ] = 1
		display_data = np.log(mask3)
		clim = (np.log(threshold),np.log(upperthreshold))
	else:
		display_data = mask2
		clim = (threshold,upperthreshold)
	can_reuse = (not initplot and _a_artist is not None and
	             _a_logmode == use_log and
	             _a_artist.get_array().shape == display_data.shape)
	if can_reuse:
		_a_artist.set_data(display_data)
		_a_artist.set_clim(*clim)
		a.set_xlim([lims[0][0],lims[0][1]])
		a.set_ylim([lims[1][0],lims[1][1]])
	else:
		a.clear()
		_a_artist = a.imshow(display_data,cmap=plt.get_cmap('bone'),interpolation='nearest',clim=clim)
		_a_logmode = use_log
		if initplot:
			initplot = 0
		else:
			a.set_xlim([lims[0][0],lims[0][1]])
			a.set_ylim([lims[1][0],lims[1][1]])
	acoord()
	a.title.set_text("Multiple Detector Display")
	canvas.draw_idle()
	canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)

def incr_plotupdater():
	global frameNr
	global framenrvar
	frameNr = int(framenrvar.get())
	frameNr += 1
	framenrvar.set(str(frameNr))
	global getMax
	getMax = getMaxVar.get()
	if getMax:
		return
	if nDetectors > 1:
		plot_updater()
	else:
		loadbplot()

def decr_plotupdater():
	global frameNr
	global framenrvar
	frameNr = int(framenrvar.get())
	frameNr -= 1
	framenrvar.set(str(frameNr))
	global getMax
	getMax = getMaxVar.get()
	if getMax:
		return
	if nDetectors > 1:
		plot_updater()
	else:
		loadbplot()

def readParams():
	global paramFN
	paramFN = paramfilevar.get()
	global folder, fileStem, padding, startDetNr, endDetNr, bigFN
	global wedge, lsd, px, bcs, tx, wl, bigdetsize, nFramesPerFile
	global firstFileNumber, darkStem, darkNum, omegaStep, nFilesPerLayer
	global omegaStart, NrPixelsY, NrPixelsZ, threshold, RingsToShow, nDetectors
	global RhoDs, LatC, sg, maxRad, border, ringslines, lsdline, hkls
	global ty, tz, p0, p1, p2, fileNumber, dark, ringRads, ringNrs, lsdlocal
	global bclocal, lsdlocalvar, WidthTTh, tolTilts, tolBC, tolLsd, tolP
	paramContents = open(paramFN,'r').readlines()
	lsd = []
	bcs = []
	tx = []
	ty = []
	tz = []
	p0 = []
	p1 = []
	p2 = []
	RingsToShow = []
	threshold = 0
	RhoDs = []
	ringslines = []
	hkls = []
	lsdline = None
	for line in paramContents:
		if line == '\n':
			continue
		if line[0] == '#':
			continue
		if 'RingThresh' == line.split()[0]:
			ringslines.append(line)
			RingsToShow.append(int(line.split()[1]))
			threshold = max(threshold,float(line.split()[2]))
		if 'tolTilts' == line.split()[0]:
			tolTilts = line.split()[1]
		if 'tolBC' == line.split()[0]:
			tolBC = line.split()[1]
		if 'tolLsd' == line.split()[0]:
			tolLsd = line.split()[1]
		if 'tolP' == line.split()[0]:
			tolP = line.split()[1]
		if 'RawFolder' == line.split()[0]:
			folder = line.split()[1]
		if 'FileStem' == line.split()[0]:
			fileStem = line.split()[1]
		if 'Padding' == line.split()[0]:
			padding = int(line.split()[1])
		if 'StartDetNr' == line.split()[0]:
			startDetNr = int(line.split()[1])
		if 'EndDetNr' == line.split()[0]:
			endDetNr = int(line.split()[1])
		if 'Wedge' == line.split()[0]:
			wedge = float(line.split()[1])
		if 'px' == line.split()[0]:
			px = float(line.split()[1])
		if 'Wavelength' == line.split()[0]:
			wl = float(line.split()[1])
		if 'BigDetSize' == line.split()[0]:
			bigdetsize = int(line.split()[1])
		if 'nFramesPerFile' == line.split()[0]:
			nFramesPerFile = int(line.split()[1])
		if 'FirstFileNumber' == line.split()[0]:
			firstFileNumber = int(line.split()[1])
			fileNumber = firstFileNumber
		if 'StartFileNrFirstLayer' == line.split()[0]:
			firstFileNumber = int(line.split()[1])
			fileNumber = firstFileNumber
		if 'DarkStem' == line.split()[0]:
			darkStem = line.split()[1]
		if 'LatticeParameter' == line.split()[0]:
			LatC = line
		if 'LatticeConstant' == line.split()[0]:
			LatC = line
		if 'Lsd' == line.split()[0]:
			lsdline = line
		if 'MaxRingRad' == line.split()[0]:
			maxRad = line
		if 'BorderToExclude' == line.split()[0]:
			border = line
		if 'DarkNum' == line.split()[0]:
			darkNum = int(line.split()[1])
		if 'SpaceGroup' == line.split()[0]:
			sg = int(line.split()[1])
		if 'OmegaStep' == line.split()[0]:
			omegaStep = float(line.split()[1])
		if 'OmegaFirstFile' == line.split()[0]:
			omegaStart = float(line.split()[1])
		if 'NrFilesPerSweep' == line.split()[0]:
			nFilesPerLayer = int(line.split()[1])
		if 'NrPixelsY' == line.split()[0]:
			NrPixelsY = int(line.split()[1])
		if 'NrPixelsZ' == line.split()[0]:
			NrPixelsZ = int(line.split()[1])
		if 'NumDetectors' == line.split()[0]:
			nDetectors = int(line.split()[1])
		if 'Lsd' == line.split()[0]:
			lsd.append(float(line.split()[1]))
		if 'tx' == line.split()[0]:
			tx.append(float(line.split()[1]))
		if 'Width' == line.split()[0]:
			WidthTTh = line.split()[1]
		if 'DetParams' == line.split()[0]:
			lsd.append(float(line.split()[1]))
			bcs.append([float(line.split()[2]),float(line.split()[3])])
			tx.append(float(line.split()[4]))
			ty.append(float(line.split()[5]))
			tz.append(float(line.split()[6]))
			p0.append(float(line.split()[7]))
			p1.append(float(line.split()[8]))
			p2.append(float(line.split()[9]))
			RhoDs.append(float(line.split()[10]))
	if folder[0] == '~':
		folder = os.path.expanduser(folder)
	bigFN = 'BigDetectorMaskEdgeSize' + str(bigdetsize) + 'x' + str(bigdetsize) + 'Unsigned16Bit.bin'
	if midas_config and midas_config.MIDAS_BIN_DIR:
		hklGenPath = os.path.join(midas_config.MIDAS_BIN_DIR, 'GetHKLList')
	else:
		hklGenPath = os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/GetHKLList')
	
	subprocess.run([hklGenPath, paramFN], check=True)
	hklfn = 'hkls.csv'
	hklfile = open(hklfn,'r')
	hklfile.readline()
	hklinfo = hklfile.readlines()
	hklfile.close()
	ringRads = []
	ringNrs = []
	lsdlocal = lsd[0]
	lsdlocalvar.set(str(lsdlocal))
	bclocal[0] = bcs[0][0]
	bclocal[1] = bcs[0][1]
	bclocalvar1.set(str(bclocal[0]))
	bclocalvar2.set(str(bclocal[1]))
	for ringNr in RingsToShow:
		for line in hklinfo:
			if int(line.split()[4]) == ringNr:
				ringRads.append(float(line.split()[-1].split('\n')[0]))
				ringNrs.append(ringNr)
				hkls.append([int(line.split()[0]),int(line.split()[1]),int(line.split()[2])])
				break
	# initialization of dark
	dark = []
	for i in range(nDetectors):
		dark.append(None)

def writeCalibrateParams(pfname,detNum,ringsToExclude):
	f = open(pfname,'w')
	f.write('Folder '+ folder+'\n')
	f.write('FileStem ' + fileStem+'\n')
	f.write('Dark ' + getfn(darkStem,darkNum,detNum)+'\n')
	f.write('Padding '+str(padding)+'\n')
	f.write('Ext .ge'+str(detNum)+'\n')
	f.write('ImTransOpt 0\n')
	f.write('BC '+str(bcs[detNum-startDetNr][0])+' '+str(bcs[detNum-startDetNr][1])+'\n')
	f.write('px '+str(px)+'\n')
	f.write('Width '+WidthTTh+'\n')
	f.write('LatticeParameter 5.411651 5.411651 5.411651 90 90 90\nSpaceGroup 225\n')
	f.write('NrPixelsY '+str(NrPixelsY)+'\n')
	f.write('NrPixelsZ '+str(NrPixelsZ)+'\n')
	f.write('Wavelength '+str(wl)+'\n')
	f.write('Lsd '+str(lsd[detNum-startDetNr])+'\n')
	f.write('RhoD '+ str(RhoDs[detNum-startDetNr])+'\n')
	f.write('StartNr ' + str(firstFileNumber)+'\n')
	f.write('EndNr ' + str(firstFileNumber+nFilesPerLayer-1)+'\n')
	f.write('tolTilts '+tolTilts+'\ntolBC '+tolBC+'\ntolLsd '+tolLsd+'\ntolP '+tolP+'\n')
	f.write('p0 '+str(p0[detNum-startDetNr])+'\np1 '+str(p1[detNum-startDetNr])+'\np2 '+str(p2[detNum-startDetNr])+'\nEtaBinSize 5\n')
	f.write('ty '+str(ty[detNum-startDetNr])+'\ntz '+str(tz[detNum-startDetNr])+'\nWedge 0\n')
	f.write('tx '+str(tx[detNum-startDetNr])+'\n')
	if len(ringsToExclude) > 0:
		for ring in ringsToExclude:
			f.write('RingsToExclude '+str(ring)+'\n')

def writeParams():
	pfname = os.getcwd() + '/GeneratedParameters.txt'
	f = open(pfname,'w')
	f.write('NumDetectors '+str(nDetectors)+'\n')
	f.write('RawFolder '+ folder+'\n')
	f.write('FileStem ' + fileStem+'\n')
	f.write('Padding '+str(padding)+'\n')
	f.write('StartDetNr '+str(startDetNr)+'\n')
	f.write('EndDetNr '+str(endDetNr)+'\n')
	f.write('Wedge '+str(wedge)+'\n')
	sep = ' '
	for i in range(nDetectors):
		strout = 'DetParams '+ str(lsd[i]) + sep + str(bcs[i][0]) + sep + str(bcs[i][1]) + sep + str(tx[i]) + sep + str(ty[i]) + sep + str(tz[i]) + sep + str(p0[i]) + sep + str(p1[i]) + sep + str(p2[i]) + sep + str(RhoDs[i]) + '\n'
		f.write(strout)
	f.write(LatC)
	if lsdline is not None:
		f.write(lsdline)
	f.write('SpaceGroup '+str(sg)+'\n')
	f.write('Wavelength '+str(wl)+'\n')
	f.write(maxRad)
	f.write('NrPixelsY '+str(NrPixelsY)+'\n')
	f.write('NrPixelsZ '+str(NrPixelsZ)+'\n')
	f.write('BigDetSize '+str(bigdetsize)+'\n')
	f.write(border)
	f.write('px '+str(px)+'\n')
	f.write('nFramesPerFile '+str(nFramesPerFile)+'\n')
	f.write('FirstFileNumber '+str(firstFileNumber)+'\n')
	f.write('DarkStem '+darkStem+'\n')
	f.write('DarkNum '+str(darkNum)+'\n')
	f.write('OmegaStep '+str(omegaStep)+'\n')
	f.write('OmegaFirstFile '+str(omegaStart)+'\n')
	f.write('NrFilesPerSweep '+str(nFilesPerLayer)+'\n')
	for line in ringslines:
		f.write(line)
	topWrite = Tk.Toplevel()
	Tk.Label(topWrite,text='File written to '+pfname).grid(row=1)
	Tk.Button(master=topWrite,text="Close",command=topWrite.destroy).grid(row=2)

def redoCalibration():
	global topCalibrate
	topCalibrate.destroy()
	askRingsToExclude()

def parseOutputs(outputs):
	global topCalibrate
	global ty, tz, p0, p1, p2, meanStrain, stdStrain
	ty = []
	tz = []
	p0 = []
	p1 = []
	p2 = []
	meanStrain = []
	stdStrain = []
	for i in range(nDetectors):
		lsdtemp = 0
		ybctemp = 0
		zbctemp = 0
		tytemp = 0
		tztemp = 0
		p0temp = 0
		p1temp = 0
		p2temp = 0
		meanstrtemp = 0
		stdstrtemp = 0
		output = outputs[i]
		fileWrite = open('DetectorCalibrationOutputDetNr'+str(i)+'.txt','w')
		for line in output:
			fileWrite.write(line+'\n')
			if 'LsdFit' in line:
				lsdtemp += float(line.split('\t')[-1])/nFilesPerLayer
			if 'YBCFit' in line:
				ybctemp += float(line.split('\t')[-1])/nFilesPerLayer
			if 'ZBCFit' in line:
				zbctemp += float(line.split('\t')[-1])/nFilesPerLayer
			if 'tyFit' in line:
				tytemp += float(line.split('\t')[-1])/nFilesPerLayer
			if 'tzFit' in line:
				tztemp += float(line.split('\t')[-1])/nFilesPerLayer
			if 'P0Fit' in line:
				p0temp += float(line.split('\t')[-1])/nFilesPerLayer
			if 'P1Fit' in line:
				p1temp += float(line.split('\t')[-1])/nFilesPerLayer
			if 'P2Fit' in line:
				p2temp += float(line.split('\t')[-1])/nFilesPerLayer
			if 'MeanStrain' in line:
				meanstrtemp += float(line.split('\t')[-1])/nFilesPerLayer
			if 'StdStrain' in line:
				stdstrtemp += float(line.split('\t')[-1])/nFilesPerLayer
		fileWrite.close()
		lsd[i] = lsdtemp
		bcs[i][0] = ybctemp
		bcs[i][1] = zbctemp
		ty.append(tytemp)
		tz.append(tztemp)
		p0.append(p0temp)
		p1.append(p1temp)
		p2.append(p2temp)
		meanStrain.append(meanstrtemp)
		stdStrain.append(stdstrtemp)
	# Display new values on screen, ask whether to run again?
	Tk.Label(topCalibrate,text="The refined values are:").grid(row=1)
	for i in range(nDetectors):
		strOut="For detector %d, Lsd: %lf, YBC: %lf, ZBC: %lf, ty: %lf, tz: %lf, Ps: %lf %lf %lf, MeanStrain: %lf, StdStrain: %lf"%(startDetNr+i,
			lsd[i],bcs[i][0],bcs[i][1],ty[i],tz[i],p0[i],p1[i],p2[i],meanStrain[i],stdStrain[i])
		Tk.Label(topCalibrate,text=strOut).grid(row=2+i)
	Tk.Label(topCalibrate,text="Do you want to run calibration again with these parameters?").grid(row=nDetectors+2)
	Tk.Button(master=topCalibrate,text='Yes',command=redoCalibration).grid(row=nDetectors+3)
	Tk.Button(master=topCalibrate,text='No',command=topCalibrate.destroy).grid(row=nDetectors+4)

def calibrateDetector():
	global ringsexcludevar
	global eringsexclude, buttonConfirmRingsExclude, ringsexcludelabel
	eringsexclude.grid_forget()
	buttonConfirmRingsExclude.grid_forget()
	ringsexcludelabel.grid_forget()
	ringsexcludestr = ringsexcludevar.get()
	if midas_config and midas_config.MIDAS_BIN_DIR:
		calibratecmd = os.path.join(midas_config.MIDAS_BIN_DIR, 'Calibrant')
	else:
		calibratecmd = os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/Calibrant')
	
	if ringsexcludestr == '0':
		ringsToExclude = []
	else:
		ringsToExclude = [int(rings) for rings in ringsexcludestr.split(',')]
	
	pfnames = []
	for i in range(startDetNr,endDetNr+1):
		pfnames.append('CalibrationDetNr' + str(i) + '.txt')
		writeCalibrateParams(pfnames[-1],i,ringsToExclude)
	
	# Safe Popen usage with list arguments
	cmds = [[calibratecmd, pfname] for pfname in pfnames]
	processes = [subprocess.Popen(cmd,
				stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True) for cmd in cmds]
	
	def get_lines(process):
		return process.communicate()[0].decode('utf-8').splitlines()
	
	outputs = Pool(len(processes)).map(get_lines,processes)
	parseOutputs(outputs)

def askRingsToExclude():
	global topCalibrate
	global ringsexcludevar
	global eringsexclude, buttonConfirmRingsExclude, ringsexcludelabel
	topCalibrate = Tk.Toplevel()
	topCalibrate.title("Select rings to exclude from Calibration")
	ringsexcludelabel = Tk.Label(master=topCalibrate,text="Please enter the rings you would like to exclude, each separated by a comma, no spaces please")
	ringsexcludelabel.grid(row=1)
	ringsexcludevar = Tk.StringVar()
	ringsexcludevar.set(str(0))
	eringsexclude = Tk.Entry(topCalibrate,textvariable=ringsexcludevar)
	eringsexclude.grid(row=2)
	buttonConfirmRingsExclude = Tk.Button(master=topCalibrate,text="Confirm",command=calibrateDetector)
	buttonConfirmRingsExclude.grid(row=3)

def paramfileselect():
	global paramFN
	global paramfilevar
	paramFN = tkFileDialog.askopenfilename()
	paramfilevar.set(paramFN)

def readBigDet():
	global mask
	bigf = open(bigFN,'rb')
	mask = np.fromfile(bigf,dtype=np.uint16,count=bigdetsize*bigdetsize)
	bigf.close()
	mask = np.reshape(mask,(bigdetsize,bigdetsize))
	mask = mask.astype(float)
	mask = mask[::-1, ::-1].copy()

def makeBigDet():
	if midas_config and midas_config.MIDAS_BIN_DIR:
		cmdf = os.path.join(midas_config.MIDAS_BIN_DIR, 'MapMultipleDetectors')
	else:
		cmdf = os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/MapMultipleDetectors')
	
	subprocess.run([cmdf, paramFN], check=True)
	readBigDet()

def loadbplot():
	global bclocal
	global lsdlocal
	global lsdorig
	global initplot2
	global origdetnum
	global bclocalvar1, bclocalvar2
	global ax
	global fileNumber, refreshPlot
	global bdata
	global lines2, NrPixelsY, NrPixelsZ
	global firstFileNumber, nFramesPerFile
	global Header, BytesPerPixel
	Header = HeaderVar.get()
	BytesPerPixel = BytesVar.get()
	if not initplot2:
		lims = [b.get_xlim(), b.get_ylim()]
	frameNr = int(framenrvar.get())
	threshold = float(thresholdvar.get())
	upperthreshold = float(maxthresholdvar.get())
	NrPixelsY = int(NrPixelsYVar.get())
	NrPixelsZ = int(NrPixelsZVar.get())
	firstFileNumber = int(firstFileNrVar.get())
	nFramesPerFile = int(nFramesPerFileVar.get())
	fileNumber = int(firstFileNumber + frameNr/nFramesPerFile)
	framesToSkip = int(frameNr % nFramesPerFile)
	bytesToSkip = Header + framesToSkip*(BytesPerPixel*NrPixelsY*NrPixelsZ)
	detnr = int(detnumbvar.get())
	if detnr != -1:
		if (detnr != origdetnum or initplot2) and bcs is not None:
			origdetnum = detnr
			bclocal[0] = bcs[detnr-startDetNr][0]
			bclocal[1] = bcs[detnr-startDetNr][1]
			bclocalvar1.set(str(bclocal[0]))
			bclocalvar2.set(str(bclocal[1]))
		else:
			bclocal[0] = float(bclocalvar1.get())
			bclocal[1] = float(bclocalvar2.get())
	else:
		bclocal[0] = float(bclocalvar1.get())
		bclocal[1] = float(bclocalvar2.get())
	bdata = getDataB(detnr,bytesToSkip)
	if nDetectors > 1:
		lsdorig = lsd[detnr-startDetNr]
	# ~ else:
		# ~ lsdorig = float(lsdlocalvar.get())
	lsdlocal = float(lsdlocalvar.get())
	#lines2 = None
	#b.clear()
	refreshPlot = 1
	doRings()
	global _b_artist, _b_logmode
	use_log = dolog.get() != 0
	if use_log:
		if threshold == 0:
			threshold = 1
		if upperthreshold == 0:
			upperthreshold = 1
		mask3 = np.copy(bdata)
		mask3 [ mask3 == 0 ] = 1
		display_data = np.log(mask3)
		clim = (np.log(threshold),np.log(upperthreshold))
	else:
		display_data = bdata
		clim = (threshold,upperthreshold)
	can_reuse = (not initplot2 and _b_artist is not None and
	             _b_logmode == use_log and
	             _b_artist.get_array().shape == display_data.shape)
	if can_reuse:
		_b_artist.set_data(display_data)
		_b_artist.set_clim(*clim)
		b.set_xlim([lims[0][0],lims[0][1]])
		b.set_ylim([lims[1][0],lims[1][1]])
	else:
		b.clear()
		_b_artist = b.imshow(display_data,cmap=plt.get_cmap('bone'),interpolation='nearest',clim=clim)
		_b_logmode = use_log
		if initplot2:
			initplot2 = 0
			b.invert_yaxis()
		else:
			b.set_xlim([lims[0][0],lims[0][1]])
			b.set_ylim([lims[1][0],lims[1][1]])
	bcoord()
	b.title.set_text("Single Detector Display")
	canvas.draw_idle()
	canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)

def acceptRings():
	global RingsToShow
	global ringRads
	global topSelectRings
	global hkls
	global plotRingsVar
	items = ListBox1.curselection()
	ringRads = [RingRad[int(item)] for item in items]
	RingsToShow = [int(item)+1 for item in items]
	hkls = [hkl[int(item)] for item in items]
	topSelectRings.destroy()
	plotRingsVar.set(1)
	doRings()
	#plotRings()
	#plotRingsOffset()

def selectRings():
	global topSelectRings
	global hklLines, hkl, ds, Ttheta, RingRad, ListBox1
def selectRings():
	global topSelectRings
	global hklLines, hkl, ds, Ttheta, RingRad, ListBox1
	
	hklinfo = []
	header = ""
	
	# Use TemporaryDirectory to contain output files safely
	with tempfile.TemporaryDirectory() as temp_dir:
		if midas_config and midas_config.MIDAS_BIN_DIR:
			hklGenPath = os.path.join(midas_config.MIDAS_BIN_DIR, 'GetHKLList')
		else:
			hklGenPath = os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/GetHKLList')
		
		pfname = os.path.join(temp_dir, 'ps_midas_ff.txt')
		with open(pfname, 'w') as f:
			f.write('Wavelength ' + str(wl) + '\n')
			f.write('SpaceGroup ' + str(sg) + '\n')
			f.write('Lsd ' + str(tempLsd) + '\n')
			f.write('MaxRingRad ' + str(tempMaxRingRad) + '\n')
			f.write('LatticeConstant ')
			for i in range(6):
				f.write(str(LatticeConstant[i]) + ' ')
			f.write('\n')
		
		# Run GetHKLList in temp_dir
		subprocess.run([hklGenPath, pfname], check=True, cwd=temp_dir)
		
		hklfn = os.path.join(temp_dir, 'hkls.csv')
		if os.path.exists(hklfn):
			with open(hklfn, 'r') as hklfile:
				header = hklfile.readline()
				header = header.replace(' ','      ')
				hklinfo = hklfile.readlines()
		else:
			print("Error: hkls.csv not found.")
			return
	maxRingNr = 101
	hkl = []
	ds = []
	Ttheta = []
	RingRad = []
	hklLines = []
	for ringNr in range(1,maxRingNr):
		for line in hklinfo:
			if int(line.split()[4]) == ringNr:
				hkl.append([int(line.split()[0]),int(line.split()[1]),int(line.split()[2])])
				ds.append(float(line.split()[3]))
				Ttheta.append(float(line.split()[9]))
				RingRad.append(float(line.split()[10].split('\n')[0]))
				hklLines.append(line.split('\n')[0])
				break
	topSelectRings = Tk.Toplevel()
	topSelectRings.title('Select Rings')
	nrhkls = len(hklLines)
	Tk.Label(master=topSelectRings,text=header.split('\n')[0]).grid(row=0,column=0,sticky=Tk.W,columnspan=2)
	ListBox1 = Tk.Listbox(topSelectRings,width=80,height=15,selectmode=Tk.EXTENDED)
	ListBox1.grid(row=1,column=0)
	yscroll=Tk.Scrollbar(topSelectRings)
	yscroll.grid(row=1,column=1,sticky=Tk.N+Tk.S)
	for line in hklLines:
		ListBox1.insert(Tk.END,line)
	ListBox1.config(yscrollcommand=yscroll.set)
	yscroll.config(command=ListBox1.yview)
	Tk.Button(master=topSelectRings,text='Done',command=acceptRings).grid(row=2,column=0,columnspan=2)

def acceptSgWlLatC():
	global wl, sg, LatticeConstant, tempLsd, tempMaxRingRad, px, bigdetsize
	global topRingMaterialSelection, lsdlocal, lsdorig
	wl = float(wlVar.get())
	if wl > 1:
		wl = 12.398/wl
	sg = int(sgVar.get())
	px = float(pxVar.get())
	tempLsd = float(tempLsdVar.get())
	for i in range(4):
		if lsd[i] == 0:
			lsd[i] = tempLsd
	lsdlocal = tempLsd
	lsdorig = tempLsd
	lsdlocalvar.set(str(tempLsd))
	tempMaxRingRad = float(tempMaxRingRadVar.get())
	for i in range(6):
		LatticeConstant[i] = float(LatticeConstantVar[i].get())
	topRingMaterialSelection.destroy()
	selectRings()

def ringSelection():
	global wlVar, sgVar, LatticeConstantVar, tempLsdVar, tempMaxRingRadVar, pxVar
	global topRingMaterialSelection, refreshPlot
	wlVar = Tk.StringVar()
	sgVar = Tk.StringVar()
	pxVar = Tk.StringVar()
	tempLsdVar = Tk.StringVar()
	tempMaxRingRadVar = Tk.StringVar()
	sgVar.set(str(sg))
	wlVar.set(str(wl))
	pxVar.set(str(px))
	tempLsdVar.set(lsdlocalvar.get())
	tempMaxRingRadVar.set(str(tempMaxRingRad))
	LatticeConstantVar = [Tk.StringVar(),Tk.StringVar(),Tk.StringVar(),Tk.StringVar(),Tk.StringVar(),Tk.StringVar()]
	for i in range(6):
		LatticeConstantVar[i].set(str(LatticeConstant[i]))
	topRingMaterialSelection = Tk.Toplevel()
	topRingMaterialSelection.title('Select the SpaceGroup, Wavelength(or Energy), Lattice Constant')
	Tk.Label(master=topRingMaterialSelection,text='Please enter the SpaceGroup, Wavelength(or Energy), Lattice Constant, Sample To Detector Distance(Lsd)').grid(row=1,column=1,columnspan=7)
	Tk.Label(master=topRingMaterialSelection,text='SpaceGroup').grid(row=2,column=1,sticky=Tk.W)
	Tk.Entry(master=topRingMaterialSelection,textvariable=sgVar,width=4).grid(row=2,column=2,sticky=Tk.W)
	Tk.Label(master=topRingMaterialSelection,text='Wavelength (A) or Energy (KeV)').grid(row=3,column=1,sticky=Tk.W)
	Tk.Entry(master=topRingMaterialSelection,textvariable=wlVar,width=8).grid(row=3,column=2,sticky=Tk.W)
	Tk.Label(master=topRingMaterialSelection,text='LatticeConstant (A)').grid(row=4,column=1,sticky=Tk.W)
	for i in range(6):
		Tk.Entry(master=topRingMaterialSelection,textvariable=LatticeConstantVar[i],width=8).grid(row=4,column=i+2,sticky=Tk.W)
	Tk.Label(master=topRingMaterialSelection,text='Lsd (um)').grid(row=5,column=1,sticky=Tk.W)
	Tk.Entry(master=topRingMaterialSelection,textvariable=tempLsdVar,width=8).grid(row=5,column=2,sticky=Tk.W)
	Tk.Label(master=topRingMaterialSelection,text='MaxRingRad (um)').grid(row=6,column=1,sticky=Tk.W)
	Tk.Entry(master=topRingMaterialSelection,textvariable=tempMaxRingRadVar,width=8).grid(row=6,column=2,sticky=Tk.W)
	Tk.Label(master=topRingMaterialSelection,text='Pixel Size (um)').grid(row=7,column=1,sticky=Tk.W)
	Tk.Entry(master=topRingMaterialSelection,textvariable=pxVar,width=8).grid(row=7,column=2,sticky=Tk.W)
	Tk.Button(master=topRingMaterialSelection,text='Continue',command=acceptSgWlLatC).grid(row=8,column=1,columnspan=7)
	refreshPlot = 0

def selectFile():
	return tkFileDialog.askopenfilename()

def firstFileSelector():
	global fileStem, folder, padding,firstFileNumber,nFramesPerFile
	global nDetectors, detnumbvar,nFramesMaxVar,nFramesPerFileVar
	global NrPixelsY,NrPixelsZ
	global Header, BytesPerPixel
	global fnextvar
	Header = HeaderVar.get()
	BytesPerPixel = BytesVar.get()
	NrPixelsY = int(NrPixelsYVar.get())
	NrPixelsZ = int(NrPixelsZVar.get())
	firstfilefullpath = selectFile()
	fullfilename = firstfilefullpath.split('/')[-1].split('.')[0]
	fileStem = '_'.join(fullfilename.split('_')[:-1])
	firstFileNumber = int(fullfilename.split('_')[-1])
	firstFileNrVar.set(firstFileNumber)
	padding = len(fullfilename.split('_')[-1])
	nDetectors = 1
	# Check here for the extension of the detector, if it contains .geX, then set detnumbvar as X, else set fnextvar
	tempext = (firstfilefullpath.split(fullfilename)[-1])[1:]
	if len(tempext) == 3 and tempext[-1].isdigit():
		detnumbvar.set(tempext[-1])
	else:
		detnumbvar.set('-1')
		fnextvar.set(tempext)
	folder = os.path.dirname(firstfilefullpath) + '/'
	statinfo = os.stat(firstfilefullpath)
	nFramesPerFile = int((statinfo.st_size - Header)/(BytesPerPixel*NrPixelsY*NrPixelsZ))
	nFramesPerFileVar.set(nFramesPerFile)
	nFramesMaxVar.set(nFramesPerFile)

def darkFileSelector():
	global darkStem,darkNum, dark
	darkfilefullpath = selectFile()
	darkfullfilename = darkfilefullpath.split('/')[-1].split('.')[0]
	darkStem = '_'.join(darkfullfilename.split('_')[:-1])
	darkNum = int(darkfullfilename.split('_')[-1])
	geNum = int(darkfilefullpath[-1])
	dark = []
	var.set(1)
	startDetNr = 1
	for i in range(geNum):
		dark.append(None)

def replot():
	global initplot, initplot2
	global lines2
	global lines
	global mask2, bdata, refreshPlot
	global _a_artist, _b_artist, _a_logmode, _b_logmode
	use_log = dolog.get() != 0
	threshold = float(thresholdvar.get())
	upperthreshold = float(maxthresholdvar.get())
	if mask2 is not None:
		if not initplot:
			lims = [a.get_xlim(), a.get_ylim()]
		if use_log:
			if threshold == 0:
				threshold = 1
			if upperthreshold == 0:
				upperthreshold = 1
			mask3 = np.copy(mask2)
			mask3 [ mask3 == 1 ] = 10
			mask3 [ mask3 == 0 ] = 1
			display_data = np.log(mask3)
			clim = (np.log(threshold),np.log(upperthreshold))
		else:
			display_data = mask2
			clim = (threshold,upperthreshold)
		can_reuse = (not initplot and _a_artist is not None and
		             _a_logmode == use_log and
		             _a_artist.get_array().shape == display_data.shape)
		if can_reuse:
			_a_artist.set_data(display_data)
			_a_artist.set_clim(*clim)
			a.set_xlim([lims[0][0],lims[0][1]])
			a.set_ylim([lims[1][0],lims[1][1]])
		else:
			a.clear()
			_a_artist = a.imshow(display_data,cmap=plt.get_cmap('bone'),interpolation='nearest',clim=clim)
			_a_logmode = use_log
			if initplot:
				initplot = 0
			else:
				a.set_xlim([lims[0][0],lims[0][1]])
				a.set_ylim([lims[1][0],lims[1][1]])
		acoord()
		a.title.set_text("Multiple Detector Display")
		canvas.draw_idle()
		canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)
	if bdata is not None:
		if not initplot2:
			lims = [b.get_xlim(), b.get_ylim()]
		if use_log:
			if threshold == 0:
				threshold = 1
			if upperthreshold == 0:
				upperthreshold = 1
			mask3 = np.copy(bdata)
			mask3 [ mask3 == 0 ] = 1
			display_data = np.log(mask3)
			clim = (np.log(threshold),np.log(upperthreshold))
		else:
			display_data = bdata
			clim = (threshold,upperthreshold)
		can_reuse = (not initplot2 and _b_artist is not None and
		             _b_logmode == use_log and
		             _b_artist.get_array().shape == display_data.shape)
		if can_reuse:
			_b_artist.set_data(display_data)
			_b_artist.set_clim(*clim)
			b.set_xlim([lims[0][0],lims[0][1]])
			b.set_ylim([lims[1][0],lims[1][1]])
		else:
			b.clear()
			_b_artist = b.imshow(display_data,cmap=plt.get_cmap('bone'),interpolation='nearest',clim=clim)
			_b_logmode = use_log
			if initplot2:
				initplot2 = 0
				b.invert_yaxis()
			else:
				b.set_xlim([lims[0][0],lims[0][1]])
				b.set_ylim([lims[1][0],lims[1][1]])
		bcoord()
		b.title.set_text("Single Detector Display")
		refreshPlot = 1
	doRings()
	canvas.draw_idle()
	canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)

# Main function
root = Tk.Tk()
root.wm_title("FF display v0.2 Dt. 2024/02/10 hsharma@anl.gov")
figur = Figure(figsize=(15,6),dpi=100)
canvas = FigureCanvasTkAgg(figur,master=root)
a = figur.add_subplot(121,aspect='equal')
b = figur.add_subplot(122,aspect='equal')
b.title.set_text("Single Detector Display")
a.title.set_text("Multiple Detector Display")
figrowspan = 10
figcolspan = 10
lsd = [0,0,0,0]
lsdlocal = 1000000
frameNr = 0
fileNumber = 0
getMax = 0
paramFN = 'PS.txt'
mask = None
bdata = None
mask2 = None
bigdetsize = 2048
initplot = 1
initplot2 = 1
_a_artist = None
_b_artist = None
_a_logmode = False
_b_logmode = False
origdetnum = 1
bclocal = [1024,1024]
ringRads = None
sg = 225
bcs = None
wl = 0.172979
px = 200
NrPixelsY = 2048
NrPixelsZ = 2048
Header = 8192
BytesPerPixel = 2
tempLsd = 1000000
tempMaxRingRad = 2000000
firstFileNrVar = Tk.StringVar()
fnextvar = Tk.StringVar()
fnextvar.set('.ge5')
nFramesPerFileVar = Tk.StringVar()
firstFileNrVar.set(str(1))
nFramesPerFileVar.set(str(240))
paramfilevar = Tk.StringVar()
paramfilevar.set(paramFN)
framenrvar = Tk.StringVar()
framenrvar.set(str(frameNr))
thresholdvar = Tk.StringVar()
threshold = 0
thresholdvar.set(str(threshold))
maxthresholdvar = Tk.StringVar()
maxthresholdvar.set(str(2000))
NrPixelsYVar = Tk.StringVar()
NrPixelsYVar.set(str(2048))
NrPixelsZVar = Tk.StringVar()
NrPixelsZVar.set(str(2048))
HeaderVar = Tk.IntVar()
HeaderVar.set(8192)
BytesVar = Tk.IntVar()
BytesVar.set(2)
LatticeConstant = np.zeros(6)
LatticeConstant[0] = 5.41116
LatticeConstant[1] = 5.41116
LatticeConstant[2] = 5.41116
LatticeConstant[3] = 90
LatticeConstant[4] = 90
LatticeConstant[5] = 90
lines = None
lines2 = None
DisplRingInfo = None
plotRingsVar = Tk.IntVar()
var = Tk.IntVar()
hydraVar = Tk.IntVar()
hydraVar.set(0)
sepfolderVar = Tk.IntVar()
getMaxVar = Tk.IntVar()
detnumbvar = Tk.StringVar()
detnumbvar.set(str(1))
lsdlocalvar = Tk.StringVar()
lsdlocalvar.set(str(lsdlocal))
bclocalvar1 = Tk.StringVar()
bclocalvar1.set(str(bclocal[0]))
bclocalvar2 = Tk.StringVar()
bclocalvar2.set(str(bclocal[1]))
nFramesMaxVar = Tk.IntVar()
maxStartFrameNrVar = Tk.IntVar()
nFramesMaxVar.set(240)
maxStartFrameNrVar.set(0)
dolog = Tk.IntVar()
hflip = Tk.IntVar()
vflip = Tk.IntVar()
transpose = Tk.IntVar()
refreshPlot = 0

canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)
toolbar_frame = Tk.Frame(root)
toolbar_frame.grid(row=figrowspan+5,column=0,columnspan=10,sticky=Tk.W)
toolbar = NavigationToolbar2Tk( canvas, toolbar_frame )
toolbar.update()

Tk.Button(master=root,text='Quit',command=_quit,font=("Helvetica",20)).grid(row=figrowspan+1,column=0,rowspan=3,sticky=Tk.W,padx=10)

# ~ zeroRowFrame = Tk.Frame(root)
# ~ zeroRowFrame.grid(row=figrowspan+1,column=1,sticky=Tk.W)
# ~ Tk.Label(master=zeroRowFrame,text='FileStem').grid(row=1,column=1,sticky=Tk.W)
# ~ Tk.Entry(master=zeroRowFrame,text=fnextvar,width=10).grid(row=1,column=2,sticky=Tk.W)

firstRowFrame = Tk.Frame(root)
firstRowFrame.grid(row=figrowspan+1,column=1,sticky=Tk.W)
Tk.Button(master=firstRowFrame,text='FirstFile',command=firstFileSelector,font=("Helvetica",12)).grid(row=1,column=1,sticky=Tk.W)
Tk.Button(master=firstRowFrame,text='DarkFile',command=darkFileSelector,font=("Helvetica",12)).grid(row=1,column=2,sticky=Tk.W)
Tk.Checkbutton(master=firstRowFrame,text="DarkCorr",variable=var).grid(row=1,column=3,sticky=Tk.W)
Tk.Label(master=firstRowFrame,text='firstFileNr').grid(row=1,column=4,sticky=Tk.W)
Tk.Entry(master=firstRowFrame,textvariable=firstFileNrVar,width=5).grid(row=1,column=5,sticky=Tk.W)
Tk.Label(master=firstRowFrame,text='nFrames/File').grid(row=1,column=6,sticky=Tk.W)
Tk.Entry(master=firstRowFrame,textvariable=nFramesPerFileVar,width=5).grid(row=1,column=7,sticky=Tk.W)
Tk.Label(master=firstRowFrame,text='FrameNr').grid(row=1,column=8,sticky=Tk.W)
Tk.Entry(master=firstRowFrame,textvariable=framenrvar,width=4).grid(row=1,column=9,sticky=Tk.W)
Tk.Button(master=firstRowFrame,text='+',command=incr_plotupdater,font=("Helvetica",12)).grid(row=1,column=10,sticky=Tk.W)
Tk.Button(master=firstRowFrame,text='-',command=decr_plotupdater,font=("Helvetica",12)).grid(row=1,column=11,sticky=Tk.W)

secondRowFrame = Tk.Frame(root)
secondRowFrame.grid(row=figrowspan+2,column=1,sticky=Tk.W)
Tk.Label(master=secondRowFrame,text='NrPixelsHor').grid(row=1,column=1,sticky=Tk.W)
Tk.Entry(master=secondRowFrame,textvariable=NrPixelsZVar,width=5).grid(row=1,column=2,sticky=Tk.W)
Tk.Label(master=secondRowFrame,text='NrPixelsVert').grid(row=1,column=3,sticky=Tk.W)
Tk.Entry(master=secondRowFrame,textvariable=NrPixelsYVar,width=5).grid(row=1,column=4,sticky=Tk.W)
Tk.Label(master=secondRowFrame,text='MinThresh').grid(row=1,column=5,sticky=Tk.W)
Tk.Entry(master=secondRowFrame,textvariable=thresholdvar,width=5).grid(row=1,column=6,sticky=Tk.W)
Tk.Label(master=secondRowFrame,text='MaxThresh').grid(row=1,column=7,sticky=Tk.W)
Tk.Entry(master=secondRowFrame,textvariable=maxthresholdvar,width=5).grid(row=1,column=8,sticky=Tk.W)
Tk.Button(master=secondRowFrame,text='UpdThresh',command=replot).grid(row=1,column=9,sticky=Tk.W)
Tk.Checkbutton(master=secondRowFrame,text="LogScale",variable=dolog).grid(row=1,column=10,sticky=Tk.W)

thirdRowFrame = Tk.Frame(root)
thirdRowFrame.grid(row=figrowspan+3,column=1,sticky=Tk.W)
Tk.Checkbutton(master=thirdRowFrame,text="MaxOverFrames",variable=getMaxVar).grid(row=1,column=1,sticky=Tk.W)
Tk.Label(master=thirdRowFrame,text="nFramesMax").grid(row=1,column=2,sticky=Tk.W)
Tk.Entry(master=thirdRowFrame,textvariable=nFramesMaxVar,width=5).grid(row=1,column=3)
Tk.Label(master=thirdRowFrame,text="startFrameNrMax").grid(row=1,column=4,sticky=Tk.W)
Tk.Entry(master=thirdRowFrame,textvariable=maxStartFrameNrVar,width=5).grid(row=1,column=5,sticky=Tk.W)
Tk.Label(master=thirdRowFrame,text="HeadSize").grid(row=1,column=6,sticky=Tk.W)
Tk.Entry(master=thirdRowFrame,textvariable=HeaderVar,width=5).grid(row=1,column=7,sticky=Tk.W)

thirdMidRowFrame = Tk.Frame(root)
thirdMidRowFrame.grid(row=figrowspan+4,column=1,sticky=Tk.W)
Tk.Button(master=thirdMidRowFrame,text="RingsMaterial",command=ringSelection).grid(row=1,column=12,sticky=Tk.W)
Tk.Checkbutton(master=thirdMidRowFrame,text='PlotRings',variable=plotRingsVar,command=clickRings).grid(row=1,column=13,sticky=Tk.W)
Tk.Checkbutton(master=thirdMidRowFrame,text="HFlip",variable=hflip).grid(row=1,column=14,sticky=Tk.W)
Tk.Checkbutton(master=thirdMidRowFrame,text="VFilp",variable=vflip).grid(row=1,column=15,sticky=Tk.W)
Tk.Checkbutton(master=thirdMidRowFrame,text="Transpose",variable=transpose).grid(row=1,column=16,sticky=Tk.W)
Tk.Label(master=thirdMidRowFrame,text="BytesPerPx").grid(row=1,column=17,sticky=Tk.W)
Tk.Entry(master=thirdMidRowFrame,textvariable=BytesVar,width=2).grid(row=1,column=18,sticky=Tk.W)

# Comment out Hydra for now, very custom code.
# fourthRowFrame = Tk.Frame(root)
# fourthRowFrame.grid(row=figrowspan+5,column=1,sticky=Tk.W)
# Tk.Label(master=fourthRowFrame,text="Hydra Only:",font=('Helvetica',15)).grid(row=1,column=1,sticky=Tk.W)
# Tk.Checkbutton(master=fourthRowFrame,text='IsHydra',variable=hydraVar).grid(row=1,column=2,sticky=Tk.W)
# Tk.Label(master=fourthRowFrame,text="ParamFile").grid(row=1,column=3,sticky=Tk.W)
# Tk.Button(master=fourthRowFrame,text="Select",command=paramfileselect).grid(row=1,column=4,sticky=Tk.W)
# Tk.Entry(master=fourthRowFrame,textvariable=paramfilevar,width=20).grid(row=1,column=5,sticky=Tk.W)
# Tk.Button(master=fourthRowFrame,text="LoadParams",command=readParams).grid(row=1,column=6,sticky=Tk.W)
# Tk.Button(master=fourthRowFrame,text="WriteParams",command=writeParams).grid(row=1,column=7,sticky=Tk.W)
# Tk.Button(master=fourthRowFrame,text="MakeBigDetector",command=makeBigDet).grid(row=1,column=8,sticky=Tk.W)
# Tk.Button(master=fourthRowFrame,text="CalibrateDetector",command=askRingsToExclude).grid(row=1,column=9,sticky=Tk.W)
# Tk.Checkbutton(master=fourthRowFrame,text='Separate Folders',variable=sepfolderVar).grid(row=1,column=10,sticky=Tk.W)
# Tk.Button(master=root,text='Load\nMultiple\nDetectors',command=plot_updater).grid(row=figrowspan+1,column=2,rowspan=3,sticky=Tk.W)

bframe = Tk.Frame(root)
bframe.grid(row=figrowspan+1,column=3,rowspan=3,sticky=Tk.W)
Tk.Label(master=bframe,text='DetNum').grid(row=1,column=1,sticky=Tk.W)
Tk.Entry(master=bframe,textvariable=detnumbvar,width=2).grid(row=1,column=2,sticky=Tk.W)
Tk.Label(master=bframe,text='Lsd').grid(row=2,column=1,sticky=Tk.W)
Tk.Entry(master=bframe,textvariable=lsdlocalvar,width=9).grid(row=2,column=2,sticky=Tk.W)
Tk.Label(master=bframe,text='BeamCenter').grid(row=3,column=1,sticky=Tk.W)
Tk.Entry(master=bframe,textvariable=bclocalvar1,width=6).grid(row=3,column=2,sticky=Tk.W)
Tk.Entry(master=bframe,textvariable=bclocalvar2,width=6).grid(row=3,column=3,sticky=Tk.W)
Tk.Button(master=root,text='Load\nSingle\nDetector',command=loadbplot).grid(row=figrowspan+1,column=4,rowspan=3)

if __name__ == "__main__":
	try:
		root.bind('<Control-w>', lambda event: root.destroy())
		Tk.mainloop()
	except KeyboardInterrupt:
		root.destroy()
