#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

import PIL
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import sys
import Tkinter as Tk
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from scipy import stats
import tkFileDialog
import math
import scipy
import scipy.ndimage as ndimage
from math import sin, cos, acos, sqrt, tan, atan, atan2, fabs
from numpy import linalg as LA
import math
from subprocess import Popen, PIPE, STDOUT
from multiprocessing.dummy import Pool

# Helpers
deg2rad = 0.0174532925199433
rad2deg = 57.2957795130823

colors = ['r','g','b','c','m','y','r','g','b','c','m','y','r','g','b',
		  'c','m','y','r','g','b','c','m','y','r','g','b','c','m','y',
		  'r','g','b','c','m','y','r','g','b','c','m','y','r','g','b',
		  'c','m','y','r','g','b','c','m','y','r','g','b','c','m','y',
		  'r','g','b','c','m','y','r','g','b','c','m','y','r','g','b',
		  'c','m','y','r','g','b','c','m','y','r','g','b','c','m','y']

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
	return [alpha,Rad*px]

def YZ4mREta(R,Eta):
	return [-R*sin(Eta*deg2rad),R*cos(Eta*deg2rad)]

def getfn(fstem,fnum,geNum):
	return folder + fstem + '_' + str(fnum).zfill(padding) + '.ge' + str(geNum)

def getImage(fn,bytesToSkip):
	print "Reading file: " + fn
	f = open(fn,'rb')
	f.seek(bytesToSkip,os.SEEK_SET)
	data = np.fromfile(f,dtype=np.uint16,count=(NrPixels*NrPixels))
	f.close()
	data = np.reshape(data,(NrPixels,NrPixels))
	data = data.astype(float)
	return data

def getImageMax(fn):
	print "Reading file: " + fn
	f = open(fn,'rb')
	f.seek(8192,os.SEEK_SET)
	dataMax = np.zeros(NrPixels*NrPixels)
	for framenr in range(nFramesPerFile):
		data = np.fromfile(f,dtype=np.uint16,count=(NrPixels*NrPixels))
		dataMax = np.maximum(dataMax,data)
	f.close()
	dataMax = np.reshape(dataMax,(NrPixels,NrPixels))
	dataMax = dataMax.astype(float)
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
		if dark[geNum-startDetNr] is None:
			darkfn = getfn(darkStem,darkNum,geNum)
			dark[geNum-startDetNr] = getImage(darkfn,8192)
		thisdark = dark[geNum-startDetNr]
		corrected = np.subtract(data,thisdark)
	else:
		corrected = data
	corrected[corrected < threshold] = 0
	nonzerocoords = np.nonzero(corrected)
	return [corrected,nonzerocoords]

def transforms(idx):
	txr = tx[idx]*deg2rad
	tyr = ty[idx]*deg2rad
	tzr = tz[idx]*deg2rad
	Rx = np.array([[1,0,0],[0,cos(txr),-sin(txr)],[0,sin(txr),cos(txr)]])
	Ry = np.array([[cos(tyr),0,sin(tyr)],[0,1,0],[-sin(tyr),0,cos(tyr)]])
	Rz = np.array([[cos(tzr),-sin(tzr),0],[sin(tzr),cos(tzr),0],[0,0,1]])
	return np.dot(Rx,np.dot(Ry,Rz))

def plotRingsOffset():
	global lines2
	global lsdlocal, bclocal
	lsdlocal = float(lsdlocalvar.get())
	bclocal[0] = float(bclocalvar1.get())
	bclocal[1] = float(bclocalvar2.get())
	Etas = np.linspace(-180,180,num=360)
	lines2 = []
	colornr = 0
	for ringrad in ringRads:
		Y = []
		Z = []
		for eta in Etas:
			ringrad2 = ringrad * (lsdlocal / lsdorig)
			tmp = YZ4mREta(ringrad2,eta)
			Y.append(tmp[0]/px + bclocal[0])
			Z.append(tmp[1]/px + bclocal[1])
		lines2.append(b.plot(Y,Z,color=colors[colornr]))
		colornr+= 1

def plotRings():
	global lines
	Etas = np.linspace(-180,180,num=360)
	lines = []
	colornr = 0
	for ringrad in ringRads:
		Y = []
		Z = []
		for eta in Etas:
			tmp = YZ4mREta(ringrad,eta)
			Y.append(tmp[0]/px + bigdetsize/2)
			Z.append(tmp[1]/px + bigdetsize/2)
		lines.append(a.plot(Y,Z,color=colors[colornr]))
		colornr+= 1

def doRings():
	global lines
	global lines2
	plotYesNo = plotRingsVar.get()
	if plotYesNo == 1:
		lines = None
		lines2 = None
		plotRings()
		plotRingsOffset()
	else:
		if lines is not None:
			for line in lines:
				line.pop(0).remove()
			lines = None
		if lines2 is not None:
			for line2 in lines2:
				line2.pop(0).remove()
			lines2 = None
	
def clickRings():
	doRings()
	canvas.show()
	canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)

def plot_updater():
	global initplot
	global fileNumber
	global frameNr
	global threshold
	global lims
	global lines
	if not initplot:
		lims = [a.get_xlim(), a.get_ylim()]
	frameNr = int(framenrvar.get())
	threshold = float(thresholdvar.get())
	upperthreshold = float(maxthresholdvar.get())
	a.clear()
	if nDetectors > 1:
		# Plot mask if wanted
		if mask is None:
			readBigDet()
		## Go through each geNum, get the data, transform it, put it on the bigDet
		mask2 = np.copy(mask)
		fileNumber = firstFileNumber + frameNr/nFramesPerFile
		framesToSkip = frameNr % nFramesPerFile
		bytesToSkip = 8192 + framesToSkip*(2*NrPixels*NrPixels)
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
	else:
		fileNumber = firstFileNumber + frameNr/nFramesPerFile
		framesToSkip = frameNr % nFramesPerFile
		bytesToSkip = 8192 + framesToSkip*(2*NrPixels*NrPixels)		
		[mask2,(rows,cols)] = getData(startDetNr,bytesToSkip)
	lines = None
	doRings()
	a.imshow(mask2,cmap=plt.get_cmap('bone'),interpolation='nearest',clim=(0,upperthreshold))
	if initplot:
		initplot = 0
	else:
		a.set_xlim([lims[0][0],lims[0][1]])
		a.set_ylim([lims[1][0],lims[1][1]])
	numrows, numcols = mask2.shape
	def format_coord(x, y):
	    col = int(x+0.5)
	    row = int(y+0.5)
	    if col>=0 and col<numcols and row>=0 and row<numrows:
	        z = mask2[row,col]
	        xD = x - bigdetsize/2
	        yD = y - bigdetsize/2
	        R = sqrt(xD*xD+yD*yD)
	        return 'x=%1.4f, y=%1.4f, Intensity=%1.4f, RingRad(pixels)=%1.4f'%(x,y,z,R)
	    else:
	        return 'x=%1.4f, y=%1.4f'%(x,y)
	a.format_coord = format_coord
	a.title.set_text("Image")
	canvas.show()
	canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)

def incr_plotupdater():
	global frameNr
	global framenrvar
	frameNr += 1
	framenrvar.set(str(frameNr))
	plot_updater()

def decr_plotupdater():
	global frameNr
	global framenrvar
	frameNr -= 1
	framenrvar.set(str(frameNr))
	plot_updater()

def readParams():
	global paramFN
	paramFN = paramfilevar.get()
	global folder, fileStem, padding, startDetNr, endDetNr, bigFN
	global wedge, lsd, px, bcs, tx, wl, bigdetsize, nFramesPerFile
	global firstFileNumber, darkStem, darkNum, omegaStep, nFilesPerLayer
	global omegaStart, NrPixels, threshold, RingsToShow, nDetectors
	global RhoDs, LatC, sg, maxRad, border, ringslines, lsdline
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
		if 'NrPixels' == line.split()[0]:
			NrPixels = int(line.split()[1])
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
		print folder
	bigFN = 'BigDetectorMaskEdgeSize' + str(bigdetsize) + 'x' + str(bigdetsize) + 'Unsigned16Bit.bin'
	hklGenPath = '~/opt/MIDAS/FF_HEDM/bin/GetHKLList '
	os.system(hklGenPath + paramFN)
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
				break
	# initialization of dark
	dark = []
	for i in range(nDetectors):
		dark.append(None)
	print "Loaded"

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
	f.write('NrPixels '+str(NrPixels)+'\n')
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
	f.write('NrPixels '+str(NrPixels)+'\n')
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
	calibratecmd = '~/opt/MIDAS/FF_HEDM/bin/Calibrant '
	if ringsexcludestr == '0':
		ringsToExclude = []
	else:
		ringsToExclude = [int(rings) for rings in ringsexcludestr.split(',')]
		print ringsToExclude
	pfnames = []
	for i in range(startDetNr,endDetNr+1):
		print "Detector number " + str(i)
		pfnames.append('CalibrationDetNr' + str(i) + '.txt')
		writeCalibrateParams(pfnames[-1],i,ringsToExclude)
	processes = [Popen(calibratecmd+pfname,shell=True,
				stdin=PIPE, stdout=PIPE, stderr=STDOUT,close_fds=True) for pfname in pfnames]
	def get_lines(process):
		return process.communicate()[0].splitlines()
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
	bigf = open(bigFN,'r')
	mask = np.fromfile(bigf,dtype=np.uint16,count=bigdetsize*bigdetsize)
	bigf.close()
	mask = np.reshape(mask,(bigdetsize,bigdetsize))
	#mask = np.transpose(mask)
	mask = mask.astype(float)
	mask = np.fliplr(np.flipud(mask))

def makeBigDet():
	cmdf = '~/opt/MIDAS/FF_HEDM/bin/MapMultipleDetectors '
	os.system(cmdf+paramFN)
	readBigDet()

def loadbplot():
	global bclocal
	global lsdlocal
	global lsdorig
	global initplot2
	global origdetnum
	global bclocalvar1, bclocalvar2
	global ax
	if not initplot2:
		lims = [b.get_xlim(), b.get_ylim()]
	frameNr = int(framenrvar.get())
	threshold = float(thresholdvar.get())
	upperthreshold = float(maxthresholdvar.get())
	b.clear()
	fileNumber = firstFileNumber + frameNr/nFramesPerFile
	framesToSkip = frameNr % nFramesPerFile
	bytesToSkip = 8192 + framesToSkip*(2*NrPixels*NrPixels)
	detnr = int(detnumbvar.get())
	if detnr != origdetnum or initplot2:
		origdetnum = detnr
		bclocal[0] = bcs[detnr-startDetNr][0]
		bclocal[1] = bcs[detnr-startDetNr][1]
		bclocalvar1.set(str(bclocal[0]))
		bclocalvar2.set(str(bclocal[1]))
	else:
		bclocal[0] = float(bclocalvar1.get())
		bclocal[1] = float(bclocalvar2.get())
	[data, coords] = getData(detnr,bytesToSkip)
	lsdorig = lsd[detnr-startDetNr]
	lsdlocal = float(lsdlocalvar.get())
	if plotRingsVar.get() == 1:
		plotRingsOffset()
	#data= np.flipud(data)
	b.imshow(data,cmap=plt.get_cmap('bone'),interpolation='nearest',clim=(threshold,upperthreshold))
	if initplot2:
		initplot2 = 0
		b.invert_yaxis()
	else:
		b.set_xlim([lims[0][0],lims[0][1]])
		b.set_ylim([lims[1][0],lims[1][1]])
	numrows, numcols = data.shape
	def format_coord(x, y):
	    col = int(x+0.5)
	    row = int(y+0.5)
	    if col>=0 and col<numcols and row>=0 and row<numrows:
	        z = data[row,col]
	        return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x,NrPixels-y,z)
	    else:
	        return 'x=%1.4f, y=%1.4f'%(x,y)
	b.format_coord = format_coord
	canvas.show()
	canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)

def acceptRings():
	global RingsToShow
	global ringRads
	global topSelectRings
	items = ListBox1.curselection()
	ringRads = [RingRad[int(item)] for item in items]
	RingsToShow = [int(item) for item in items]
	topSelectRings.destroy()

def selectRings():
	global topSelectRings
	global hklLines, hkl, ds, Ttheta, RingRad, ListBox1
	hklGenPath = '~/opt/MIDAS/FF_HEDM/bin/GetHKLList '
	pfname = 'ps_midas_ff.txt'
	f = open(pfname,'w')
	f.write('Wavelength ' + str(wl) + '\n')
	f.write('SpaceGroup ' + str(sg) + '\n')
	f.write('Lsd ' + str(tempLsd) + '\n')
	f.write('MaxRingRad ' + str(tempMaxRingRad) + '\n')
	f.write('LatticeConstant ')
	for i in range(6):
		f.write(str(LatticeConstant[i]) + ' ')
	f.write('\n')
	f.close()
	os.system(hklGenPath + pfname)
	#os.system('rm '+pfname)
	hklfn = 'hkls.csv'
	hklfile = open(hklfn,'r')
	header = hklfile.readline()
	header = header.replace(' ','      ')
	hklinfo = hklfile.readlines()
	hklfile.close()
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
	sg = int(sgVar.get())
	LatticeConstant = np.zeros(6)
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
	global topRingMaterialSelection
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
	for i in range(3):
		LatticeConstantVar[i].set(str(3.6))
		LatticeConstantVar[i+3].set(str(90))
	topRingMaterialSelection = Tk.Toplevel()
	topRingMaterialSelection.title('Select the SpaceGroup, Wavelength(or Energy), Lattice Constant')
	Tk.Label(master=topRingMaterialSelection,text='Please enter the SpaceGroup, Wavelength(or Energy), Lattice Constant, Sample To Detector Distance(Lsd)').grid(row=1,column=1,columnspan=7)
	Tk.Label(master=topRingMaterialSelection,text='SpaceGroup').grid(row=2,column=1,sticky=Tk.W)
	Tk.Entry(master=topRingMaterialSelection,textvariable=sgVar,width=4).grid(row=2,column=2,sticky=Tk.W)
	Tk.Label(master=topRingMaterialSelection,text='Wavelength (A)').grid(row=3,column=1,sticky=Tk.W)
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

# Main function
root = Tk.Tk()
root.wm_title("FF display v0.1 Dt. 2017/03/29 hsharma@anl.gov")
figur = Figure(figsize=(19.5,8.5),dpi=100)
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
bigdetsize = 2048
initplot = 1
initplot2 = 1
origdetnum = 1
bclocal = [0,0]
sg = 225
wl = 0.172979
px = 200
tempLsd = 1000000
tempMaxRingRad = 2000000

canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=5,sticky=Tk.W+Tk.E+Tk.N+Tk.S)#pack(side=Tk.TOP,fill=Tk.BOTH)
toolbar_frame = Tk.Frame(root)
toolbar_frame.grid(row=figrowspan+4,column=0,columnspan=5,sticky=Tk.W)
toolbar = NavigationToolbar2TkAgg( canvas, toolbar_frame )
toolbar.update()

firstRowFrame = Tk.Frame(root)
firstRowFrame.grid(row=figrowspan+1,column=1,sticky=Tk.W)

Tk.Label(master=firstRowFrame,text="ParamFile").grid(row=1,column=1,sticky=Tk.W)
buttonparam = Tk.Button(master=firstRowFrame,text="Select",command=paramfileselect)
buttonparam.grid(row=1,column=2,sticky=Tk.W)
paramfilevar = Tk.StringVar()
paramfilevar.set(paramFN)
e0 = Tk.Entry(master=firstRowFrame,textvariable=paramfilevar,width=20)
e0.grid(row=1,column=3,sticky=Tk.W)

buttonLoadParam = Tk.Button(master=firstRowFrame,text="LoadParams",command=readParams)
buttonLoadParam.grid(row=1,column=4,sticky=Tk.W)



var = Tk.IntVar()
c = Tk.Checkbutton(master=firstRowFrame,text="Subtract Dark",variable=var)
c.grid(row=1,column=5,sticky=Tk.W)

getMaxVar = Tk.IntVar()
c2 = Tk.Checkbutton(master=firstRowFrame,text="MaxOverFrames",variable=getMaxVar)
c2.grid(row=1,column=6,sticky=Tk.W)

lines = None
plotRingsVar = Tk.IntVar()
cplotRings = Tk.Checkbutton(master=firstRowFrame,text='Plot Rings',variable=plotRingsVar,command=clickRings)
cplotRings.grid(row=1,column=7,sticky=Tk.E)

secondRowFrame = Tk.Frame(root)
secondRowFrame.grid(row=figrowspan+2,column=1,sticky=Tk.W)

firstFileNrVar = Tk.StringVar()
nFramesPerFileVar = Tk.StringVar()
firstFileNrVar.set(str(1))
nFramesPerFileVar.set(str(240))
Tk.Label(master=secondRowFrame,text='firstFileNr').grid(row=1,column=1,sticky=Tk.W)
efirstfile = Tk.Entry(master=secondRowFrame,textvariable=firstFileNrVar,width=5)
efirstfile.grid(row=1,column=2,sticky=Tk.W)
Tk.Label(master=secondRowFrame,text='nFramesPerFile').grid(row=1,column=3,sticky=Tk.W)
enFrames = Tk.Entry(master=secondRowFrame,textvariable=nFramesPerFileVar,width=5)
enFrames.grid(row=1,column=4,sticky=Tk.W)

Tk.Label(master=secondRowFrame,text='FrameNr').grid(row=1,column=5,sticky=Tk.W)
framenrvar = Tk.StringVar()
framenrvar.set(str(frameNr))
eFrameNr = Tk.Entry(master=secondRowFrame,textvariable=framenrvar,width=4)
eFrameNr.grid(row=1,column=6,sticky=Tk.W)

buttonIncr = Tk.Button(master=secondRowFrame,text='+',command=incr_plotupdater,font=("Helvetica",12))
buttonIncr.grid(row=1,column=7,sticky=Tk.W)
buttonDecr = Tk.Button(master=secondRowFrame,text='-',command=decr_plotupdater,font=("Helvetica",12))
buttonDecr.grid(row=1,column=8,sticky=Tk.W)

Tk.Label(master=secondRowFrame,text='MinThreshold').grid(row=1,column=9,sticky=Tk.W)
thresholdvar = Tk.StringVar()
threshold = 0
thresholdvar.set(str(threshold))
ethreshold = Tk.Entry(master=secondRowFrame,textvariable=thresholdvar,width=5)
ethreshold.grid(row=1,column=10,sticky=Tk.W)

Tk.Label(master=secondRowFrame,text='MaxThreshold').grid(row=1,column=11,sticky=Tk.W)
maxthresholdvar = Tk.StringVar()
maxthresholdvar.set(str(2000))
Tk.Entry(master=secondRowFrame,textvariable=maxthresholdvar,width=5).grid(row=1,column=12,sticky=Tk.W)

Tk.Label(master=secondRowFrame,text='NrPixels').grid(row=1,column=13,sticky=Tk.W)
NrPixelsVar = Tk.StringVar()
NrPixelsVar.set(str(2048))
enPixels = Tk.Entry(master=secondRowFrame,textvariable=NrPixelsVar,width=5)
enPixels.grid(row=1,column=14,sticky=Tk.W)

thirdRowFrame = Tk.Frame(root)
thirdRowFrame.grid(row=figrowspan+3,column=1,sticky=Tk.W)

buttonMakeBigDet = Tk.Button(master=thirdRowFrame,text="MakeBigDetector",command=makeBigDet)
buttonMakeBigDet.grid(row=1,column=1,sticky=Tk.W)

buttonCalibrate = Tk.Button(master=thirdRowFrame,text="CalibrateDetector",command=askRingsToExclude)
buttonCalibrate.grid(row=1,column=2,sticky=Tk.W)

buttonCalibrate2 = Tk.Button(master=thirdRowFrame,text="WriteParams",command=writeParams)
buttonCalibrate2.grid(row=1,column=3,sticky=Tk.W)

buttonSelectRings = Tk.Button(master=thirdRowFrame,text="SelectRingsAndMaterial",command=ringSelection)
buttonSelectRings.grid(row=1,column=4,sticky=Tk.W)

button = Tk.Button(master=root,text='Quit',command=_quit,font=("Helvetica",20))
button.grid(row=figrowspan+1,column=0,rowspan=3,sticky=Tk.W,padx=10)

button2 = Tk.Button(master=root,text='Load\nMultiple\nDetectors',command=plot_updater)
button2.grid(row=figrowspan+1,column=2,rowspan=3,sticky=Tk.E,padx=10)

bframe = Tk.Frame(root)
bframe.grid(row=figrowspan+1,column=3,rowspan=3,sticky=Tk.W)

Tk.Label(master=bframe,text='DetNum').grid(row=1,column=1,sticky=Tk.W)
detnumbvar = Tk.StringVar()
detnumbvar.set(str(1))
Tk.Entry(master=bframe,textvariable=detnumbvar,width=2).grid(row=1,column=2,sticky=Tk.W)

Tk.Label(master=bframe,text='Lsd').grid(row=2,column=1,sticky=Tk.W)
lsdlocalvar = Tk.StringVar()
lsdlocalvar.set(str(lsdlocal))
Tk.Entry(master=bframe,textvariable=lsdlocalvar,width=9).grid(row=2,column=2,sticky=Tk.W)

Tk.Label(master=bframe,text='BeamCenter').grid(row=3,column=1,sticky=Tk.W)
bclocalvar1 = Tk.StringVar()
bclocalvar1.set(str(bclocal[0]))
Tk.Entry(master=bframe,textvariable=bclocalvar1,width=6).grid(row=3,column=2,sticky=Tk.W)
bclocalvar2 = Tk.StringVar()
bclocalvar2.set(str(bclocal[1]))
Tk.Entry(master=bframe,textvariable=bclocalvar2,width=6).grid(row=3,column=3,sticky=Tk.W)

Tk.Button(master=root,text='Load\nSingle\nDetector',command=loadbplot).grid(row=figrowspan+1,column=4,rowspan=3)

Tk.mainloop()
