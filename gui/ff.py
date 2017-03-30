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

def getData(geNum,bytesToSkip):
	fn = getfn(fileStem,fileNumber,geNum)
	data = getImage(fn,bytesToSkip)
	doDark = var.get()
	if doDark == 1:
		if dark[geNum-startDetNr] is None:
			darkfn = getfn(darkStem,darkNum,geNum)
			dark[geNum-startDetNr] = getImage(darkfn,8192)
		thisdark = dark[geNum-startDetNr]
		corrected = np.subtract(data,thisdark)
	else:
		corrected = data
	#corrected = np.transpose(corrected)
	#corrected = np.flipud(corrected)
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

def plotRings():
	global lines
	Etas = np.linspace(-180,180,num=360)
	lines = []
	for ringrad in ringRads:
		Y = []
		Z = []
		for eta in Etas:
			tmp = YZ4mREta(ringrad,eta)
			Y.append(tmp[0]/px + bigdetsize/2)
			Z.append(tmp[1]/px + bigdetsize/2)
		lines.append(a.plot(Y,Z))

def doRings():
	global lines
	plotYesNo = plotRingsVar.get()
	if plotYesNo == 1:
		lines = None
		plotRings()
	else:
		if lines is not None:
			for line in lines:
				line.pop(0).remove()
			lines = None
	
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
			[thresholded,(cols,rows)] = getData(i,bytesToSkip)
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
		[mask2,(cols,rows)] = getData(startDetNr,bytesToSkip)
	lines = None
	doRings()
	a.imshow(mask2,cmap=plt.get_cmap('bone'),interpolation='nearest',clim=(threshold,upperthreshold))
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
	        return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x,y,z)
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
	global ty, tz, p0, p1, p2, fileNumber, dark, ringRads, ringNrs
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
	for line in paramContents:
		if line == '\n':
			continue
		if 'RingThresh' in line.split()[0]:
			ringslines.append(line)
			RingsToShow.append(int(line.split()[1]))
			threshold = max(threshold,float(line.split()[2]))
		if 'RawFolder' in line.split()[0]:
			folder = line.split()[1]
		if 'FileStem' in line.split()[0]:
			fileStem = line.split()[1]
		if 'Padding' in line.split()[0]:
			padding = int(line.split()[1])
		if 'StartDetNr' in line.split()[0]:
			startDetNr = int(line.split()[1])
		if 'EndDetNr' in line.split()[0]:
			endDetNr = int(line.split()[1])
		if 'Wedge' in line.split()[0]:
			wedge = float(line.split()[1])
		if 'BC' in line.split()[0]:
			bcs.append([float(line.split()[1]),float(line.split()[2])])
		if 'px' in line.split()[0]:
			px = float(line.split()[1])
		if 'Wavelength' in line.split()[0]:
			wl = float(line.split()[1])
		if 'BigDetSize' in line.split()[0]:
			bigdetsize = int(line.split()[1])
		if 'nFramesPerFile' in line.split()[0]:
			nFramesPerFile = int(line.split()[1])
		if 'FirstFileNumber' in line.split()[0]:
			firstFileNumber = int(line.split()[1])
			fileNumber = firstFileNumber
		if 'DarkStem' in line.split()[0]:
			darkStem = line.split()[1]
		if 'LatticeParameter' in line.split()[0]:
			LatC = line
		if 'Lsd' in line.split()[0]:
			lsdline = line
		if 'MaxRingRad' in line.split()[0]:
			maxRad = line
		if 'BorderToExclude' in line.split()[0]:
			border = line
		if 'DarkNum' in line.split()[0]:
			darkNum = int(line.split()[1])
		if 'SpaceGroup' in line.split()[0]:
			sg = int(line.split()[1])
		if 'OmegaStep' in line.split()[0]:
			omegaStep = float(line.split()[1])
		if 'OmegaFirstFile' in line.split()[0]:
			omegaStart = float(line.split()[1])
		if 'NrFilesPerSweep' in line.split()[0]:
			nFilesPerLayer = int(line.split()[1])
		if 'NrPixels' in line.split()[0]:
			NrPixels = int(line.split()[1])
		if 'NumDetectors' in line.split()[0]:
			nDetectors = int(line.split()[1])
		if 'Lsd' in line.split()[0]:
			lsd.append(float(line.split()[1]))
		if 'tx' in line.split()[0]:
			tx.append(float(line.split()[1]))
		if 'DetParams' in line.split()[0]:
			lsd.append(float(line.split()[1]))
			bcs.append([float(line.split()[2]),float(line.split()[3])])
			tx.append(float(line.split()[4]))
			ty.append(float(line.split()[5]))
			tz.append(float(line.split()[6]))
			p0.append(float(line.split()[7]))
			p1.append(float(line.split()[8]))
			p2.append(float(line.split()[9]))
			RhoDs.append(float(line.split()[10]))
	bigFN = 'BigDetectorMaskEdgeSize' + str(bigdetsize) + 'x' + str(bigdetsize) + 'Unsigned16Bit.bin'
	hklGenPath = '~/opt/MIDAS/FF_HEDM/bin/GetHKLList '
	os.system(hklGenPath + paramFN)
	hklfn = 'hkls.csv'
	hklfile = open(hklfn,'r')
	hklfile.readline()
	hklinfo = hklfile.readlines()
	ringRads = []
	ringNrs = []
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
	f.write('Width 1500\n')
	f.write('LatticeParameter 5.411651 5.411651 5.411651 90 90 90\nSpaceGroup 225\n')
	f.write('NrPixels '+str(NrPixels)+'\n')
	f.write('Wavelength '+str(wl)+'\n')
	f.write('Lsd '+str(lsd[detNum-startDetNr])+'\n')
	f.write('RhoD '+ str(RhoDs[detNum-startDetNr])+'\n')
	f.write('StartNr ' + str(firstFileNumber)+'\n')
	f.write('EndNr ' + str(firstFileNumber+nFilesPerLayer-1)+'\n')
	f.write('tolTilts 4\ntolBC 10\ntolLsd 5000\ntolP 1E-3\np0 0\np1 0\np2 0\nEtaBinSize 5\n')
	f.write('ty 0\ntz 0\nWedge 0\n')
	f.write('tx '+str(tx[detNum-startDetNr])+'\n')
	if len(ringsToExclude) > 0:
		for ring in ringsToExclude:
			f.write('RingsToExclude '+str(ring)+'\n')

def writeParams():
	pfname = os.getcwd() + 'GeneratedParameters.txt'
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
		for line in output:
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

# Main function
root = Tk.Tk()
root.wm_title("FF display v0.1 Dt. 2017/03/29 hsharma@anl.gov")
figur = Figure(figsize=(20,7.5),dpi=100)
canvas = FigureCanvasTkAgg(figur,master=root)
a = figur.add_subplot(121,aspect='equal')
b = figur.add_subplot(122)
figrowspan = 10
figcolspan = 10
canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=5,sticky=Tk.W+Tk.E+Tk.N+Tk.S)#pack(side=Tk.TOP,fill=Tk.BOTH)
toolbar_frame = Tk.Frame(root)
toolbar_frame.grid(row=figrowspan+4,column=0,columnspan=5,sticky=Tk.W)
toolbar = NavigationToolbar2TkAgg( canvas, toolbar_frame )
toolbar.update()

firstRowFrame = Tk.Frame(root)
firstRowFrame.grid(row=figrowspan+1,column=1,sticky=Tk.W)

frameNr = 0
fileNumber = 0
paramFN = 'PS.txt'
Tk.Label(master=firstRowFrame,text="ParamFile").grid(row=1,column=1,sticky=Tk.W)#pack(side=Tk.LEFT)
buttonparam = Tk.Button(master=firstRowFrame,text="Select",command=paramfileselect)
buttonparam.grid(row=1,column=2,sticky=Tk.W)
paramfilevar = Tk.StringVar()
paramfilevar.set(paramFN)
e0 = Tk.Entry(master=firstRowFrame,textvariable=paramfilevar,width=40)
e0.grid(row=1,column=3,sticky=Tk.W)#pack(side=Tk.LEFT)


buttonLoadParam = Tk.Button(master=firstRowFrame,text="LoadParams",command=readParams)
buttonLoadParam.grid(row=1,column=4,sticky=Tk.W)

var = Tk.IntVar()
c = Tk.Checkbutton(master=firstRowFrame,text="Subtract Dark",variable=var)
c.grid(row=1,column=5,sticky=Tk.W)#pack(side=Tk.LEFT)

mask = None
buttonMakeBigDet = Tk.Button(master=firstRowFrame,text="MakeBigDetector",command=makeBigDet)
buttonMakeBigDet.grid(row=1,column=6,sticky=Tk.W)

initplot = 1

buttonCalibrate = Tk.Button(master=firstRowFrame,text="CalibrateDetector",command=askRingsToExclude)
buttonCalibrate.grid(row=1,column=7,sticky=Tk.W)

buttonCalibrate = Tk.Button(master=firstRowFrame,text="WriteParams",command=writeParams)
buttonCalibrate.grid(row=1,column=8,sticky=Tk.W)

secondRowFrame = Tk.Frame(root)
secondRowFrame.grid(row=figrowspan+2,column=1,sticky=Tk.W)

Tk.Label(master=secondRowFrame,text='FrameNr').grid(row=1,column=1,sticky=Tk.W)
framenrvar = Tk.StringVar()
framenrvar.set(str(frameNr))
eFrameNr = Tk.Entry(master=secondRowFrame,textvariable=framenrvar,width=4)
eFrameNr.grid(row=1,column=2,sticky=Tk.W)

buttonIncr = Tk.Button(master=secondRowFrame,text='+',command=incr_plotupdater,font=("Helvetica",12))
buttonIncr.grid(row=1,column=3,sticky=Tk.W)#pack(side=Tk.LEFT)
buttonDecr = Tk.Button(master=secondRowFrame,text='-',command=decr_plotupdater,font=("Helvetica",12))
buttonDecr.grid(row=1,column=4,sticky=Tk.W)#pack(side=Tk.LEFT)

Tk.Label(master=secondRowFrame,text='MinThreshold').grid(row=1,column=5,sticky=Tk.W)
thresholdvar = Tk.StringVar()
threshold = 0
thresholdvar.set(str(threshold))
ethreshold = Tk.Entry(master=secondRowFrame,textvariable=thresholdvar,width=5)
ethreshold.grid(row=1,column=6,sticky=Tk.W)

Tk.Label(master=secondRowFrame,text='MaxThreshold').grid(row=1,column=7,sticky=Tk.W)
maxthresholdvar = Tk.StringVar()
maxthresholdvar.set(str(2000))
Tk.Entry(master=secondRowFrame,textvariable=maxthresholdvar,width=5).grid(row=1,column=8,sticky=Tk.W)

lines = None
plotRingsVar = Tk.IntVar()
cplotRings = Tk.Checkbutton(master=secondRowFrame,text='Plot Rings',variable=plotRingsVar,command=clickRings)
cplotRings.grid(row=1,column=9,sticky=Tk.E)

button = Tk.Button(master=root,text='Quit',command=_quit,font=("Helvetica",20))
button.grid(row=figrowspan+1,column=0,rowspan=3,sticky=Tk.W,padx=10)#pack(side=Tk.LEFT)

button2 = Tk.Button(master=root,text='Load',command=plot_updater,font=("Helvetica",20))
button2.grid(row=figrowspan+1,column=2,rowspan=3,sticky=Tk.E,padx=10)#pack(side=Tk.LEFT)

Tk.mainloop()
