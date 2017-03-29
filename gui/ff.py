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

def _quit():
	root.quit()
	root.destroy()

# Helpers
deg2rad = 0.0174532925199433
rad2deg = 57.2957795130823

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
	return folder + fstem + str(fnum).zfill(padding) + '.ge' + str(geNum)

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
	corrected = np.transpose(corrected)
	thresholded = stats.threshold(corrected,threshmin=threshold)
	nonzerocoords = np.nonzero(thresholded)
	return [thresholded,nonzerocoords]

def plot_updater():
	a.clear()
	# Plot mask if wanted
	if mask is not None:
		readBigDet()
		a.imshow(mask,extent=[-bigdetsize/2,bigdetsize/2,-bigdetsize/2,bigdetsize/2])
		canvas.show()
		canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)
	else:
		makeBigDet()

def readParams():
	global paramFN
	paramFN = paramfilevar.get()
	global folder, fileStem, padding, startDetNr, endDetNr, bigFN
	global wedge, lsd, px, bcs, tx, wl, bigdetsize, nFramesPerFile
	global firstFileNumber, darkStem, darkNum, omegaStep, nFilesPerLayer
	global omegaStart, NrPixels, threshold, RingsToShow, nDetectors
	paramContents = open(paramFN,'r').readlines()
	lsd = []
	bcs = []
	tx = []
	RingsToShow = []
	threshold = 0
	for line in paramContents:
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
		if 'DarkStem' in line.split()[0]:
			darkStem = line.split()[1]
		if 'DarkNum' in line.split()[0]:
			darkNum = int(line.split()[1])
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
		if 'RingThresh' in line.split()[0]:
			RingsToShow.append(int(line.split()[1]))
			threshold = max(threshold,float(line.split()[2]))
		if 'Lsd' in line.split()[0]:
			lsd.append(float(line.split()[1]))
		if 'tx' in line.split()[0]:
			tx.append(float(line.split()[1]))
		if 'DetParams' in line.split()[0]:
			lsd.append(float(line.split()[1]))
			bcs.append([float(line.split()[2]),float(line.split()[3])])
			tx.append(float(line.split()[4]))
	bigFN = 'BigDetectorMaskEdgeSize' + str(bigdetsize) + 'x' + str(bigdetsize) + 'Unsigned16Bit.bin'
	hklGenPath = '~/opt/MIDAS/FF_HEDM/bin/GetHKLList '
	os.system(hklGenPath + paramFN)
	# initialization of dark
	dark = []
	for i in range(nDetectors):
		dark.append(None)
	print "Loaded"

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

paramFN = 'PS.txt'
Tk.Label(master=root,text="ParamFile").grid(row=figrowspan+1,column=1,sticky=Tk.W)#pack(side=Tk.LEFT)
paramfilevar = Tk.StringVar()
paramfilevar.set(paramFN)
e0 = Tk.Entry(master=root,textvariable=paramfilevar,width=50)
e0.grid(row=figrowspan+1,column=1,sticky=Tk.E,padx=90)#pack(side=Tk.LEFT)

def paramfileselect():
	global paramFN
	global paramfilevar
	paramFN = tkFileDialog.askopenfilename()
	paramfilevar.set(paramFN)

buttonparam = Tk.Button(master=root,text="Select",command=paramfileselect)
buttonparam.grid(row=figrowspan+1,column=1,sticky=Tk.W,padx=70)

buttonLoadParam = Tk.Button(master=root,text="LoadParams",command=readParams)
buttonLoadParam.grid(row=figrowspan+1,column=2,sticky=Tk.W)

var = Tk.IntVar()
c = Tk.Checkbutton(master=root,text="Subtract Dark",variable=var)
c.grid(row=figrowspan+1,column=2,sticky=Tk.W,padx=120)#pack(side=Tk.LEFT)

def readBigDet():
	global mask
	bigf = open(bigFN,'r')
	mask = np.fromfile(bigf,dtype=np.uint16,count=bigdetsize*bigdetsize)
	bigf.close()
	mask = np.reshape(mask,(bigdetsize,bigdetsize))
	mask = np.transpose(mask)

def makeBigDet():
	cmdf = '~/opt/MIDAS/FF_HEDM/bin/MapMultipleDetectors '
	os.system(cmdf+paramFN)
	readBigDet()
	a.imshow(mask,extent=[-bigdetsize/2,bigdetsize/2,-bigdetsize/2,bigdetsize/2])
	canvas.show()
	canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)

mask = None
buttonMakeBigDet = Tk.Button(master=root,text="MakeBigDetector",command=makeBigDet)
buttonMakeBigDet.grid(row=figrowspan+1,column=2,sticky=Tk.E)

button = Tk.Button(master=root,text='Quit',command=_quit,font=("Helvetica",20))
button.grid(row=figrowspan+1,column=0,rowspan=3,sticky=Tk.W,padx=10)#pack(side=Tk.LEFT)

button2 = Tk.Button(master=root,text='Load',command=plot_updater,font=("Helvetica",20))
button2.grid(row=figrowspan+1,column=5,rowspan=3,sticky=Tk.E,padx=10)#pack(side=Tk.LEFT)

Tk.mainloop()

'''
# Arguments
folder = '/var/host/media/removable/Data3/Sharma_Oct14/ge/'
paramFN = 'PS.txt'
fileStem = 'OMC_Cu_hydra_scan1'
hklGenPath = '~/opt/MIDAS/FF_HEDM/bin/GetHKLList '
fileNumber = 8
startDetNr = 1
endDetNr = 4
initFrameNumber = 10
darkName = 'dark_00001'
omegaStep = -0.25
totalFrames = 1440
NrPixels = 2048
RingsToShow = [1,2,3,4,5,6,7,8,9,10]
threshold = 100
minMarkerSize = 10
maxMarkerSize = 70

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
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.signal import argrelmax
from scipy import stats
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from math import sin, cos, acos, sqrt, tan, atan, atan2, fabs
from numpy import linalg as LA
import math
start_time = time.time()
root = Tk.Tk()
root.wm_title("GE Images Display v0.1 Dated 2016/11/14 hsharma@anl.gov")

nDetectors = endDetNr-startDetNr +1
figur = Figure(figsize=(4,4),dpi=100)
canvas = FigureCanvasTkAgg(figur,master=root)

# Helpers
deg2rad = 0.0174532925199433
rad2deg = 57.2957795130823

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

def Plotter(geNum,fileNum,framesToSkip,bytesToSkip):
	fileName = folder + fileStem + '_{:05d}'.format(fileNum) + '.ge' + str(geNum)
	print fileName
	f = open(fileName,'rb')
	f.seek(bytesToSkip,os.SEEK_SET)
	data = np.fromfile(f,dtype=np.uint16,count=(NrPixels*NrPixels))
	f.close()
	data = np.reshape(data,(NrPixels,NrPixels))
	data = data.astype(float)
	darkFileName = folder + darkName + '.ge' + str(geNum)
	f2 = open(darkFileName,'rb')
	f2.seek(8192,os.SEEK_SET)
	dark = np.fromfile(f2, dtype=np.uint16,count=(NrPixels*NrPixels))
	f2.close()
	dark = np.reshape(dark,(2048,2048))
	dark = dark.astype(float)
	corrected = np.subtract(data,dark)
	corrected = np.transpose(corrected)
	thresholded = stats.threshold(corrected,threshmin=threshold)
	rows,cols = argrelmax(thresholded)
	maximaVals = thresholded[rows,cols]
	sizeVals = minMarkerSize + (maxMarkerSize-minMarkerSize)*((maximaVals-threshold)/(14000-threshold))
	sizeVals = sizeVals.astype(int)
	sizeVals = list(sizeVals)
	txr = ts[geNum-1][0]*deg2rad
	tyr = ts[geNum-1][1]*deg2rad
	tzr = ts[geNum-1][2]*deg2rad
	Rx = np.array([[1,0,0],[0,cos(txr),-sin(txr)],[0,sin(txr),cos(txr)]])
	Ry = np.array([[cos(tyr),0,sin(tyr)],[0,1,0],[-sin(tyr),0,cos(tyr)]])
	Rz = np.array([[cos(tzr),-sin(tzr),0],[sin(tzr),cos(tzr),0],[0,0,1]])
	TRs = np.dot(Rx,np.dot(Ry,Rz))
	Xc = np.zeros(rows.shape)
	Yc = -(rows - BC[geNum-1][0])*px 
	Zc = (cols - BC[geNum-1][1])*px
	ABC = np.array([Xc,Yc,Zc])
	ABCPr = np.dot(TRs,ABC)
	XYZ = np.array([ABCPr[0,:]+Lsd[geNum-1],ABCPr[1,:],ABCPr[2,:]])
	Rad = np.multiply((LsdMean/XYZ[0,:]),LA.norm(XYZ[1:,:],axis = 0))
	Eta = CalcEtaAngle(XYZ)
	a.scatter(XYZ[1,:]/px,XYZ[2,:]/px,s=sizeVals,color='red')
	
def PlotRings(ringRads):
	Etas = np.linspace(-180,180,num=360)
	for ringrad in ringRads:
		Y = []
		Z = []
		for eta in Etas:
			tmp = YZ4mREta(ringrad,eta)
			Y.append(tmp[0]/px)
			Z.append(tmp[1]/px)
		a.plot(Y,Z)

def draw_plot(frameNumber,ringRads):
	global initplot
	global floatText
	global AllMaximaVals
	if not initplot:
		lims = [a.get_xlim(), a.get_ylim()]
	a.clear()
	floatText = None
	# Plot mask
	a.imshow(mask,extent=[-BigDetSize/2,BigDetSize/2,-BigDetSize/2,BigDetSize/2])
	fileNum = fileNumber + frameNumber/nFrames
	framesToSkip = frameNumber % nFrames
	bytesToSkip = 8192 + (framesToSkip) * (2*NrPixels*NrPixels) # 2 bytes
	# go through each hydra detector
	print "Reading files"
	for geNum in range(startDetNr,endDetNr+1,1):
		Plotter(geNum,fileNum,framesToSkip,bytesToSkip)
	PlotRings(ringRads)
	a.axis('equal')
	if initplot:
		initplot = 0
	else:
		a.set_xlim([lims[0][0],lims[0][1]])
		a.set_ylim([lims[1][0],lims[1][1]])
	canvas.show()
	canvas.get_tk_widget().pack(side=Tk.TOP,fill=Tk.BOTH,expand=1)

def plot_updater():
	global ringRads
	global frameNumber
	newframeNumber = int(r.get())
	if newframeNumber != frameNumber:
		frameNumber = newframeNumber
		draw_plot(frameNumber,ringRads)

def update_plot(event):
	print "Pressed Enter"
	plot_updater()

def spotToGV(y,z):
	ome = 0
	tht = atan((y*y+z*z)/LsdMean)/2
	eta = atan2(-y,z)
	ds = 2*sin(tht)/wl
	k1 = -ds*sin(tht)
	k2 = -ds*cos(tht)*sin(eta)
	k3 =  ds*cos(tht)*cos(eta)
	k1f = k1*cos(wedge) + k3*sin(wedge)
	k3f = k3*cos(wedge) - k1*sin(wedge)
	k2f = k2
	g1a = k1f * cos(ome) + k2f * sin(ome)
	g2a = k2f * cos(ome) - k1f * sin(ome)
	g3a = k3f
	normGa = sqrt(g1a*g1a + g2a*g2a + g3a*g3a)
	g1 = g1a*ds/normGa
	g2 = g2a*ds/normGa
	g3 = g3a*ds/normGa
	return [g1,g2,g3]

def CalcOmes(Gv):
	g1 = Gv[0]
	g2 = Gv[1]
	g3 = Gv[2]
	Length_G = sqrt((g1*g1)+(g2*g2)+(g3*g3))
	k1i = -(Length_G*Length_G)*(wl/2)
	A = (k1i+(g3*sin(wedge)))/cos(wedge)
	a_Sin = (g1*g1) + (g2*g2)
	b_Sin = 2*A*g2
	c_Sin = (A*A) - (g1*g1)
	a_Cos = a_Sin
	b_Cos = -2*A*g1
	c_Cos = (A*A) - (g2*g2)
	Par_Sin = (b_Sin*b_Sin)-(4*a_Sin*c_Sin)
	Par_Cos = (b_Cos*b_Cos)-(4*a_Cos*c_Cos)
	P_check_Sin = 0
	P_check_Cos = 0
	if (Par_Sin >=0):
		P_Sin = sqrt(Par_Sin)
	else:
		P_Sin = 0
		P_check_Sin = 1
	if (Par_Cos >=0):
		P_Cos = sqrt(Par_Cos)
	else:
		P_Cos = 0
		P_check_Cos = 1
	Sin_Omega1 = ((-b_Sin)-(P_Sin))/(2*a_Sin)
	Sin_Omega2 = ((-b_Sin)+(P_Sin))/(2*a_Sin)
	Cos_Omega1 = ((-b_Cos)-(P_Cos))/(2*a_Cos)
	Cos_Omega2 = ((-b_Cos)+(P_Cos))/(2*a_Cos)
	if (Sin_Omega1 < -1):
	    Sin_Omega1 = 0
	elif (Sin_Omega1 > 1):
	    Sin_Omega1 = 0
	elif (Sin_Omega2 > 1):
	    Sin_Omega2 = 0
	elif (Sin_Omega2 < -1):
	    Sin_Omega2 = 0
	if (Cos_Omega1 < -1):
	    Cos_Omega1 = 0
	elif (Cos_Omega1 > 1):
	    Cos_Omega1 = 0
	elif (Cos_Omega2 > 1):
	    Cos_Omega2 = 0
	elif (Cos_Omega2 < -1):
	    Cos_Omega2 = 0
	if (P_check_Sin == 1):
	    Sin_Omega1 = 0
	    Sin_Omega2 = 0
	if (P_check_Cos == 1):
	    Cos_Omega1 = 0
	    Cos_Omega2 = 0
	Option_1 = fabs((Sin_Omega1*Sin_Omega1) + (Cos_Omega1*Cos_Omega1) - 1)
	Option_2 = fabs((Sin_Omega1*Sin_Omega1) + (Cos_Omega2*Cos_Omega2) - 1)
	if (Option_1 < Option_2):
		Omega_1 = rad2deg*(atan2(Sin_Omega1,Cos_Omega1))
		Omega_2 = rad2deg*(atan2(Sin_Omega2,Cos_Omega2))
	else:
		Omega_1 = rad2deg*(atan2(Sin_Omega1,Cos_Omega2))
		Omega_2 = rad2deg*(atan2(Sin_Omega2,Cos_Omega1))
	if fabs(Omega_1) < 0.001:
		Omega_1 = 0
	elif fabs(Omega_2) < 0.001:
		Omega_2 = 0
	if fabs(fabs(Omega_1) - 180) < 0.001:
		Omega_1 = 180
	elif fabs(fabs(Omega_2) - 180) < 0.001:
		Omega_2 = 180
	return [Omega_1,Omega_2]

def calcFriedelFrames(y,z): # wl - wavelength, LsdMean - Lsd
	Gv = spotToGV(y,z)
	Omegas = CalcOmes(Gv)
	Omegas2 = CalcOmes([-Gv[0],-Gv[1],-Gv[2]])
	Omegas.extend(Omegas2)
	posz = [z, z, -z, -z]
	try:
		idx = Omegas.index(0)
	except ValueError:
		idx = -1
		print "Wow, could not find the spot itself."
	try:
		idx180 = Omegas.index(180)
	except ValueError:
		idx180 = -1
		print "Wow, could not find 180 Friedel Pair."
	if idx == 0:
		if idx180 == 3:
			posy = [y, -y, -y, y]
		else:
			posy = [y, -y, y, -y]
	else:
		if idx180 == 3:
			posy = [-y, y, -y, y]
		else:
			posy = [-y, y, y, -y]
	Frames = []
	for omega in Omegas:
		thisFrameNr = frameNumber+int(round(omega/omegaStep))
		if thisFrameNr < 1:
			thisFrameNr = totalFrames + thisFrameNr
		Frames.append(thisFrameNr)
	return Frames, posy, posz

def onclick(event):
	global floatText
	global ringRads
	global LsdMean
	global ringNrs
	ix, iy = event.xdata, event.ydata
	EtaR = CalcEtaAngleRad(ix,iy)
	Ttheta = rad2deg*math.atan(EtaR[1]/LsdMean)
	Frames, posy, posz = calcFriedelFrames(ix,iy)
	frameStr = "FriedelPairs Info:\nFrameNr, PosHor PosVer (px):"
	for nr in range(0,4):
		strAdd = "\n%d, %0.0f, %0.0f"%(Frames[nr],posy[nr],posz[nr])
		frameStr = frameStr + strAdd
	ringRad = EtaR[1]
	minDiff = 1e15
	for i in range(len(ringRads)):
		rad = ringRads[i]
		ringnr = ringNrs[i]
		if fabs(rad -ringRad) < minDiff:
			minDiff = fabs(rad -ringRad)
			bestRad = rad
			bestRing = ringnr
	strAdd2 = "\nRingNumber: %d\n"%(bestRing)
	frameStr = frameStr + strAdd2
	outText = "Eta: %0.2f degrees\nRad: %0.2f microns\n2Theta: %0.4f\nCoordinates (px): %0.0f, %0.0f\n%s-----------"%(EtaR[0],EtaR[1],Ttheta,ix,iy,frameStr)
	print outText
	if floatText:
		floatText.remove()
	floatText = a.text(ix,iy,outText,fontsize=10)
	canvas.draw()

a = figur.add_subplot(111,aspect='equal')
neighborhood = generate_binary_structure(2,2)

# Read ParamFN, get parameters
os.system(hklGenPath+paramFN)
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
paramFile = open(paramFN,'r')
paramContents = paramFile.readlines()
Lsd = []
BC = []
ts = []
wedge = 0
for line in paramContents:
	if 'DetParams ' in line:
		Lsd.append(float(line.split()[1]))
		BC.append([float(line.split()[2]),float(line.split()[3])])
		ts.append([float(line.split()[4]),float(line.split()[5]),float(line.split()[6])])
	if 'px ' in line:
		px = float(line.split()[1])
	if 'Wavelength ' in line:
		wl = float(line.split()[1])
	if 'Wedge ' in line:
		wedge = float(line.split()[1])
	if 'Mask ' in line:
		MaskFN = line.split()[1]
LsdMean = 0
for i in range(0,nDetectors):
	LsdMean = LsdMean + Lsd[i]/nDetectors

# Read Mask
BigDetSize = int(MaskFN.split('Size')[1].split('x')[0])
MaskF = open(MaskFN,"r")
mask = np.fromfile(MaskF,dtype=np.uint16,count=BigDetSize*BigDetSize)
MaskF.close()
mask = np.reshape(mask,(BigDetSize,BigDetSize))
mask = np.transpose(mask)

# Get nFrames to calculate fileNum and framesToSkip 
fileName = folder + fileStem + '_{:05d}'.format(fileNumber) + '.ge' + str(1)
darkFileName = folder + darkName + '.ge' + str(1)
size = os.path.getsize(fileName)
nFrames = (size-8192)/(2*NrPixels*NrPixels)

floatText = None
AllMaximaVals = None
canvas.mpl_connect('button_press_event',onclick)
initplot = 1
frameNumber = initFrameNumber
draw_plot(initFrameNumber,ringRads)

toolbar = NavigationToolbar2TkAgg(canvas,root)
toolbar.update()
canvas._tkcanvas.pack(side=Tk.TOP,fill=Tk.BOTH)

def _quit():
	root.quit()
	root.destroy()

button = Tk.Button(master=root,text='Quit',command=_quit)
button.pack(side=Tk.LEFT)

Tk.Label(master=root,text="Frame Number").pack(side=Tk.LEFT)
r = Tk.StringVar() 
r.set(str(initFrameNumber))
e1 = Tk.Entry(master=root,textvariable=r)
e1.pack(side=Tk.LEFT)
e1.focus_set()
e1.bind('<Return>',update_plot)

button2 = Tk.Button(master=root,text='Load',command=plot_updater)
button2.pack(side=Tk.LEFT)

Tk.mainloop()
'''
