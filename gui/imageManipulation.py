#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import Tkinter as Tk
import sys
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import time
import os
import glob
import tkFileDialog
import math
import scipy
from scipy.misc import imsave
from subprocess import Popen, PIPE, STDOUT, call
from multiprocessing.dummy import Pool
import multiprocessing

deg2rad = 0.0174532925199433
rad2deg = 57.2957795130823

def _quit():
	root.quit()
	root.destroy()

def selectFile():
	return tkFileDialog.askopenfilename()

def firstFileSelector():
	global fileStemVar, padding
	global extvar, folderVar
	global firstFileNrVar, outFolderVar
	firstfilefullpath = selectFile()
	folder = os.path.dirname(firstfilefullpath) + '/'
	fullfilename = firstfilefullpath.split('/')[-1].split('.')[0]
	extvar.set('.' + firstfilefullpath.split('.')[-1])
	fileStemVar.set('_'.join(fullfilename.split('_')[:-1]))
	firstFileNumber = int(fullfilename.split('_')[-1])
	firstFileNrVar.set(firstFileNumber)
	padding = len(fullfilename.split('_')[-1])
	folderVar.set(folder)
	outFolderVar.set(folder)

def darkFileSelector():
	global darkfilefullpathVar,doDark
	darkfilefullpathVar.set(selectFile())
	doDark.set(1)

def getfn(fstem,fnum):
	return folderVar.get() +'/'+ fstem + '_' + str(fnum).zfill(padding) + extvar.get()

def getoutfn(fstem,fnum):
	return outFolderVar.get() +'/'+ fstem + '_' + str(fnum).zfill(padding) + extvar.get()

def getDarkImage(fn,bytesToSkip):
	dataDark = np.zeros(NrPixels*NrPixels)
	statinfo = os.stat(fn)
	nFramesPerFile = (statinfo.st_size - 8192)/(2*NrPixels*NrPixels)
	f = open(fn,'rb')
	f.seek(bytesToSkip,os.SEEK_SET)
	for framenr in range(nFramesPerFile):
		data = np.fromfile(f,dtype=np.uint16,count=(NrPixels*NrPixels))
		data = data.astype(float)
		dataDark = np.add(dataDark,data)
	f.close()
	dataDark = dataDark/nFramesPerFile
	return dataDark

def saveFile(arr,fname,fileTypeWrite):
	# Look at renormalization for 1 and 2
	if fileTypeWrite == 1: # GE output to uint16
		arr += -(np.min(arr))
		arr /= np.max(arr)/(65535)
		arr = arr.astype(np.uint16)
		with open(fname,'wb') as f:
			np.array(header).tofile(f)
			np.array(arr).tofile(f)
	elif fileTypeWrite == 2:
		# Rescale arr to shape.
		arr2 = np.copy(arr)
		arr2 += -(np.min(arr2))
		arr2 /= np.max(arr2)/(255)
		arr2 = arr2.astype(np.uint8)
		imsave(fname+'.tif',np.reshape(arr2,(NrPixels,NrPixels)))
	elif fileTypeWrite == 3: # BatchCorr output
		with open(fname,'wb') as f:
			np.array(arr).tofile(f)

def processFile(fnr): # fnr is the line number in the fnames.txt file
	global header
	f = open('imparams.txt','r')
	params = f.readlines()
	doBadProcessing = int(params[0].rstrip())
	darkProcessing = int(params[1].rstrip())
	allFrames = int(params[2].rstrip())
	sumWrite = int(params[3].rstrip())
	meanWrite = int(params[4].rstrip())
	maxWrite = int(params[5].rstrip())
	fileTypeWrite = int(params[6].rstrip())
	doIntegration = int(params[7].rstrip())
	f.close()
	f = open('fnames.txt','r')
	fnames = f.readlines()
	f.close()
	fn = fnames[fnr].rstrip()
	f = open('outputFnames.txt','r')
	outfnames = f.readlines()
	f.close()
	outfn = outfnames[fnr].rstrip()
	dark = np.zeros(NrPixels*NrPixels)
	if doBadProcessing is 1:
		detNr = fn[-1]
		badF = open(os.path.expanduser('~')+'/opt/MIDAS/gui/GEBad/BadImg.ge'+detNr,'rb')
		badF.seek(8192,os.SEEK_SET)
		badData = np.fromfile(badF,dtype=np.uint16,count=NrPixels*NrPixels)
		badData = np.nonzero(badData)
		nBadData = len(badData[0])
	if darkProcessing is 1:
		darkfn = darkfilefullpath[:-1] + fn[-1]
		dark = getDarkImage(darkfn,8192)
	statinfo = os.stat(fn)
	nFramesPerFile = (statinfo.st_size - 8192)/(2*NrPixels*NrPixels)
	f = open(fn,'rb')
	header = np.fromfile(f,dtype=np.uint8,count=8192)
	sumArr = np.zeros(NrPixels*NrPixels)
	aveArr = np.zeros(NrPixels*NrPixels)
	maxArr = np.zeros(NrPixels*NrPixels)
	for frameNr in range(nFramesPerFile):
		bytesToSkip = 8192 + frameNr*(2*NrPixels*NrPixels)
		f.seek(bytesToSkip,os.SEEK_SET)
		data = np.fromfile(f,dtype=np.uint16,count=NrPixels*NrPixels)
		data = data.astype(float)
		if doBadProcessing:
			### For each non-zero element in badData, correct with mean of 
			### neighbours, no precaution for edge pixels for now
			for idx in range(nBadData):
				data[idx] = (data[idx-1] + data[idx-NrPixels] + data[idx+1] + data[idx+NrPixels])/4
		corr = np.subtract(data,dark)
		if allFrames:
			writefn = outfn+'.frame.'+str(frameNr)+'.cor'
			saveFile(corr,writefn,fileTypeWrite)
			if doIntegration is 1:
				call([os.path.expanduser('~')+'/opt/MIDAS/FF_HEDM/bin/Integrator','ps_midas.txt',writefn])
		if sumWrite:
			sumArr = np.add(sumArr,corr)
		if maxWrite:
			maxArr = np.maximum(maxArr,corr)
	if sumWrite:
		writefn = outfn+'.sum'
		saveFile(sumArr,writefn,fileTypeWrite)
		if doIntegration is 1:
			call([os.path.expanduser('~')+'/opt/MIDAS/FF_HEDM/bin/Integrator','ps_midas.txt',writefn])
	if meanWrite:
		writefn = outfn+'.ave'
		saveFile(sumArr/nFramesPerFile,writefn,fileTypeWrite)
		if doIntegration is 1:
			call([os.path.expanduser('~')+'/opt/MIDAS/FF_HEDM/bin/Integrator','ps_midas.txt',writefn])
	if maxWrite:
		writefn = outfn+'.max'
		saveFile(maxArr,writefn,fileTypeWrite)
		if doIntegration is 1:
			call([os.path.expanduser('~')+'/opt/MIDAS/FF_HEDM/bin/Integrator','ps_midas.txt',writefn])

def processImages():
	global darkfilefullpath
	starttime = time.time()
	f = open('imparams.txt','w')
	f.write(str(doBad.get())+'\n')
	f.write(str(doDark.get())+'\n')
	f.write(str(doAllFrames.get())+'\n')
	f.write(str(doSum.get())+'\n')
	f.write(str(doMean.get())+'\n')
	f.write(str(doMax.get())+'\n')
	f.write(str(fileTypeVar.get())+'\n')
	f.write(str(integrateVar.get())+'\n') # Not to do integration
	f.close()
	pool = Pool(processes=multiprocessing.cpu_count())
	# We create a fnames.txt file with filenames for each file to process
	f = open('fnames.txt','w')
	fout = open('outputFnames.txt','w')
	fileStem = fileStemVar.get()
	if nFilesVar.get() is not 0:
		nrFiles = nFilesVar.get()
		startNr = firstFileNrVar.get()
		if doHydra.get():
			for fnr in range(startNr,startNr+nrFiles):
				fnTemp = getfn(fileStem,fnr)
				outfnTemp = getoutfn(fileStem,fnr)
				for detNr in range(1,5):
					f.write(fnTemp[:-1]+str(detNr)+'\n')
					fout.write(fnTemp[:-1]+str(detNr)+'\n')
		else:
			for fnr in range(startNr,startNr+nrFiles):
				f.write(getfn(fileStem,fnr)+'\n')
				fout.write(getoutfn(fileStem,fnr)+'\n')
	else:
		## Do a pattern search.
		if fileStem is not '*':
			fnames = glob.glob(fileStem+'*')
		else:
			fnames = glob.glob('*.ge*')
		for fname in fnames:
			f.write(folderVar.get()+'/'+fname+'\n')
			fout.write(outFolderVar.get()+'/'+fname+'\n')
		nrFiles = len(fnames)
	if not os.path.exists(outFolderVar.get()):
		os.makedirs(outFolderVar.get())
	darkfilefullpath = darkfilefullpathVar.get()
	f.close()
	fout.close()
	pipout = range(nrFiles) #firstFileNrVar.get(),firstFileNrVar.get()+nFilesVar.get())
	results = pool.map(processFile,pipout)
	os.remove('imparams.txt')
	os.remove('fnames.txt')
	os.remove('outputFnames.txt')
	if integrateVar.get() is 1:
		os.remove('ps_midas.txt')
		os.remove('Map.bin')
		os.remove('nMap.bin')
	print time.time() - starttime

def acceptParameters():
	global topIntegrateParametersSelection
	topIntegrateParametersSelection.destroy()
	# Write out all the parameters
	f = open('ps_midas.txt','w')
	f.write('EtaBinSize '+str(EtaBinSizeVar.get())+'\n')
	f.write('RBinSize '+str(RBinSizeVar.get())+'\n')
	f.write('RMax '+str(RMaxVar.get())+'\n')
	f.write('RMin '+str(RMinVar.get())+'\n')
	f.write('EtaMax '+str(EtaMaxVar.get())+'\n')
	f.write('EtaMin '+str(EtaMinVar.get())+'\n')
	f.write('NrPixels '+str(NrPixelsVar.get())+'\n')
	f.write('Normalize '+str(NormalizeVar.get())+'\n')
	f.write('FloatFile '+str(FloatFileVar.get())+'\n')
	f.write('tx '+str(txVar.get())+'\n')
	f.write('ty '+str(tyVar.get())+'\n')
	f.write('tz '+str(tzVar.get())+'\n')
	f.write('px '+str(pxVar.get())+'\n')
	f.write('BC '+str(yBCVar.get())+' '+str(zBCVar.get())+'\n')
	f.write('Lsd '+str(LsdVar.get())+'\n')
	f.write('RhoD '+str(RhoDVar.get())+'\n')
	f.write('p0 '+str(p0Var.get())+'\n')
	f.write('p1 '+str(p1Var.get())+'\n')
	f.write('p2 '+str(p2Var.get())+'\n')
	f.close()
	# call DetectorMapper
	cmdname = os.path.expanduser('~')+'/opt/MIDAS/FF_HEDM/bin/DetectorMapper'
	call([cmdname,'ps_midas.txt'])
	# call processImages
	processImages()

def integrate():
	global EtaBinSizeVar, RBinSizeVar, RMaxVar, RMinVar, EtaMaxVar, EtaMinVar
	global NrPixelsVar, NormalizeVar, FloatFileVar, txVar, tyVar, tzVar
	global pxVar, yBCVar, zBCVar, LsdVar, RhoDVar, p0Var, p1Var, p2Var
	global topIntegrateParametersSelection, integrateVar, fileTypeVar
	fileTypeVar.set(1)
	integrateVar.set(1)
	EtaBinSizeVar = Tk.DoubleVar()
	RBinSizeVar = Tk.DoubleVar()
	RMaxVar = Tk.DoubleVar()
	RMinVar = Tk.DoubleVar()
	EtaMaxVar = Tk.DoubleVar()
	EtaMinVar = Tk.DoubleVar()
	NrPixelsVar = Tk.IntVar()
	NormalizeVar = Tk.IntVar()
	FloatFileVar = Tk.IntVar()
	txVar = Tk.DoubleVar()
	tyVar = Tk.DoubleVar()
	tzVar = Tk.DoubleVar()
	pxVar = Tk.DoubleVar()
	yBCVar = Tk.DoubleVar()
	zBCVar = Tk.DoubleVar()
	LsdVar = Tk.DoubleVar()
	RhoDVar = Tk.DoubleVar()
	p0Var = Tk.DoubleVar()
	p1Var = Tk.DoubleVar()
	p2Var = Tk.DoubleVar()
	EtaBinSizeVar.set(5.0)
	RBinSizeVar.set(1.0)
	RMaxVar.set(1024.0)
	RMinVar.set(10.0)
	EtaMaxVar.set(180.0)
	EtaMinVar.set(-180.0)
	NrPixelsVar.set(2048)
	NormalizeVar.set(1)
	FloatFileVar.set(0)
	txVar.set(0.0)
	tyVar.set(0.0)
	tzVar.set(0.0)
	pxVar.set(200.0)
	yBCVar.set(1024.0)
	zBCVar.set(1024.0)
	LsdVar.set(1000000.0)
	RhoDVar.set(200000.0)
	p0Var.set(0.0)
	p1Var.set(0.0)
	p2Var.set(0.0)
	topIntegrateParametersSelection = Tk.Toplevel()
	topIntegrateParametersSelection.title("Select parameters for integration")
	Tk.Label(master=topIntegrateParametersSelection,text=
			"Please select the parameters for integration").grid(row=1,
			column=1,columnspan=10)
	Tk.Label(master=topIntegrateParametersSelection,
			text="EtaBinSize (Deg)").grid(row=2,column=1)
	Tk.Entry(master=topIntegrateParametersSelection,
			textvariable=EtaBinSizeVar).grid(row=2,column=2)
	Tk.Label(master=topIntegrateParametersSelection,
			text="   RBinSize (px)").grid(row=2,column=3)
	Tk.Entry(master=topIntegrateParametersSelection,
			textvariable=RBinSizeVar).grid(row=2,column=4)
	Tk.Label(master=topIntegrateParametersSelection,
			text="       RMax (px)").grid(row=2,column=5)
	Tk.Entry(master=topIntegrateParametersSelection,
			textvariable=RMaxVar).grid(row=2,column=6)
	Tk.Label(master=topIntegrateParametersSelection,
			text="       RMin (px)").grid(row=2,column=7)
	Tk.Entry(master=topIntegrateParametersSelection,
			textvariable=RMinVar).grid(row=2,column=8)
	Tk.Label(master=topIntegrateParametersSelection,
			text="    EtaMax (Deg)").grid(row=3,column=1)
	Tk.Entry(master=topIntegrateParametersSelection,
			textvariable=EtaMaxVar).grid(row=3,column=2)
	Tk.Label(master=topIntegrateParametersSelection,
			text="    EtaMin (Deg)").grid(row=3,column=3)
	Tk.Entry(master=topIntegrateParametersSelection,
			textvariable=EtaMinVar).grid(row=3,column=4)
	Tk.Label(master=topIntegrateParametersSelection,
			text="       NrPixels").grid(row=3,column=5)
	Tk.Entry(master=topIntegrateParametersSelection,
			textvariable=NrPixelsVar).grid(row=3,column=6)
	Tk.Label(master=topIntegrateParametersSelection,
			text="        TX (Deg)").grid(row=3,column=7)
	Tk.Entry(master=topIntegrateParametersSelection,
			textvariable=txVar).grid(row=3,column=8)
	Tk.Label(master=topIntegrateParametersSelection,
			text="        TY (Deg)").grid(row=4,column=1)
	Tk.Entry(master=topIntegrateParametersSelection,
			textvariable=tyVar).grid(row=4,column=2)
	Tk.Label(master=topIntegrateParametersSelection,
			text="        TZ (Deg)").grid(row=4,column=3)
	Tk.Entry(master=topIntegrateParametersSelection,
			textvariable=tzVar).grid(row=4,column=4)
	Tk.Label(master=topIntegrateParametersSelection,
			text="  PixelSize (um)").grid(row=4,column=5)
	Tk.Entry(master=topIntegrateParametersSelection,
			textvariable=pxVar).grid(row=4,column=6)
	Tk.Label(master=topIntegrateParametersSelection,
			text="        YBC (px)").grid(row=4,column=7)
	Tk.Entry(master=topIntegrateParametersSelection,
			textvariable=yBCVar).grid(row=4,column=8)
	Tk.Label(master=topIntegrateParametersSelection,
			text="        ZBC (px)").grid(row=5,column=1)
	Tk.Entry(master=topIntegrateParametersSelection,
			textvariable=zBCVar).grid(row=5,column=2)
	Tk.Label(master=topIntegrateParametersSelection,
			text="   Distance (um)").grid(row=5,column=3)
	Tk.Entry(master=topIntegrateParametersSelection,
			textvariable=LsdVar).grid(row=5,column=4)
	Tk.Label(master=topIntegrateParametersSelection,
			text="       RhoD (um)").grid(row=5,column=5)
	Tk.Entry(master=topIntegrateParametersSelection,
			textvariable=RhoDVar).grid(row=5,column=6)
	Tk.Label(master=topIntegrateParametersSelection,
			text="             P0").grid(row=5,column=7)
	Tk.Entry(master=topIntegrateParametersSelection,
			textvariable=p0Var).grid(row=5,column=8)
	Tk.Label(master=topIntegrateParametersSelection,
			text="             P1").grid(row=6,column=1)
	Tk.Entry(master=topIntegrateParametersSelection,
			textvariable=p1Var).grid(row=6,column=2)
	Tk.Label(master=topIntegrateParametersSelection,
			text="             P2").grid(row=6,column=3)
	Tk.Entry(master=topIntegrateParametersSelection,
			textvariable=p2Var).grid(row=6,column=4)
	Tk.Checkbutton(master=topIntegrateParametersSelection,
		text="NormalizeIntensity",variable=NormalizeVar).grid(row=6,
		column=5,columnspan=2)
	Tk.Checkbutton(master=topIntegrateParametersSelection,
		text="FloatFileFormat",variable=FloatFileVar).grid(row=6,
		column=7,columnspan=2)
	Tk.Button(master=topIntegrateParametersSelection,
		text="Continue",command=acceptParameters).grid(row=20,
		column=1,columnspan=10)

root = Tk.Tk()
root.wm_title("Image Manipulation Software, MIDAS, v0.1 Dt. 2018/01/30 hsharma@anl.gov")
figur = Figure(figsize=(14.5,8.5),dpi=100)
canvas = FigureCanvasTkAgg(figur,master=root)
a = figur.add_subplot(111,aspect='equal')
a.title.set_text("Selected Image")

firstFileNrVar = Tk.IntVar()
firstFileNrVar.set(0)
folder = ''
folderVar = Tk.StringVar()
outfolder = ''
outFolderVar = Tk.StringVar()
folderVar.set(folder)
NrPixels = 2048
nFramesPerFile = 240
nFilesVar = Tk.IntVar()
nFilesVar.set(0)
doDark = Tk.IntVar()
doDark.set(0)
doAllFrames = Tk.IntVar()
doAllFrames.set(0)
doSum = Tk.IntVar()
doSum.set(0)
doMean = Tk.IntVar()
doMean.set(0)
doMax = Tk.IntVar()
doMax.set(0)
fileStem = ''
fileStemVar = Tk.StringVar()
fileStemVar.set('')
darkfilefullpathVar=Tk.StringVar()
extvar = Tk.StringVar()
extvar.set('')
doBad = Tk.IntVar()
doBad.set(0)
doHydra = Tk.IntVar()
doHydra.set(0)
integrateVar = Tk.IntVar()
integrateVar.set(0)

rowFigSize = 3
colFigSize = 3

Tk.Label(master=root,text="Image pre-processing and conversion using MIDAS",
	font=("Helvetica",20)).grid(row=0,column=0,rowspan=rowFigSize,
	columnspan=colFigSize,sticky=Tk.W+Tk.E+Tk.N+Tk.S)

leftSideFrame = Tk.Frame(root)
leftSideFrame.grid(row=rowFigSize+1,column=0,rowspan=7,sticky=Tk.W)
firstRowFrame = Tk.Frame(root)
firstRowFrame.grid(row=rowFigSize+1,column=1,columnspan=2,sticky=Tk.W)
secondRowFrame = Tk.Frame(root)
secondRowFrame.grid(row=rowFigSize+2,column=1,sticky=Tk.W)
midRowFrame = Tk.Frame(root)
midRowFrame.grid(row=rowFigSize+3,column=1,sticky=Tk.W)
twoThirdRowFrame = Tk.Frame(root)
twoThirdRowFrame.grid(row=rowFigSize+4,column=1,sticky=Tk.W)
threeThirdRowFrame = Tk.Frame(root)
threeThirdRowFrame.grid(row=rowFigSize+5,column=1,sticky=Tk.W)
thirdRowFrame = Tk.Frame(root)
thirdRowFrame.grid(row=rowFigSize+6,column=1,sticky=Tk.W)
fourthRowFrame = Tk.Frame(root)
fourthRowFrame.grid(row=rowFigSize+7,column=1,sticky=Tk.W)
fifthRowFrame = Tk.Frame(root)
fifthRowFrame.grid(row=rowFigSize+8,column=1,sticky=Tk.W)
rightSideFrame = Tk.Frame(root)
rightSideFrame.grid(row=rowFigSize+2,column=2,rowspan=7,sticky=Tk.W)
bottomFrame = Tk.Frame(root)
bottomFrame.grid(row=rowFigSize+9,column=0,columnspan=4,sticky=Tk.W)

Tk.Button(master=leftSideFrame,text='Quit',command=_quit,
	font=("Helvetica",20)).grid(row=0,column=0,padx=10,pady=10)

Tk.Button(master=leftSideFrame,text='Calibrate',command=_quit,
	font=("Helvetica",20)).grid(row=1,column=0,padx=10,pady=10)

Tk.Button(master=firstRowFrame,text='SelectFirstFile',
	command=firstFileSelector,font=("Helvetica",12)).grid(row=1,
	column=0,sticky=Tk.W)

Tk.Button(master=firstRowFrame,text='SelectDarkFile',
	command=darkFileSelector,font=("Helvetica",12)).grid(row=1,
	column=1,sticky=Tk.W)

Tk.Checkbutton(master=firstRowFrame,text="Subtract Dark",
	variable=doDark).grid(row=1,column=2,sticky=Tk.W)

Tk.Checkbutton(master=firstRowFrame,text="Correct BadPixels",
	variable=doBad).grid(row=1,column=3,sticky=Tk.W)

Tk.Checkbutton(master=firstRowFrame,text="Hydra",
	variable=doHydra).grid(row=1,column=4,sticky=Tk.W)

###### Folder info
Tk.Label(master=midRowFrame,text="Input Folder ").grid(row=1,column=0,sticky=Tk.W)
Tk.Entry(master=midRowFrame,textvariable=folderVar,width=66).grid(row=1,column=1,sticky=Tk.W)

###### Rest info
Tk.Label(master=secondRowFrame,text='FileStem').grid(row=1,column=0,sticky=Tk.W)
Tk.Entry(master=secondRowFrame,textvariable=fileStemVar,width=40).grid(row=1,column=1,sticky=Tk.W)

Tk.Label(master=secondRowFrame,text='FirstFileNr').grid(row=1,column=2,sticky=Tk.W)
Tk.Entry(master=secondRowFrame,textvariable=firstFileNrVar,width=6).grid(row=1,column=3,sticky=Tk.W)

Tk.Label(master=secondRowFrame,text='nFiles').grid(row=1,column=4,sticky=Tk.W)
Tk.Entry(master=secondRowFrame,textvariable=nFilesVar,width=5).grid(row=1,column=5,sticky=Tk.W)

###### Output folder info
Tk.Label(master=twoThirdRowFrame,text="Output Folder").grid(row=1,column=0,sticky=Tk.W)
Tk.Entry(master=twoThirdRowFrame,textvariable=outFolderVar,width=66).grid(row=1,column=1,sticky=Tk.W)

###### Dark file info
Tk.Label(master=threeThirdRowFrame,text="Dark Filename").grid(row=1,column=0,sticky=Tk.W)
Tk.Entry(master=threeThirdRowFrame,textvariable=darkfilefullpathVar,width=66).grid(row=1,column=1,sticky=Tk.W)

###### Output Options: allFrames, sum, Ave, Mean
Lb1 = Tk.Label(master=thirdRowFrame,text="Output Options:  ")
Lb1.grid(row=1,column=0,sticky=Tk.W)
Lb1.config(bg="gray")

Tk.Checkbutton(master=thirdRowFrame,text="WriteAllFrames",
	variable=doAllFrames).grid(row=1,column=1,sticky=Tk.W)

Tk.Checkbutton(master=thirdRowFrame,text="WriteSum",
	variable=doSum).grid(row=1,column=2,sticky=Tk.W)

Tk.Checkbutton(master=thirdRowFrame,text="WriteMean",
	variable=doMean).grid(row=1,column=3,sticky=Tk.W)

Tk.Checkbutton(master=thirdRowFrame,text="WriteMax",
	variable=doMax).grid(row=1,column=4,sticky=Tk.W)

###### Saveas types
Lb2 = Tk.Label(master=fourthRowFrame,text="Output Filetype: ")
Lb2.grid(row=1,column=0,sticky=Tk.W)
Lb2.config(bg="yellow")

FILEOPTS = [("GE",1),("TIFF",2),("BATCHCORR",3)]
fileTypeVar = Tk.IntVar()
fileTypeVar.set(1)

for text, val in FILEOPTS:
	Tk.Radiobutton(master=fourthRowFrame,text=text,variable=fileTypeVar,
		value=val,font=("Helvetica 16 bold")).grid(row=1,column=val,sticky=Tk.W)

###### Run processing
Tk.Button(master=rightSideFrame,text="Integrate",
	command=integrate,font=('Helvetica',20)).grid(row=0,
	column=0,padx=10,pady=10)

Tk.Button(master=rightSideFrame,text="Process Images",
	command=processImages,font=("Helvetica",20)).grid(row=1,
	column=0,padx=10,pady=10)


###### Show some help
Tk.Label(master=bottomFrame,text='NOTE:',font=('Helvetica 16 bold')).grid(row=0,column=0,sticky=Tk.W)
Tk.Label(master=bottomFrame,text='1. nFiles=0 would process all the files starting with FileStem name.',
	font=('Helvetica 16 bold')).grid(row=1,column=0,sticky=Tk.W)
Tk.Label(master=bottomFrame,text='2. Put FileStem to * to process the whole folder with *.ge* extension, but not in subfolders.',
	font=('Helvetica 16 bold')).grid(row=2,column=0,sticky=Tk.W)
Tk.Label(master=bottomFrame,text='3. For HYDRA, select a dark file for one of the panels.',
	font=('Helvetica 16 bold')).grid(row=3,column=0,sticky=Tk.W)
Tk.Label(master=bottomFrame,text='4. For integration, output filetype will always be reset to GE.',
	font=('Helvetica 16 bold')).grid(row=4,column=0,sticky=Tk.W)

Tk.mainloop()
