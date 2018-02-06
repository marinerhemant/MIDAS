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
from subprocess import Popen, PIPE, STDOUT
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
	global firstFileNrVar
	firstfilefullpath = selectFile()
	folder = os.path.dirname(firstfilefullpath) + '/'
	fullfilename = firstfilefullpath.split('/')[-1].split('.')[0]
	extvar.set('.' + firstfilefullpath.split('.')[-1])
	fileStemVar.set('_'.join(fullfilename.split('_')[:-1]))
	firstFileNumber = int(fullfilename.split('_')[-1])
	firstFileNrVar.set(firstFileNumber)
	padding = len(fullfilename.split('_')[-1])
	folderVar.set(folder)

def darkFileSelector():
	global darkfilefullpath,doDark
	darkfilefullpath = selectFile()
	doDark.set(1)

def getfn(fstem,fnum):
	return folderVar.get() +'/'+ fstem + '_' + str(fnum).zfill(padding) + extvar.get()

def getDarkImage(fn,bytesToSkip):
	global dark
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
	dark = dataDark

def saveFile(arr,fname,fileTypeWrite):
	if fileTypeWrite == 1: # GE output to uint16
		arr += -(np.min(arr))
		arr /= np.max(arr)/(65535)
		arr = arr.astype(np.uint16)
		with open(fname,'wb') as f:
			np.array(header).tofile(f)
			np.array(arr).tofile(f)
	elif fileTypeWrite == 2:
		imsave(fname+'.tif',np.reshape(arr,(NrPixels,NrPixels)))
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
	f.close()
	f = open('fnames.txt','r')
	fnames = f.readlines()
	fn = fnames[fnr].rstrip()
	dark = np.zeros(NrPixels*NrPixels)
	if doBadProcessing:
		detNr = fn[-1]
		badF = open(os.path.expanduser('~')+'/opt/MIDAS/gui/GEBad/BadImg.ge'+detNr,'rb')
		badF.seek(8192,os.SEEK_SET)
		badData = np.fromfile(badF,dtype=np.uint16,count=NrPixels*NrPixels)
		badData = np.nonzero(badData)
		nBadData = len(badData[0])
		print nBadData
	if darkProcessing:
		darkfn = darkfilefullpath[:-1] + fn[-1]
		darkfn = darkfilefullpath
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
			saveFile(corr,fn+'.frame.'+str(frameNr)+'.cor',fileTypeWrite)
		if sumWrite:
			sumArr = np.add(sumArr,corr)
		if maxWrite:
			maxArr = np.maximum(maxArr,corr)
	if sumWrite:
		saveFile(sumArr,fn+'.sum',fileTypeWrite)
	if meanWrite:
		saveFile(sumArr/nFramesPerFile,fn+'.ave',fileTypeWrite)
	if maxWrite:
		saveFile(maxArr,fn+'.max',fileTypeWrite)

def processImages():
	starttime = time.time()
	f = open('imparams.txt','w')
	f.write(str(doBad.get())+'\n')
	f.write(str(doDark.get())+'\n')
	f.write(str(doAllFrames.get())+'\n')
	f.write(str(doSum.get())+'\n')
	f.write(str(doMean.get())+'\n')
	f.write(str(doMax.get())+'\n')
	f.write(str(fileTypeVar.get())+'\n')
	f.close()
	pool = Pool(processes=multiprocessing.cpu_count())
	# We create a fnames.txt file with filenames for each file to process
	f = open('fnames.txt','w')
	fileStem = fileStemVar.get()
	if nFilesVar.get() is not 0:
		nrFiles = nFilesVar.get()
		startNr = firstFileNrVar.get()
		if doHydra.get():
			for fnr in range(startNr,startNr+nrFiles):
				fnTemp = getfn(fileStem,fnr)
				for detNr in range(1,5):
					f.write(fnTemp[:-1]+str(detNr)+'\n')
		else:
			for fnr in range(startNr,startNr+nrFiles):
				f.write(getfn(fileStem,fnr)+'\n')
	else:
		## Do a pattern search.
		if fileStem is not '*':
			fnames = glob.glob(fileStem+'*')
		else:
			fnames = glob.glob('*.ge*')
		for fname in fnames:
			f.write(folderVar.get()+'/'+fname+'\n')
		nrFiles = len(fnames)
	f.close()
	pipout = range(nrFiles) #firstFileNrVar.get(),firstFileNrVar.get()+nFilesVar.get())
	results = pool.map(processFile,pipout)
	print time.time() - starttime

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
darkilefullpath=''
extvar = Tk.StringVar()
extvar.set('')
doBad = Tk.IntVar()
doBad.set(0)
doHydra = Tk.IntVar()
doHydra.set(0)

rowFigSize = 3
colFigSize = 3

Tk.Label(master=root,text="Image pre-processing and conversion using MIDAS",
	font=("Helvetica",20)).grid(row=0,column=0,rowspan=rowFigSize,
	columnspan=colFigSize,sticky=Tk.W+Tk.E+Tk.N+Tk.S)

leftSideFrame = Tk.Frame(root)
leftSideFrame.grid(row=rowFigSize+2,column=0,rowspan=5,sticky=Tk.W)
firstRowFrame = Tk.Frame(root)
firstRowFrame.grid(row=rowFigSize+1,column=1,columnspan=2,sticky=Tk.W)
midRowFrame = Tk.Frame(root)
midRowFrame.grid(row=rowFigSize+2,column=1,sticky=Tk.W)
secondRowFrame = Tk.Frame(root)
secondRowFrame.grid(row=rowFigSize+3,column=1,sticky=Tk.W)
thirdRowFrame = Tk.Frame(root)
thirdRowFrame.grid(row=rowFigSize+4,column=1,sticky=Tk.W)
fourthRowFrame = Tk.Frame(root)
fourthRowFrame.grid(row=rowFigSize+5,column=1,sticky=Tk.W)
rightSideFrame = Tk.Frame(root)
rightSideFrame.grid(row=rowFigSize+2,column=2,rowspan=5,sticky=Tk.W)
bottomFrame = Tk.Frame(root)
bottomFrame.grid(row=rowFigSize+7,column=0,columnspan=3,sticky=Tk.W)

button = Tk.Button(master=leftSideFrame,text='Quit',command=_quit,font=("Helvetica",20))
button.grid(row=0,column=0,sticky=Tk.W,padx=10)

buttonFirstFile = Tk.Button(master=firstRowFrame,text='SelectFirstFile',
	command=firstFileSelector,font=("Helvetica",12))
buttonFirstFile.grid(row=1,column=0,sticky=Tk.W)

buttonDarkFile = Tk.Button(master=firstRowFrame,text='SelectDarkFile',
	command=darkFileSelector,font=("Helvetica",12))
buttonDarkFile.grid(row=1,column=1,sticky=Tk.W)

cDark = Tk.Checkbutton(master=firstRowFrame,text="Subtract Dark",variable=doDark)
cDark.grid(row=1,column=2,sticky=Tk.W)

cBad = Tk.Checkbutton(master=firstRowFrame,text="Correct BadPixels",variable=doBad)
cBad.grid(row=1,column=3,sticky=Tk.W)

cHydra = Tk.Checkbutton(master=firstRowFrame,text="Hydra",variable=doHydra)
cHydra.grid(row=1,column=4,sticky=Tk.W)

###### Folder info
Tk.Label(master=midRowFrame,text="Folder  ").grid(row=1,column=0,sticky=Tk.W)
eFolder= Tk.Entry(master=midRowFrame,textvariable=folderVar,width=71)
eFolder.grid(row=1,column=1,sticky=Tk.W)

###### Rest info
Tk.Label(master=secondRowFrame,text='FileStem').grid(row=1,column=0,sticky=Tk.W)
Tk.Entry(master=secondRowFrame,textvariable=fileStemVar,width=40).grid(row=1,column=1,sticky=Tk.W)

Tk.Label(master=secondRowFrame,text='FirstFileNr').grid(row=1,column=2,sticky=Tk.W)
efirstfile = Tk.Entry(master=secondRowFrame,textvariable=firstFileNrVar,width=6)
efirstfile.grid(row=1,column=3,sticky=Tk.W)

Tk.Label(master=secondRowFrame,text='nFiles').grid(row=1,column=4,sticky=Tk.W)
enFiles = Tk.Entry(master=secondRowFrame,textvariable=nFilesVar,width=5)
enFiles.grid(row=1,column=5,sticky=Tk.W)

###### Output Options: allFrames, sum, Ave, Mean
Lb1 = Tk.Label(master=thirdRowFrame,text="Output Options:  ")
Lb1.grid(row=1,column=0,sticky=Tk.W)
Lb1.config(bg="gray")

cAllFrames = Tk.Checkbutton(master=thirdRowFrame,text="WriteAllFrames",variable=doAllFrames)
cAllFrames.grid(row=1,column=1,sticky=Tk.W)

cSum = Tk.Checkbutton(master=thirdRowFrame,text="WriteSum",variable=doSum)
cSum.grid(row=1,column=2,sticky=Tk.W)

cMean = Tk.Checkbutton(master=thirdRowFrame,text="WriteMean",variable=doMean)
cMean.grid(row=1,column=3,sticky=Tk.W)

cMax = Tk.Checkbutton(master=thirdRowFrame,text="WriteMax",variable=doMax)
cMax.grid(row=1,column=4,sticky=Tk.W)

###### Saveas types
Lb2 = Tk.Label(master=fourthRowFrame,text="Select Filetype: ")
Lb2.grid(row=1,column=0,sticky=Tk.W)
Lb2.config(bg="yellow")

FILEOPTS = [("GE",1),("TIFF",2),("BATCHCORR",3)]
fileTypeVar = Tk.IntVar()
fileTypeVar.set(1)

for text, val in FILEOPTS:
	Tk.Radiobutton(master=fourthRowFrame,text=text,variable=fileTypeVar,
		value=val,font=("Helvetica 16 bold")).grid(row=1,column=val,sticky=Tk.W)

###### Run processing
buttonProcessImages = Tk.Button(master=rightSideFrame,text="Process Images",
	command=processImages,font=("Helvetica",20))
buttonProcessImages.grid(row=0,column=0,sticky=Tk.W,padx=10,pady=10)

###### Show some help
Tk.Label(master=bottomFrame,text='NOTE:',font=('Helvetica 16 bold')).grid(row=0,column=0,sticky=Tk.W)
Tk.Label(master=bottomFrame,text='1. nFiles=0 would process all the files starting with FileStem name.',
	font=('Helvetica 16 bold')).grid(row=1,column=0,sticky=Tk.W)
Tk.Label(master=bottomFrame,text='2. Put FileStem to * to process the whole folder with *.ge* extension, but not in subfolders.',
	font=('Helvetica 16 bold')).grid(row=2,column=0,sticky=Tk.W)
Tk.Label(master=bottomFrame,text='3. For HYDRA, select a dark file for one of the panels.',
	font=('Helvetica 16 bold')).grid(row=3,column=0,sticky=Tk.W)

Tk.mainloop()
