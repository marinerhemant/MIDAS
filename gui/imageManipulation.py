#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as Tk
import sys
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import time
import os
import glob
import tkinter.filedialog as tkFileDialog
import math
import scipy
import imageio
from subprocess import Popen, PIPE, STDOUT, call
from multiprocessing.dummy import Pool
import multiprocessing
from numba import jit
import PIL
import sys

# Try to import midas_config from utils
try:
    # Add utils directory to sys.path if not already there
    # Assuming this script is in gui/ and utils/ is in ../utils
    current_dir = os.path.dirname(os.path.abspath(__file__))
    utils_dir = os.path.join(os.path.dirname(current_dir), 'utils')
    if utils_dir not in sys.path:
        sys.path.append(utils_dir)
    import midas_config
except ImportError as e:
    print(f"Warning: Could not import midas_config: {e}")
    # Fallback or allow subsequent failures to handle it
    midas_config = None

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
	global GE1Var, GE2Var, GE3Var, GE4Var, GE5Var
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
	detNr = int(extvar.get()[-1])
	if detNr is 1:
		GE1Var.set(1)
	elif detNr is 2:
		GE2Var.set(1)
	elif detNr is 3:
		GE3Var.set(1)
	elif detNr is 4:
		GE4Var.set(1)
	elif detNr is 5:
		GE5Var.set(1)

def darkFileSelector():
	global darkfilefullpathVar,doDark
	darkfilefullpathVar.set(selectFile())
	doDark.set(1)

def getfn(fstem,fnum):
	return folderVar.get() +'/'+ fstem + '_' + str(fnum).zfill(padding) + extvar.get()

def getoutfn(fstem,fnum):
	return outFolderVar.get() +'/'+ fstem + '_' + str(fnum).zfill(padding) + extvar.get()

def getDarkImage(fn,bytesToSkip,doBadProcessing,badData):
	dataDark = np.zeros(NrPixels*NrPixels)
	statinfo = os.stat(fn)
	nFramesPerFile = int((statinfo.st_size - 8192)/(2*NrPixels*NrPixels))
	f = open(fn,'rb')
	f.seek(bytesToSkip,os.SEEK_SET)
	for framenr in range(nFramesPerFile):
		data = np.fromfile(f,dtype=np.uint16,count=(NrPixels*NrPixels))
		data = data.astype(float)
		if doBadProcessing:
			### For each non-zero element in badData, correct with mean of
			### neighbours, no precaution for edge pixels for now
			for idx in badData:
				if idx < NrPixels * (NrPixels-1) and idx >= NrPixels:
					data[idx] = (data[idx-1] + data[idx-NrPixels] + data[idx+1] + data[idx+NrPixels])/4
				else:
					data[idx] = 0
		dataDark = np.add(dataDark,data)
	f.close()
	dataDark = dataDark/nFramesPerFile
	return dataDark

#create an empty array with np.empty and then pass it as an argument, don't return
@jit('void(float64[:], float64[:], float64[:], float64[:], float64[:], int64[:])',nopython=True,nogil=True)
def calcFastIntegration2D(mapR, mapEta, Image, params, Result, nElements):
	NrPixels = int(params[0])
	RMin = params[1]
	RBin = params[2]
	EtaMin = params[3]
	EtaBin = params[4]
	nEtaBins = int(params[5])
	RMax = params[6]
	EtaMax = params[7]
	for i in range(int(NrPixels)*int(NrPixels)):
		RThis = mapR[i]
		EtaThis = mapEta[i]
		if RThis < RMin or RThis >= RMax or EtaThis < EtaMin or EtaThis >= EtaMax:
			continue
		RBinNr = int((int(RThis) - int(RMin))/int(RBin))
		EtaBinNr = int((int(EtaThis)-int(EtaMin))/int(EtaBin))
		Pos = RBinNr*nEtaBins + EtaBinNr
		Result[Pos] += Image[i]
		nElements[Pos] += 1

#create an empty array with np.empty and then pass it as an argument, don't return
@jit('void(float64[:], float64[:], float64[:], float64[:], int64[:])',nopython=True,nogil=True)
def calcFastIntegration1D(mapR, Image, params, Result, nElements):
	NrPixels = int(params[0])
	RMin = params[1]
	RBin = params[2]
	RMax = params[3]
	for i in range(NrPixels*NrPixels):
		RThis = mapR[i]
		if RThis < RMin or RThis >= RMax:
			continue
		RBinNr = int((int(RThis) - int(RMin))/int(RBin))
		Result[RBinNr] += Image[i]
		nElements[RBinNr] += 1

def saveFastIntegrate(arr, OneDOut, outfn):
	inittime = time.time()
	nRBins = int(math.ceil((RMax-RMin)/RBinSize))
	mapR = Rads.astype(float)
	if OneDOut is 0:
		mapEta = Etas.astype(float)
	Image = arr.astype(float)
	RArr = []
	outfile = open(outfn,'w')
	for i in range(nRBins):
		RArr.append(RMin + i*RBinSize + RBinSize/2.0)
	if OneDOut is 0:
		nEtaBins = int(math.ceil((EtaMax-EtaMin)/EtaBinSize))
		EtaArr = []
		for i in range(nEtaBins):
			EtaArr.append(EtaMin + i*EtaBinSize + EtaBinSize/2.0)
		Result = np.zeros(nRBins*nEtaBins,dtype=float)
		nElements = np.zeros(nRBins*nEtaBins,dtype=int)
		params = np.array([float(NrPixels),RMin,RBinSize,EtaMin,EtaBinSize,float(nEtaBins),RMax,EtaMax]).astype(float)
		calcFastIntegration2D(mapR,mapEta,Image,params,Result,nElements)
		outfile.write("Radius(px) Eta(degrees) Intensity(counts)\n")
		if normalizer is 0:
			nElements.fill(1)
		outArr = np.vstack((np.array(RArr*nEtaBins),np.array(EtaArr*nRBins)))
		outArr = np.vstack((outArr,np.divide(Result,nElements)))
		np.savetxt(outfile,outArr.T,fmt='%10.5f',delimiter=' ',newline='\n')
	else:
		Result = np.zeros(nRBins,dtype=float)
		nElements = np.zeros(nRBins,dtype=int)
		params = np.array([float(NrPixels),RMin,RBinSize,RMax]).astype(float)
		calcFastIntegration1D(mapR,Image,params,Result,nElements)
		outfile.write("Radius(px) Intensity(counts)\n")
		if normalizer is 0:
			nElements.fill(1)
		outArr = np.vstack((np.array(RArr),np.divide(Result,nElements)))
		np.savetxt(outfile,outArr.T,fmt='%10.5f',delimiter=' ',newline='\n')
	outfile.close()
	print(time.time() - inittime)

def transforms():
	txr = txVar.get()*deg2rad
	tyr = tyVar.get()*deg2rad
	tzr = tzVar.get()*deg2rad
	Rx = np.array([[1,0,0],[0,math.cos(txr),-math.sin(txr)],[0,math.sin(txr),math.cos(txr)]])
	Ry = np.array([[math.cos(tyr),0,math.sin(tyr)],[0,1,0],[-math.sin(tyr),0,math.cos(tyr)]])
	Rz = np.array([[math.cos(tzr),-math.sin(tzr),0],[math.sin(tzr),math.cos(tzr),0],[0,0,1]])
	return np.dot(Rx,np.dot(Ry,Rz))

def mapFastIntegration():
	global Rads, Etas
	yArr = -(np.array([np.arange(NrPixels)]*NrPixels).reshape((NrPixels*NrPixels)).astype(float)) + yBCVar.get()
	zArr = np.transpose(np.array([np.arange(NrPixels)]*NrPixels)).reshape((NrPixels*NrPixels)).astype(float) - zBCVar.get()
	xArr = np.zeros(NrPixels*NrPixels)
	ABC = np.array([xArr,yArr,zArr])
	TRs = transforms()
	ABCPr = np.dot(TRs,ABC)
	ABCPr[0,:] = ABCPr[0,:]+LsdVar.get()
	Lens = np.sqrt(np.add(np.multiply(ABCPr[1,:],ABCPr[1,:]),np.multiply(ABCPr[2,:],ABCPr[2:])))
	Rads = np.multiply(np.divide(LsdVar.get(),ABCPr[0,:]),Lens)
	Rads = np.reshape(Rads,(NrPixels*NrPixels))
	if OneDOutVar.get() is 0:
		Etas = rad2deg*np.arccos(np.divide(ABCPr[2:],Lens))
		Etas = np.reshape(Etas,(NrPixels*NrPixels))
		Etas[np.reshape(ABCPr[1,:]>0,(NrPixels*NrPixels))] *= -1

def saveFile(arr,fname,fileTypeWrite):
	if fileTypeWrite == 1: # GE output to uint16
		print([np.min(arr), np.max(arr), np.where(arr==np.min(arr))])
		arr += -(np.min(arr))
		arr /= np.max(arr)/(65535)
		arr = arr.astype(np.uint16)
		with open(fname,'wb') as f:
			np.array(header).tofile(f)
			np.array(arr).tofile(f)
	elif fileTypeWrite == 2:
		# Rescale arr to tiff(0--255).
		arr2 = np.copy(arr)
		arr2 += -(np.min(arr2))
		arr2 /= np.max(arr2)/(255)
		arr2 = arr2.astype(np.uint8)
		imageio.imwrite(fname+'.tif',np.reshape(arr2,(NrPixels,NrPixels)))
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
	fastIntegration = int(params[8].rstrip())
	OneDOut = int(params[9].rstrip())
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
		if midas_config and midas_config.MIDAS_GUI_DIR:
			badFN = os.path.join(midas_config.MIDAS_GUI_DIR, 'GEBad', 'BadImg.ge'+detNr)
		else:
			# Fallback (though this should ideally be unreachable if configured correctly)
			badFN = os.path.expanduser('~')+'/opt/MIDAS/gui/GEBad/BadImg.ge'+detNr
		print(badFN)
		badF = open(badFN,'rb')
		badF.seek(8192,os.SEEK_SET)
		badData = np.fromfile(badF,dtype=np.uint16,count=NrPixels*NrPixels)
		badData = np.nonzero(badData)[0]
	else:
		badData = []
	if darkProcessing is 1:
		darkfn = darkfilefullpath[:-1] + fn[-1]
		dark = getDarkImage(darkfn,8192,doBadProcessing,badData)
	statinfo = os.stat(fn)
	nFramesPerFile = int((statinfo.st_size - 8192)/(2*NrPixels*NrPixels))
	f = open(fn,'rb')
	header = np.fromfile(f,dtype=np.uint8,count=8192)
	sumArr = np.zeros(NrPixels*NrPixels,dtype=float)
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
			for idx in badData:
				if idx < NrPixels * (NrPixels-1) and idx >= NrPixels:
					data[idx] = (data[idx-1] + data[idx-NrPixels] + data[idx+1] + data[idx+NrPixels])/4
				else :
						data[idx] = 0
		corr = np.subtract(data,dark)
		idxz = np.nonzero(corr==np.min(corr))[0][0]
		print([np.min(corr), np.max(corr), idxz,data[idxz],dark[idxz]])
		if allFrames:
			writefn = outfn+'.frame.'+str(frameNr)+'.cor'
			saveFile(corr,writefn,fileTypeWrite)
			if doIntegration is 1:
				if fastIntegration is 0:
					if midas_config and midas_config.MIDAS_BIN_DIR:
						integrator_exe = os.path.join(midas_config.MIDAS_BIN_DIR, 'Integrator')
					else:
						integrator_exe = os.path.expanduser('~')+'/opt/MIDAS/FF_HEDM/bin/Integrator'
					call([integrator_exe,'ps_midas.txt',writefn])
				else:
					saveFastIntegrate(corr, OneDOut, writefn+'_integrated_framenr_'+str(frameNr)+'.csv')
		if sumWrite:
			sumArr = np.add(sumArr,corr)
		if maxWrite:
			maxArr = np.maximum(maxArr,corr)
	if sumWrite:
		writefn = outfn+'.sum'
		saveFile(sumArr,writefn,fileTypeWrite)
		if doIntegration is 1:
			if fastIntegration is 0:
				if midas_config and midas_config.MIDAS_BIN_DIR:
					integrator_exe = os.path.join(midas_config.MIDAS_BIN_DIR, 'Integrator')
				else:
					integrator_exe = os.path.expanduser('~')+'/opt/MIDAS/FF_HEDM/bin/Integrator'
				call([integrator_exe,'ps_midas.txt',writefn])
			else:
				saveFastIntegrate(sumArr, OneDOut, writefn+'_integrated_sum_'+'.csv')
	if meanWrite:
		writefn = outfn+'.ave'
		saveFile(sumArr/nFramesPerFile,writefn,fileTypeWrite)
		if doIntegration is 1:
			if fastIntegration is 0:
				if midas_config and midas_config.MIDAS_BIN_DIR:
					integrator_exe = os.path.join(midas_config.MIDAS_BIN_DIR, 'Integrator')
				else:
					integrator_exe = os.path.expanduser('~')+'/opt/MIDAS/FF_HEDM/bin/Integrator'
				call([integrator_exe,'ps_midas.txt',writefn])
			else:
				saveFastIntegrate(sumArr/nFramesPerFile, OneDOut, writefn+'_integrated_mean_'+'.csv')
	if maxWrite:
		writefn = outfn+'.max'
		saveFile(maxArr,writefn,fileTypeWrite)
		if doIntegration is 1:
			if fastIntegration is 0:
				if midas_config and midas_config.MIDAS_BIN_DIR:
					integrator_exe = os.path.join(midas_config.MIDAS_BIN_DIR, 'Integrator')
				else:
					integrator_exe = os.path.expanduser('~')+'/opt/MIDAS/FF_HEDM/bin/Integrator'
				call([integrator_exe,'ps_midas.txt',writefn])
			else:
				saveFastIntegrate(maxArr, OneDOut, writefn+'_integrated_max_'+'.csv')

def processImages():
	global darkfilefullpath, nFilesVar, NrPixels, RBinSize
	global EtaBinSize, RMin, RMax, EtaMin, EtaMax
	if folderVar.get() is '':
		return
	starttime = time.time()
	f = open('imparams.txt','w')
	f.write(str(doBad.get())+'\n')
	f.write(str(doDark.get())+'\n')
	f.write(str(doAllFrames.get())+'\n')
	f.write(str(doSum.get())+'\n')
	f.write(str(doMean.get())+'\n')
	f.write(str(doMax.get())+'\n')
	f.write(str(fileTypeVar.get())+'\n')
	f.write(str(integrateVar.get())+'\n')
	f.write(str(FastIntegrateVar.get())+'\n')
	f.write(str(OneDOutVar.get())+'\n')
	f.close()
	nrChecked = doAllFrames.get()+doSum.get()+doMean.get()+doMax.get()
	if nrChecked is 0:
		return
	pool = Pool(processes=multiprocessing.cpu_count())
	# We create a fnames.txt file with filenames for each file to process
	f = open('fnames.txt','w')
	fout = open('outputFnames.txt','w')
	fileStem = fileStemVar.get()
	darkfilefullpath = darkfilefullpathVar.get()
	NrPixels = NrPixelsVar.get()
	RBinSize = RBinSizeVar.get()
	EtaBinSize = EtaBinSizeVar.get()
	RMin = RMinVar.get()
	RMax = RMaxVar.get()
	EtaMin = EtaMinVar.get()
	EtaMax = EtaMaxVar.get()
	detExts = []
	if GE1Var.get():
		detExts.append(1)
	if GE2Var.get():
		detExts.append(2)
	if GE3Var.get():
		detExts.append(3)
	if GE4Var.get():
		detExts.append(4)
	if GE5Var.get():
		detExts.append(5)
	if '*' in fileStem:
		nFilesVar.set(0)
	if nFilesVar.get() is not 0:
		nrFiles = nFilesVar.get()
		startNr = firstFileNrVar.get()
		for fnr in range(startNr,startNr+nrFiles):
			fnTemp = getfn(fileStem,fnr)
			outfnTemp = getoutfn(fileStem,fnr)
			for detNr in detExts:
				f.write(fnTemp[:-1]+str(detNr)+'\n')
				fout.write(outfnTemp[:-1]+str(detNr)+'\n')
	else:
		## Do a pattern search.
		if fileStem is not '*':
			fnames = []
			for detNr in detExts:
				fnames += glob.glob(fileStem+'*'+'.ge'+str(detNr))
		else:
			fnames = []
			for detNr in detExts:
				fnames += glob.glob('*.ge'+str(detNr))
		for fname in fnames:
			# Remove dark file from todo files
			if darkfilefullpath[darkfilefullpath.rfind('/')+1:] in fname:
				continue
			f.write(folderVar.get()+'/'+fname+'\n')
			fout.write(outFolderVar.get()+'/'+fname+'\n')
		nrFiles = len(fnames)
	if not os.path.exists(outFolderVar.get()):
		os.makedirs(outFolderVar.get())
	f.close()
	fout.close()
	pipout = range(nrFiles)
	results = pool.map(processFile,pipout)
	os.remove('imparams.txt')
	os.remove('fnames.txt')
	os.remove('outputFnames.txt')
	if integrateVar.get() is 1 and FastIntegrateVar.get() is 0:
		os.remove('ps_midas.txt')
		os.remove('Map.bin')
		os.remove('nMap.bin')
	print(time.time() - starttime)

def acceptParameters():
	global topIntegrateParametersSelection, normalizer
	topIntegrateParametersSelection.destroy()
	normalizer = NormalizeVar.get()
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
	# call DetectorMapper if slow integration was selected
	if FastIntegrateVar.get() is 0:
		if midas_config and midas_config.MIDAS_BIN_DIR:
			cmdname = os.path.join(midas_config.MIDAS_BIN_DIR, 'DetectorMapper')
		else:
			cmdname = os.path.expanduser('~')+'/opt/MIDAS/FF_HEDM/bin/DetectorMapper'
		call([cmdname,'ps_midas.txt'])
	else:
		st = time.time()
		mapFastIntegration() # this will create 2 arrays: Rads and/or Etas.
		print(time.time() - st)
	# call processImages
	processImages()

def Enable1D():
	global cButton1D, OneDOutVar
	if FastIntegrateVar.get() is 1:
		cButton1D.config(state=Tk.ACTIVE)
	else:
		cButton1D.config(state=Tk.DISABLED)
		OneDOutVar.set(0)

def processSquare():
	global tomImageConvert
	global npxyvar, npxzvar
	thisfolder = thisfoldervar.get()
	fstem = fstemvar.get()
	outfstem = outfstemvar.get()
	startnr = stnrvar.get()
	fnext = fnextvar.get()
	pad = paddingvar.get()
	npxy = npxyvar.get()
	npxz = npxzvar.get()
	if fnext == 'tif':
		print("We have tif")
		fn = thisfolder+'/'+fstem+'_'+str(startnr).zfill(pad)+'.'+fnext
		print(fn)
		head = np.fromfile(open(fn,'rb'),dtype=np.uint8,count=8192)
		im = PIL.Image.open(fn)
		img = np.array(im,dtype=np.int32)
		npxyvar.set(img.shape[0])
		npxzvar.set(img.shape[1])
		npxy = npxyvar.get()
		npxz = npxzvar.get()
		bigdim = max(npxy,npxz)
		img = np.where(img==-2,16*65535,img)
		img = np.where(img==-1,16*65535,img)
		img = img.astype(np.double)
		img = img / 16.0
		img = img.round()
		outimg = np.zeros((bigdim,bigdim))
		outF = open(thisfolder+'/'+outfstem+'_square_'+str(bigdim)+'_px_'+str(startnr).zfill(pad)+'.ge3','wb')
		np.array(head).tofile(outF)
		outimg[:npxy,:npxz]=img
		outimg = outimg.astype(np.uint16)
		np.array(outimg).tofile(outF)
	else:
		print("We have RAW.")
		fn = thisfolder+'/'+fstem+'_'+str(startnr).zfill(pad)+'.'+fnext
		print(fn)
		sizefile = os.stat(fn).st_size
		sizeframe = npxy*npxz*4 # hard coded that float32
		nFrames = (sizefile - 8192)/sizeframe
		print(nFrames)
		inF = open(fn,'rb')
		head = np.fromfile(inF,dtype=np.uint8,count=8192)
		bytesToSkip = 8192
		#inF.seek(bytesToSkip,os.SEEK_SET)
		bigdim = max(npxy,npxz)
		outimg = np.zeros((bigdim,bigdim))
		outimg = outimg.astype(np.uint16)
		outF = open(thisfolder+'/'+outfstem+'_square_'+str(bigdim)+'_px_'+str(startnr).zfill(pad)+'.ge3','wb')
		np.array(head).tofile(outF)
		for framenr in range(nFrames):
			img = np.fromfile(inF,dtype=np.int32,count=(npxy*npxz))
			img = np.where(img==-2,16*65535,img)
			img = np.where(img==-1,16*65535,img)
			img = img.astype(np.double)
			img = img / 16.0
			img = img.round()
			print(framenr)
			img = img.astype(np.uint16)
			img = img.reshape((npxy,npxz))
			outimg[:npxy,:npxz] = img
			np.array(outimg).tofile(outF)
	returnBack()

def returnBack():
	global tomImageConvert
	topImageConvert.destroy()

def raw_to_ge():
	global thisfoldervar, fstemvar, stnrvar, endnrvar, fnextvar, paddingvar, outfstemvar
	global npxyvar, npxzvar
	global topImageConvert
	topImageConvert = Tk.Toplevel()
	topImageConvert.title('Convert Rectangle Shaped Images to Square Images')
	Tk.Label(master=topImageConvert,text='            Folder').grid(row=1,column=1)
	Tk.Entry(master=topImageConvert,textvariable=thisfoldervar,width=50).grid(row=1,column=2,sticky=Tk.W)
	Tk.Label(master=topImageConvert,text='         File Stem').grid(row=2,column=1)
	Tk.Entry(master=topImageConvert,textvariable=fstemvar,width=50).grid(row=2,column=2,sticky=Tk.W)
	Tk.Label(master=topImageConvert,text='      OutFile Stem').grid(row=3,column=1)
	Tk.Entry(master=topImageConvert,textvariable=outfstemvar,width=50).grid(row=3,column=2,sticky=Tk.W)
	Tk.Label(master=topImageConvert,text='      File Number').grid(row=4,column=1)
	Tk.Entry(master=topImageConvert,textvariable=stnrvar,width=6 ).grid(row=4,column=2,sticky=Tk.W)
	Tk.Label(master=topImageConvert,text='           Padding').grid(row=6,column=1)
	Tk.Entry(master=topImageConvert,textvariable=paddingvar,width=6 ).grid(row=6,column=2,sticky=Tk.W)
	Tk.Label(master=topImageConvert,text='    File Extension').grid(row=7,column=1)
	Tk.Entry(master=topImageConvert,textvariable=fnextvar,width=6 ).grid(row=7,column=2,sticky=Tk.W)
	Tk.Label(master=topImageConvert,text='         NrPixelsY').grid(row=8,column=1)
	Tk.Entry(master=topImageConvert,textvariable=npxyvar,width=6 ).grid(row=8,column=2,sticky=Tk.W)
	Tk.Label(master=topImageConvert,text='         NrPixelsZ').grid(row=9,column=1)
	Tk.Entry(master=topImageConvert,textvariable=npxzvar,width=6 ).grid(row=9,column=2,sticky=Tk.W)
	Tk.Button(master=topImageConvert,text="Process",command=processSquare).grid(row=10,column=1)
	Tk.Button(master=topImageConvert,text="Exit",command=returnBack).grid(row=10,column=2)
	Tk.Label(master=topImageConvert,text="For tiff input: Will take StartNr to EndNr i").grid(row=11,column=1,columnspan=2)


def processStitch():
	global topStitch
	thisfolder = thisfoldervar.get()
	fstem = fstemvar.get()
	outfstem = outfstemvar.get()
	startnr = stnrvar.get()
	fnext = fnextvar.get()
	pad = paddingvar.get()
	dfn = darkfnvar.get()
	nfilesperscan = nfilesperscanvar.get()
	nscans = nscansvar.get()
	nlayers = nlayersvar.get()
	translation = translationvar.get()
	npx = npxvar.get()
	nframes = nframesvar.get()
	dfile = open(dfn,'rb')
	head = np.fromfile(dfile,dtype=np.uint8,count=8192)
	dArr = np.fromfile(dfile,dtype=np.uint16,count=(npx*npx))
	dArr = dArr.astype(float)
	sttime = time.time()
	for layerNr in range(nlayers):
		for scanFileNr in range(nfilesperscan):
			outFNr = startnr + layerNr*nfilesperscan + scanFileNr
			outF = open(thisfolder+'/'+outfstem+'_Stitch_'+str(outFNr).zfill(pad)+fnext,'wb')
			np.array(head).tofile(outF)
			for frameNr in range(nframes):
				thistime = time.time()
				print('Processing FrameNr: ' + str(frameNr) + ' out of ' + str(nframes) + '. Time taken till now: ' + str(thistime - sttime) + 's.')
				imgArr = np.zeros(npx*npx)
				imgArr = imgArr.astype(float)
				for scanNr in range(nscans):
					zeroArr = np.zeros(scanNr*translation*npx)
					positions = range(npx*(npx-scanNr*translation),npx*npx)
					thisFileNr = layerNr*nfilesperscan*nscans + scanFileNr*nscans + scanNr + startnr
					bytesToSkip = 8192 + frameNr*npx*npx*2
					inFN = thisfolder+'/'+fstem+'_'+str(thisFileNr).zfill(pad)+fnext
					inF = open(inFN,'rb')
					inF.seek(bytesToSkip,os.SEEK_SET)
					data = np.fromfile(inF,dtype=np.uint16,count=(npx*npx))
					data = data.astype(float)
					data = data - dArr
					# Convert to 2D, translate, convert to 1D
					data = data.reshape((npx,npx))
					data = np.roll(data,scanNr*translation,axis=1) # Test
					data = data.reshape(npx*npx)
					imgArr += data
				imgArr += dArr
				imgArr = imgArr.astype(np.uint16)
				np.array(imgArr).tofile(outF)
	topStitch.destroy()

def stitch_ff():
	global thisfoldervar, fstemvar, stnrvar, fnextvar, darkfnvar, paddingvar, outfstemvar
	global nfilesperscanvar, nscansvar, nlayersvar, translationvar, npxvar, nframesvar
	global topStitch
	topStitch = Tk.Toplevel()
	topStitch.title('Stitch FF GE images for further processing')
	Tk.Label(master=topStitch,text='            Folder').grid(row=1,column=1)
	Tk.Entry(master=topStitch,textvariable=thisfoldervar,width=50).grid(row=1,column=2,sticky=Tk.W)
	Tk.Label(master=topStitch,text='         File Stem').grid(row=2,column=1)
	Tk.Entry(master=topStitch,textvariable=fstemvar,width=50).grid(row=2,column=2,sticky=Tk.W)
	Tk.Label(master=topStitch,text='      OutFile Stem').grid(row=3,column=1)
	Tk.Entry(master=topStitch,textvariable=outfstemvar,width=50).grid(row=3,column=2,sticky=Tk.W)
	Tk.Label(master=topStitch,text='      Start Number').grid(row=4,column=1)
	Tk.Entry(master=topStitch,textvariable=stnrvar,width=6 ).grid(row=4,column=2,sticky=Tk.W)
	Tk.Label(master=topStitch,text='           Padding').grid(row=5,column=1)
	Tk.Entry(master=topStitch,textvariable=paddingvar,width=6 ).grid(row=5,column=2,sticky=Tk.W)
	Tk.Label(master=topStitch,text='    File Extension').grid(row=6,column=1)
	Tk.Entry(master=topStitch,textvariable=fnextvar,width=6 ).grid(row=6,column=2,sticky=Tk.W)
	Tk.Label(master=topStitch,text='         Dark File').grid(row=7,column=1)
	Tk.Entry(master=topStitch,textvariable=darkfnvar,width=50).grid(row=7,column=2,sticky=Tk.W)
	Tk.Label(master=topStitch,text='  NrFiles Per Scan').grid(row=8,column=1)
	Tk.Entry(master=topStitch,textvariable=nfilesperscanvar,width=6 ).grid(row=8,column=2,sticky=Tk.W)
	Tk.Label(master=topStitch,text='           NrScans').grid(row=9,column=1)
	Tk.Entry(master=topStitch,textvariable=nscansvar,width=6 ).grid(row=9,column=2,sticky=Tk.W)
	Tk.Label(master=topStitch,text='          NrLayers').grid(row=10,column=1)
	Tk.Entry(master=topStitch,textvariable=nlayersvar,width=6 ).grid(row=10,column=2,sticky=Tk.W)
	Tk.Label(master=topStitch,text='TranslationAmt(px)').grid(row=11,column=1)
	Tk.Entry(master=topStitch,textvariable=translationvar,width=6 ).grid(row=11,column=2,sticky=Tk.W)
	Tk.Label(master=topStitch,text='Towards the door is positive.').grid(row=11,column=2)
	Tk.Label(master=topStitch,text='          NrPixels').grid(row=12,column=1)
	Tk.Entry(master=topStitch,textvariable=npxvar,width=6 ).grid(row=12,column=2,sticky=Tk.W)
	Tk.Label(master=topStitch,text='   NrFramesPerFile').grid(row=13,column=1)
	Tk.Entry(master=topStitch,textvariable=nframesvar,width=6 ).grid(row=13,column=2,sticky=Tk.W)
	Tk.Button(master=topStitch,text="Process",command=processStitch).grid(row=14,column=1,columnspan=2)

def integrate():
	global EtaBinSizeVar, RBinSizeVar, RMaxVar, RMinVar, EtaMaxVar, EtaMinVar
	global NrPixelsVar, NormalizeVar, FloatFileVar, txVar, tyVar, tzVar
	global pxVar, yBCVar, zBCVar, LsdVar, RhoDVar, p0Var, p1Var, p2Var
	global topIntegrateParametersSelection, integrateVar, fileTypeVar
	global FastIntegrateVar, OneDOutVar
	global cButton1D
	nrChecked = doAllFrames.get()+doSum.get()+doMean.get()+doMax.get()
	if nrChecked is 0:
		return
	if folderVar.get() is '':
		return
	topIntegrateParametersSelection = Tk.Toplevel()
	integrateVar.set(1)
	fileTypeVar.set(1)
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
	cButtonFast = Tk.Checkbutton(master=topIntegrateParametersSelection,
		text="Fast Integration(reduced accuracy)",
		variable=FastIntegrateVar,command=Enable1D)
	cButtonFast.grid(row=7,column=1,columnspan=3)
	cButton1D = Tk.Checkbutton(master=topIntegrateParametersSelection,
		text="1D output (compatible with FastIntegration only)",
		variable=OneDOutVar)
	cButton1D.grid(row=7,column=4,columnspan=4)
	Tk.Button(master=topIntegrateParametersSelection,
		text="Continue",command=acceptParameters).grid(row=20,
		column=1,columnspan=10)

def CheckGEs():
	if doHydra.get():
		GE1Var.set(1)
		GE2Var.set(1)
		GE3Var.set(1)
		GE4Var.set(1)
	else:
		GE1Var.set(0)
		GE2Var.set(0)
		GE3Var.set(0)
		GE4Var.set(0)

root = Tk.Tk()
root.wm_title("Image Manipulation Software, MIDAS, v0.1 Dt. 2018/01/30 hsharma@anl.gov")
figur = Figure(figsize=(14.5,8.5),dpi=100)
canvas = FigureCanvasTkAgg(figur,master=root)
a = figur.add_subplot(111,aspect='equal')
a.title.set_text("Selected Image")

fileTypeVar = Tk.IntVar()
fileTypeVar.set(1)
firstFileNrVar = Tk.IntVar()
firstFileNrVar.set(0)
folder = ''
folderVar = Tk.StringVar()
outfolder = ''
outFolderVar = Tk.StringVar()
folderVar.set(folder)
NrPixels = 2048
NrPixelsVar = Tk.IntVar()
NrPixelsVar.set(NrPixels)
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
FastIntegrateVar = Tk.IntVar()
OneDOutVar = Tk.IntVar()
FastIntegrateVar.set(1)
OneDOutVar.set(1)
GE1Var = Tk.IntVar()
GE1Var.set(0)
GE2Var = Tk.IntVar()
GE2Var.set(0)
GE3Var = Tk.IntVar()
GE3Var.set(0)
GE4Var = Tk.IntVar()
GE4Var.set(0)
GE5Var = Tk.IntVar()
GE5Var.set(0)
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
EtaBinSizeVar = Tk.DoubleVar()
RBinSizeVar = Tk.DoubleVar()
RMaxVar = Tk.DoubleVar()
RMinVar = Tk.DoubleVar()
EtaMaxVar = Tk.DoubleVar()
EtaMinVar = Tk.DoubleVar()
EtaBinSizeVar.set(5.0)
RBinSizeVar.set(1.0)
RMaxVar.set(1024.0)
RMinVar.set(10.0)
EtaMaxVar.set(180.0)
EtaMinVar.set(-180.0)
thisfoldervar = Tk.StringVar()
fstemvar = Tk.StringVar()
outfstemvar = Tk.StringVar()
stnrvar = Tk.IntVar()
endnrvar = Tk.IntVar()
fnextvar = Tk.StringVar()
fnextvar.set('tif')
paddingvar = Tk.IntVar()
paddingvar.set(6)
darkfnvar = Tk.StringVar()
nfilesperscanvar = Tk.IntVar()
nscansvar = Tk.IntVar()
nlayersvar = Tk.IntVar()
translationvar = Tk.IntVar()
npxvar = Tk.IntVar()
npxyvar = Tk.IntVar()
npxzvar = Tk.IntVar()
nframesvar = Tk.IntVar()

rowFigSize = 3
colFigSize = 3

Tk.Label(master=root,text="Image pre-processing and conversion using MIDAS",
	font=("Helvetica",20)).grid(row=0,column=0,rowspan=rowFigSize,
	columnspan=colFigSize,sticky=Tk.W+Tk.E+Tk.N+Tk.S)

leftSideFrame = Tk.Frame(root)
leftSideFrame.grid(row=rowFigSize+1,column=0,rowspan=8,sticky=Tk.W)
firstRowFrame = Tk.Frame(root)
firstRowFrame.grid(row=rowFigSize+1,column=1,columnspan=2,sticky=Tk.W)
firstSecondRowFrame = Tk.Frame(root)
firstSecondRowFrame.grid(row=rowFigSize+2,column=1,columnspan=2,sticky=Tk.W)
secondRowFrame = Tk.Frame(root)
secondRowFrame.grid(row=rowFigSize+3,column=1,sticky=Tk.W)
midRowFrame = Tk.Frame(root)
midRowFrame.grid(row=rowFigSize+4,column=1,sticky=Tk.W)
twoThirdRowFrame = Tk.Frame(root)
twoThirdRowFrame.grid(row=rowFigSize+5,column=1,sticky=Tk.W)
threeThirdRowFrame = Tk.Frame(root)
threeThirdRowFrame.grid(row=rowFigSize+6,column=1,sticky=Tk.W)
thirdRowFrame = Tk.Frame(root)
thirdRowFrame.grid(row=rowFigSize+7,column=1,sticky=Tk.W)
fourthRowFrame = Tk.Frame(root)
fourthRowFrame.grid(row=rowFigSize+8,column=1,sticky=Tk.W)
fifthRowFrame = Tk.Frame(root)
fifthRowFrame.grid(row=rowFigSize+9,column=1,sticky=Tk.W)
rightSideFrame = Tk.Frame(root)
rightSideFrame.grid(row=rowFigSize+2,column=2,rowspan=8,sticky=Tk.W)
bottomFrame = Tk.Frame(root)
bottomFrame.grid(row=rowFigSize+10,column=0,columnspan=4,sticky=Tk.W)

Tk.Button(master=leftSideFrame,text='Quit',command=_quit,
	font=("Helvetica",20)).grid(row=0,column=0,padx=10,pady=10)

Tk.Button(master=leftSideFrame,text='StitchFF',command=stitch_ff,
	font=("Helvetica",20)).grid(row=1,column=0,padx=10,pady=10)

Tk.Button(master=leftSideFrame,text='Raw2GE',command=raw_to_ge,
	font=("Helvetica",20)).grid(row=3,column=0,padx=10,pady=10)

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
	variable=doHydra,command=CheckGEs).grid(row=1,column=4,sticky=Tk.W)

###### Detector Types and NrPixels
Tk.Label(master=firstSecondRowFrame,text="Extensions  ").grid(row=1,column=0,sticky=Tk.W)
Tk.Checkbutton(master=firstSecondRowFrame,text="GE1   ",
	variable=GE1Var).grid(row=1,column=1,sticky=Tk.W)
Tk.Checkbutton(master=firstSecondRowFrame,text="GE2   ",
	variable=GE2Var).grid(row=1,column=2,sticky=Tk.W)
Tk.Checkbutton(master=firstSecondRowFrame,text="GE3   ",
	variable=GE3Var).grid(row=1,column=3,sticky=Tk.W)
Tk.Checkbutton(master=firstSecondRowFrame,text="GE4   ",
	variable=GE4Var).grid(row=1,column=4,sticky=Tk.W)
Tk.Checkbutton(master=firstSecondRowFrame,text="GE5   ",
	variable=GE5Var).grid(row=1,column=5,sticky=Tk.W)

Tk.Label(master=firstSecondRowFrame,text="NrPixels ").grid(row=1,column=6,sticky=Tk.W)
Tk.Entry(master=firstSecondRowFrame,textvariable=NrPixelsVar,width=5).grid(row=1,
	column=7,sticky=Tk.W)

###### Folder info
Tk.Label(master=midRowFrame,text="Input Folder ").grid(row=1,column=0,sticky=Tk.W)
Tk.Entry(master=midRowFrame,textvariable=folderVar,width=66).grid(row=1,column=1,sticky=Tk.W)

###### Rest info
Tk.Label(master=secondRowFrame,text='FileStem     ').grid(row=1,column=0,sticky=Tk.W)
Tk.Entry(master=secondRowFrame,textvariable=fileStemVar,width=35).grid(row=1,column=1,sticky=Tk.W)

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

for text, val in FILEOPTS:
	Tk.Radiobutton(master=fourthRowFrame,text=text,variable=fileTypeVar,
		value=val,font=("Helvetica 16 bold")).grid(row=1,column=val,sticky=Tk.W)

###### Run processing
Tk.Button(master=firstRowFrame,text="Integrate",
	command=integrate,font=('Helvetica',20)).grid(row=1,
	column=5,padx=10,pady=10)

Tk.Button(master=rightSideFrame,text="Process Images",
	command=processImages,font=("Helvetica",20)).grid(row=0,
	column=0,padx=10,pady=10)

Tk.Button(master=rightSideFrame,text="Batch Images",
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
