def remote_prepare(data): #paramFileName startLayerNr endLayerNr timePath StartFileNrFirstLayer NrFilesPerSweep FileStem SeedFolder StartNr EndNr
	import numpy as np
	import os, subprocess
	import datetime
	from pathlib import Path
	import shutil
	paramFN = data['paramFileName']
	startLayerNr = int(data['startLayerNr'])
	endLayerNr = int(data['endLayerNr'])
	time_path = data['timePath']
	startFileNrFirstLayer = int(data['StartFileNrFirstLayer'])
	nrFilesPerSweep = int(data['NrFilesPerSweep'])
	fStem = data['FileStem']
	topdir = data['SeedFolder']
	startNr = int(data['StartNr'])
	endNr = int(data['EndNr'])
	os.chdir(topdir)
	baseNameParamFN = paramFN.split('/')[-1]
	homedir = os.path.expanduser('~')
	nFrames = endNr - startNr + 1
	for layerNr in range(startLayerNr,endLayerNr+1):
		# Single node commands
		startTime = time.time()
		thisStartNr = startNrFirstLayer + (layerNr-1)*nrFilesPerSweep
		folderName = fStem + '_Layer_' + str(layerNr).zfill(4) + '_Analysis_Time_' + time_path
		thisDir = topdir + '/' + folderName + '/'
		Path(thisDir).mkdir(parents=True,exist_ok=True)
		os.chdir(thisDir)
		thisParamFN = thisDir + baseNameParamFN
		thisPF = open(thisParamFN,'w')
		for line in paramContents:
			thisPF.write(line)
		thisPF.write('RawFolder '+topdir+'\n')
		thisPF.write('SeedFolder '+topdir+'\n')
		thisPF.write('Folder '+thisDir+'\n')
		thisPF.write('LayerNr '+str(layerNr)+'\n')
		thisPF.write('StartFileNr '+str(thisStartNr)+'\n')
		thisPF.close()
		Path(thisDir+'/Temp').mkdir(parents=True,exist_ok=True)
		Path(thisDir+'Output').mkdir(parents=True,exist_ok=True)
		Path(thisDir+'Results').mkdir(parents=True,exist_ok=True)
		subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/GetHKLList")+" "+thisParamFN,shell=True)

def remote_peaksearch(data): # startLayerNr endLayerNr nFrames numProcs numBlocks blockNr timePath FileStem SeedFolder paramFileName
	import os, subprocess
	startLayerNr = int(data['startLayerNr'])
	endLayerNr = int(data['endLayerNr'])
	nFrames = int(data['nFrames'])
	numProcs = int(data['numProcs'])
	numBlocks = int(data['numBlocks'])
	blockNr = int(data['blockNr'])
	time_path = data['timePath']
	fStem = data['FileStem']
	topdir = data['SeedFolder']
	paramFN = data['paramFileName']
	baseNameParamFN = paramFN.split('/')[-1]
	for layerNr in range(startLayerNr,endLayerNr+1):
		folderName = fStem + '_Layer_' + str(layerNr).zfill(4) + '_Analysis_Time_' + time_path
		thisDir = topdir + '/' + folderName + '/'
		os.chdir(thisDir)
		subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/PeaksFittingOMP")+' '+baseNameParamFN+' '+ blockNr + ' ' + numBlocks + ' '+str(nFrames)+' '+str(numProcs),shell=True)

def remote_transforms(data): # startLayerNr endLayerNr timePath FileStem SeedFolder paramFileName
	import os, subprocess
	startLayerNr = int(data['startLayerNr'])
	endLayerNr = int(data['endLayerNr'])
	time_path = data['timePath']
	fStem = data['FileStem']
	topdir = data['SeedFolder']
	paramFN = data['paramFileName']
	baseNameParamFN = paramFN.split('/')[-1]
	for layerNr in range(startLayerNr,endLayerNr+1):
		folderName = fStem + '_Layer_' + str(layerNr).zfill(4) + '_Analysis_Time_' + time_path
		thisDir = topdir + '/' + folderName + '/'
		os.chdir(thisDir)
		subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/MergeOverlappingPeaksAll")+' '+baseNameParamFN,shell=True)
		subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/CalcRadiusAll")+' '+baseNameParamFN,shell=True)
		subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/FitSetup")+' '+baseNameParamFN,shell=True)
		subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/SaveBinData"),shell=True)

def remote_indexrefine(data): # startLayerNr endLayerNr numProcs numBlocks blockNr timePath FileStem SeedFolder
	import os, subprocess
	startLayerNr = int(data['startLayerNr'])
	endLayerNr = int(data['endLayerNr'])
	numProcs = int(data['numProcs'])
	numBlocks = int(data['numBlocks'])
	blockNr = int(data['blockNr'])
	time_path = data['timePath']
	fStem = data['FileStem']
	topdir = data['SeedFolder']
	for layerNr in range(startLayerNr,endLayerNr+1):
		folderName = fStem + '_Layer_' + str(layerNr).zfill(4) + '_Analysis_Time_' + time_path
		thisDir = topdir + '/' + folderName + '/'
		os.chdir(thisDir)
		nSpotsToIndex = len(open('SpotsToIndex.csv').readlines())
		subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/IndexerOMP")+' paramstest.txt '+blockNr+' '+numBlocks+' '+str(nSpotsToIndex)+' '+str(numProcs),shell=True)
		subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/FitPosOrStrainsOMP")+' paramstest.txt '+blockNr+' '+numBlocks+' '+str(nSpotsToIndex)+' '+str(numProcs),shell=True)

def remote_find_grains(data): # startLayerNr endLayerNr timePath FileStem SeedFolder paramFileName
	import os, subprocess
	startLayerNr = int(data['startLayerNr'])
	endLayerNr = int(data['endLayerNr'])
	time_path = data['timePath']
	fStem = data['FileStem']
	topdir = data['SeedFolder']
	paramFN = data['paramFileName']
	baseNameParamFN = paramFN.split('/')[-1]
	for layerNr in range(startLayerNr,endLayerNr+1):
		folderName = fStem + '_Layer_' + str(layerNr).zfill(4) + '_Analysis_Time_' + time_path
		thisDir = topdir + '/' + folderName + '/'
		os.chdir(thisDir)
		subprocess.call(os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/ProcessGrains') + ' ' + baseNameParamFN,shell=True)
		os.chdir(topdir)
	subprocess.call('tar -czf recon_'+time_path+'.tar.gz *_Analysis_Time_'+time_path+'*',shell=True)
