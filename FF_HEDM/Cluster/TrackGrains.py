#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

import sys
import os
import shutil
from glob import glob
from subprocess import call
from subprocess import check_call
from subprocess import check_output
from os.path import expanduser
from time import gmtime, strftime

def fileAppend(fileName,line):
	f = open(fileName,'a')
	f.write(line+'\n')
	f.close()

def getValueFromParamFile(paramfn,searchStr,nLines=1,wordNr=1,nWords=1):
	ret_list = []
	nrLines = 0
	f = open(paramfn,'r')
	PSContents = f.readlines()
	for line in PSContents:
		if line.startswith(searchStr+' '):
			line = line.replace('\t',' ')
			line = line.replace('\n',' ')
			words = line.split(' ')
			words = filter(None,words)
			ret_list.append(words[wordNr:wordNr+nWords])
			nrLines += 1
			if (nrLines == nLines):
				return ret_list
	return ret_list

def removeLinesFile(fn,patternMatch):
	f = open(fn,'r')
	lines = f.readlines()
	f.close()
	f = open(fn,'w')
	for line in lines:
		if !line.startswith(patternMatch):
			f.write(line)
	f.close()

def moveMultipleFiles(sourceDir,destDir,patternMatch):
	for fn in glob(sourceDir+patternMatch):
		shutil.move(fn,destDir)

def runPeaksMult(paramFile,nNodes,ringNrsFile,paramFNStem,fstm,machineName):
	startNr = getValueFromParamFile(paramFile,'StartNr')[0][0]
	endNr = getValueFromParamFile(paramFile,'EndNr')[0][0]
	print 'Peaks:'
	myenv = os.environ.copy()
	myenv['nNODES'] = nNodes
	if 'edison' in machineName:
		hn = check_output(['hostname'])		
		hn = int(hn.split()[0][-2:])
		hn += 20
		intHN = '128.55.203.'+str(hn)
	elif 'cori' in machineName:
		hn = check_output(['hostname'])		
		hn = int(hn.split()[0][-2:])
		hn += 30
		intHN = '128.55.224.'+str(hn)
	else:
		intHN = '10.10.10.100'
	myenv['intHN'] = intHN
	myenv['JAVA_HOME'] = expanduser('~')+'/.MIDAS/jre1.8.0_181/'
	myenv['PATH'] = myenv['JAVA_HOME']+'/bin:'+myenv['PATH']
	cmd = [swiftdir+'/swift','-config',pfdir+'/sites.conf','-sites',machineName,pfdir+'RunPeaksMultPeaksOnly.swift',
		'-paramsfile='+paramFile,'-ringfile='+ringNrsFile,'-fstm='+fstm,'-startnr='+startNr,'-endnr='+endNr]
	check_call(cmd,env=myenv,shell=True)
	print "Process Peaks"
	cmd = [swiftdir+'/swift','-config',pfdir+'/sites.conf','-sites',machineName,pfdir+'RunPeaksMultProcessOnly.swift',
		'-paramsfile='+paramFile,'-ringfile='+ringNrsFile,'-fstm='+fstm,'-startnr='+startNr,'-endnr='+endNr]
	check_call(cmd,env=myenv,shell=True)

def GrainTracking(paramFile,layerNr,nNodes,machineName):
	startNrFirstLayer = int(getValueFromParamFile(paramFile,'StartFileNrFirstLayer')[0][0])
	ringNrs = getValueFromParamFile(paramFile,'RingThresh',100)
	sgNum = getValueFromParamFile(paramFile,'SpaceGroup')[0][0]
	thresholds = getValueFromParamFile(paramFile,'RingThresh',100,2,1)
	finalRingToIndex = getValueFromParamFile(paramFile,'OverAllRingToIndex')[0][0]
	seedFolder = getValueFromParamFile(paramFile,'SeedFolder')[0][0]
	nrFilesPerLayer = int(getValueFromParamFile(paramFile,'NrFilesPerSweep')[0][0])
	margABC = getValueFromParamFile(paramFile,'MargABC')[0][0]
	margABG = getValueFromParamFile(paramFile,'MargABG')[0][0]
	sNr = getValueFromParamFile(paramFile,'StartNr')[0][0]
	eNr = getValueFromParamFile(paramFile,'EndNr')[0][0]
	minNrSpots = getValueFromParamFile(paramFile,'MinNrSpots')[0][0]
	print 'Ring to be used for seed points: '+finalRingToIndex
	flr = os.path.dirname(paramFile) + '/'
	fstm = paramFile.split('/')[-1]
	print fstm
	fileStem = getValueFromParamFile(paramFile,'FileStem')[0][0]
	OldFolder = getValueFromParamFile(paramFile,'OldFolder')[0][0]
	startFileNr = startNrFirstLayer + nrFilesPerLayer*(layerNr-1)
	print os.getcwd()
	call([binfolder+'/GetHKLList',paramFile])
	print 'StartFileNr ' + str(startFileNr)
	print 'Creating overall parameter file for this layer:'
	PFName = flr+'/Layer'+str(layerNr)+'_MultiRing_'+fstm
	print PFName
	shutil.copyfile(paramFile,PFName)
	shutil.copyfile(os.getcwd()+'/hkls.csv',seedFolder+'/hkls.csv')
	fileAppend(PFName,'LayerNr '+str(layerNr)+'\n')
	fileAppend(PFName,'RingToIndex '+finalRingToIndex+'\n')
	fileAppend(PFName,'Folder '+seedFolder+'\n')
	ide = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
	layerDir = fileStem + '_Layer' + str(layerNr) + '_Analysis_Time_' + ide
	outFldr = seedFolder + '/' + layerDir + '/'
	os.chdir(seedFolder)
	ringNrsFile = flr + '/Layer' + str(layerNr)+'_RingInfo.txt'
	if os.path.exists(ringNrsFile):
		os.remove(ringNrsFile)
	paramFNStem = flr + '/Layer' + str(layerNr)+'_Ring'
	if not os.path.exists(outFldr):
		os.makedirs(outFldr)
	shutil.copyfile(os.getcwd()+'/hkls.csv',outFldr+'/hkls.csv')
	shutil.copy2(paramFile,outFldr)
	i=0
	for rings in ringNrs:
		thisParamFN = paramFNStem+rings[0]+'_'+fstm
		print 'ParameterFile used: ' + thisParamFN
		shutil.copyfile(paramFile,thisParamFN)
		print 'Ring Number: ' + rings[0] + ', Threshold: ' + thresholds[i][0]
		Fldr = seedFolder + '/Ring'+ rings[0]
		if not os.path.exists(Fldr):
			os.makedirs(Fldr)
		shutil.copyfile(os.getcwd()+'/hkls.csv',Fldr+'/hkls.csv')
		fileAppend(thisParamFN,'Folder '+Fldr)
		fileAppend(thisParamFN,'RingToIndex '+rings[0])
		fileAppend(thisParamFN,'RingNumbers '+rings[0])
		fileAppend(thisParamFN,'LowerBoundThreshold '+thresholds[i][0])
		fileAppend(thisParamFN,'LayerNr '+str(layerNr))
		fileAppend(thisParamFN,'StartFileNr '+str(startFileNr))
		fileAppend(PFName,'RingNumbers ' + rings[0])
		fileAppend(ringNrsFile,rings[0])
		i+=1
	runPeaksMult(paramFile,nNodes,ringNrsFile,paramFNStem,fstm,machineName)
	shutil.move(ringNrsFile,outFldr)
	for rings in ringNrs:
		thisParamFN = paramFNStem+rings[0]+'_'+fstm
		Fldr = seedFolder + '/Ring'+ rings[0]
		shutil.copy2(fldr+'/PeakSearch/'+fileStem+'_'+str(layerNr)+'/paramstest.txt',outFldr+'/paramstest_RingNr'+rings[0]+'.txt')
		shutil.copy2(fldr+'/PeakSearch/'+fileStem+'_'+str(layerNr)+'/Radius_StartNr_'+sNr+'_EndNr_'+eNr+'_RingNr_'+rings[0]+'.csv',outFldr)
		RingX = rings[0]
		shutil.move(thisParamFN,outFldr)
	call([binfolder+'/MergeMultipleRings',PFName])
	os.chdir(seedFolder)
	moveMultipleFiles(os.getcwd()+'/',outFldr,'SpotsToIndex*')
	moveMultipleFiles(os.getcwd()+'/',outFldr,'InputAll*')
	shutil.move(PFName,outFldr)
	shutil.move(os.getcwd()+'/IDsHash.csv',outFldr)
	shutil.copy2(os.getcwd()+'/hkls.csv',outFldr)
	os.chdir(outFldr)
	shutil.copy2(os.getcwd()+'/paramstest_RingNr'+RingX+'.txt',os.getcwd()+'/paramstest.txt')
	removeLinesFile(os.getcwd()+'/paramstest.txt','OutputFolder')
	removeLinesFile(os.getcwd()+'/paramstest.txt','ResultFolder')
	removeLinesFile(os.getcwd()+'/paramstest.txt','RingRadii')
	removeLinesFile(os.getcwd()+'/paramstest.txt','RingNumbers')
	fileAppend(os.getcwd()+'/paramstest.txt','OutputFolder '+outFldr+'/Output')
	fileAppend(os.getcwd()+'/paramstest.txt','ResultFolder '+outFldr+'/Results')
	fileAppend(os.getcwd()+'/paramstest.txt','MargABC '+margABC)
	fileAppend(os.getcwd()+'/paramstest.txt','MargABG '+margABG)
	for rings in ringNrs:
		fileAppend(os.getcwd()+'/paramstest.txt','RingNumbers '+rings[0])
		paramstestThisRing = outFldr+'/paramstest_RingNr'+rings[0]+'.txt'
		rad = getValueFromParamFile(paramstestThisRing,'RingRadii')[0][0]
		fileAppend(os.getcwd()+'/paramstest.txt','RingRadii '+rad)
	fileAppend(os.getcwd()+'/paramstest.txt','SpaceGroup '+sgNum)
	fileAppend(os.getcwd()+'/paramstest.txt','GrainTracking 1')
	fileAppend(os.getcwd()+'/paramstest.txt','OldFolder '+OldFolder)
	call([pfdir+'/RefineTracking.sh',nNodes,paramFile,OldFolder,machineName])

##### main

if (len(sys.argv)!=7):
	print "Provide ParametersFile StartLayerNr EndLayerNr Number of NODEs to use MachineName and EmailAddress!"
	print "EG. " + sys.argv[0] + " Parameters.txt 1 1 6 orhtros(or orthrosextra) hsharma@anl.gov"
	print "The parameter file must have a parameter called OldStateFolder which is the seed folder used in the previous state."
	print "MinNrSpots must be 1!!!!!"
	print "**********NOTE: For local runs, nNodes should be nCPUs.**********"
	sys.exit()

paths = open(expanduser("~")+'/.MIDAS/paths').readlines()
for line in paths:
	if 'BINFOLDER' in line:
		binfolder = line.split('=')[1].rstrip()
	if 'PFDIR' in line:
		pfdir = line.split('=')[1].rstrip()
	if 'SWIFTDIR' in line:
		swiftdir = line.split('=')[1].rstrip()

topParamFile = sys.argv[1]
if topParamFile[0] != '/':
	topParamFile = os.getcwd() + '/' + topParamFile
startLayerNr = int(sys.argv[2])
endLayerNr = int(sys.argv[3])
oldStateFolder = getValueFromParamFile(topParamFile,'OldStateFolder')[0][0]

for layerNr in range(startLayerNr,endLayerNr+1):
	os.chdir(oldStateFolder)
	folders = [s for s in glob('*/') if 'Layer'+str(layerNr) in s]
	oldFolder = folders[-1]
	PSThisLayer = topParamFile+'.Layer'+str(layerNr)+'.txt'
	shutil.copyfile(topParamFile,PSThisLayer)
	fileAppend(PSThisLayer,"OldFolder "+oldStateFolder+'/'+oldFolder+"\n")
	GrainTracking(PSThisLayer,layerNr,sys.argv[4],sys.argv[5])

