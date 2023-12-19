import glob
from calcMiso import *
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.transform import iradon
from PIL import Image
import subprocess
import warnings
warnings.filterwarnings('ignore')

def runRecon(folder,startFNr,nScans,nFrames,sgnum,numProcs,maxang=1,tol_ome=1,tol_eta=1):
	uniqueOrients = []
	bestConfs = []
	uniqueFNames = []
	os.chdir(folder)
	files = glob.glob(folder+'/Output/*.csv')
	nVoxels = nScans*nScans

	for voxNr in range(nVoxels):
		blurb = '_'+str.zfill(str(voxNr),6)+'_'
		fns = [fn for fn in files if blurb in fn]
		if len(fns) == 0:
			continue
		PropList = []
		highestConf = -1
		for fn in fns:
			f = open(fn)
			str1= f.readline()
			str1= f.readline()
			line = f.readline().split()
			f.close()
			IAthis = float(line[0][:-1])
			OMthis = [float(a[:-1]) for a in line[1:10]]
			nExp = float(line[-2][:-1])
			nObs = float(line[-1][:-1])
			ConfThis = nObs/nExp
			idnr = int((fn.split('.')[-2]).split('_')[-1])
			# ~ print(idnr)
			if ConfThis > highestConf:
				highestConf = ConfThis
			PropList.append([ConfThis,IAthis,OMthis,idnr,voxNr])
		sortedPropList = sorted(PropList,key=lambda x: (x[0],x[1]),reverse=True)
		id_this = sortedPropList[0][3]
		fnBest = folder + '/Output/BestPos_'+str.zfill(str(voxNr),6)+'_'+str.zfill(str(id_this),9)+'.csv'
		if len(uniqueOrients) == 0:
			uniqueOrients.append(sortedPropList[0][2])
			bestConfs.append(sortedPropList[0][0])
			uniqueFNames.append(fnBest)
			print(sortedPropList[0][2],sortedPropList[0][0],fnBest)
			continue
		isFound = 0
		isBest = 0
		isBestLoc = 0
		for orientNr in range(len(uniqueOrients)):
			orient = uniqueOrients[orientNr]
			conf = bestConfs[orientNr]
			ang = rad2deg*GetMisOrientationAngleOM(orient,sortedPropList[0][2],sgnum)[0]
			if ang < maxang:
				isFound = 1
				if sortedPropList[0][0] > conf:
					isBest = 1
					isBestLoc = orientNr
		if isBest == 1:
			uniqueOrients[isBestLoc] = sortedPropList[0][2]
			bestConfs[isBestLoc] = sortedPropList[0][0]
			uniqueFNames[isBestLoc] = fnBest
			print(sortedPropList[0][2],sortedPropList[0][0],fnBest)
		if isFound == 0:
			uniqueOrients.append(sortedPropList[0][2])
			bestConfs.append(sortedPropList[0][0])
			uniqueFNames.append(fnBest)
			print(sortedPropList[0][2],sortedPropList[0][0],fnBest)
	f = open('fNamesUniqueOrients.csv','w')
	for lineNr in range(len(uniqueFNames)):
		line = uniqueFNames[lineNr]
		orient = uniqueOrients[lineNr]
		f.write(line+' '+str(bestConfs[lineNr])+' '+str(orient[0])+' '+str(orient[1])+
			' '+str(orient[2])+' '+str(orient[3])+' '+str(orient[4])+' '+str(orient[5])+
			' '+str(orient[6])+' '+str(orient[7])+' '+str(orient[8])+'\n')
	f.close()

	lines = open('fNamesUniqueOrients.csv','r').readlines()
	IDsMergedScanning = np.genfromtxt('IDsMergedScanning.csv',skip_header=1,delimiter=',')
	nrhkls = 2 * (np.genfromtxt('hkls.csv',skip_header=1)).shape[0]
	nGrs = len(lines)
	pos_arr = np.zeros((nGrs*nrhkls,5))
	nrSps = np.zeros(nGrs)
	spDone = 0
	for lineNr in range(nGrs):
		line = lines[lineNr]
		fname = line.split()[0]
		info = np.genfromtxt(fname,skip_header=3,delimiter=',')
		info = info[info[:,7].argsort()]
		IDsThis = info[:,13]
		hklnr = 0
		for idNr in range(len(IDsThis)):
			id = int(IDsThis[idNr])
			origIDThis = IDsMergedScanning[IDsMergedScanning[:,0]==id,1]
			scanNrThis = IDsMergedScanning[IDsMergedScanning[:,0]==id,2]
			fNr = startFNr+scanNrThis
			if (len(scanNrThis) == 0): 
				continue
			fNr = int(fNr)
			idsKey = np.genfromtxt(str(fNr)+'/IDRings.csv',skip_header=1)
			orig_ID = idsKey[idsKey[:,2]==origIDThis,1]
			radius_info = np.genfromtxt(str(fNr)+'/Radius_StartNr_1_EndNr_'+str(nFrames)+'.csv',skip_header=1)
			integ_intensity = radius_info[radius_info[:,0]==orig_ID,1]
			omega = radius_info[radius_info[:,0]==orig_ID,2]
			# Take uncorrected position, this will be easier to match.
			ringNr = radius_info[radius_info[:,0]==orig_ID,13]
			eta = radius_info[radius_info[:,0]==orig_ID,10]
			pos_arr[spDone][0] = omega[0]
			pos_arr[spDone][1] = eta[0]
			pos_arr[spDone][2] = ringNr[0]
			pos_arr[spDone][3] = lineNr
			pos_arr[spDone][4] = hklnr
			hklnr += 1
			spDone += 1
		nrSps[lineNr] = hklnr

	pos_arr = pos_arr[:spDone,:]
	# idx = np.lexsort((pos_arr[:,0],pos_arr[:,3]))
	# pos_arr = pos_arr[idx]
	# Now go through each Radius*.csv file, find matching spots and save them in the sinogram for each grain
	Sinos = np.zeros((nGrs,nrhkls,nScans))
	omegas = np.zeros((nGrs,nrhkls))
	grainSpots = nrSps.astype(np.int32)
	for scanNr in range(nScans):
		fNr = startFNr + scanNr
		radius_info = np.genfromtxt(str(fNr)+'/Radius_StartNr_1_EndNr_'+str(nFrames)+'.csv',skip_header=1)
		if len(radius_info)==0: continue
		# print(radius_info.shape)
		# Find all spots matching the spots that interest us.
		for spotNr in range(spDone):
			omega = pos_arr[spotNr][0]
			eta = pos_arr[spotNr][1]
			ringNr = pos_arr[spotNr][2]
			grNr = int(pos_arr[spotNr][3])
			grSp = int(pos_arr[spotNr][4])
			spots_filtered = radius_info[radius_info[:,13]==ringNr,:]
			if len(spots_filtered) == 0: continue
			spots_filtered_omega = spots_filtered[np.abs(spots_filtered[:,2]-omega)<=tol_ome,:]
			if len(spots_filtered_omega) == 0: continue
			spots_filtered_eta = spots_filtered_omega[np.abs(spots_filtered_omega[:,10]-eta)<=tol_eta,:]
			if len(spots_filtered_eta) == 0: continue
			if len(spots_filtered_eta.shape) == 1:
				intensity = spots_filtered_eta[15]
				omega_f = spots_filtered_eta[2]
				eta_f = spots_filtered_eta[10]
			else:
				bestRow = np.argmax(spots_filtered_eta[:,1])
				intensity = spots_filtered_eta[bestRow,15]
				omega_f = spots_filtered_eta[bestRow,2]
				eta_f = spots_filtered_eta[bestRow,10]
			Sinos[grNr,grSp,scanNr] = intensity
			omegas[grNr,grSp] = omega_f

	os.makedirs('Sinos',exist_ok=True)
	os.makedirs('Thetas',exist_ok=True)
	os.makedirs('Recons',exist_ok=True)

	all_recons = np.zeros((nGrs,nScans,nScans))
	im_list = []
	for grNr in range(nGrs):
		nSp = grainSpots[grNr]
		thetas = omegas[grNr,:nSp]
		sino = np.transpose(Sinos[grNr,:nSp,:])
		Image.fromarray(sino).save('Sinos/sino_grNr_'+str.zfill(str(grNr),4)+'.tif')
		np.savetxt('Thetas/thetas_grNr_'+str.zfill(str(grNr),4)+'.txt',thetas,fmt='%.6f')
		recon = iradon(sino,theta=thetas)
		all_recons[grNr,:,:] = recon
		im_list.append(Image.fromarray(recon))
		Image.fromarray(recon).save('Recons/recon_grNr_'+str.zfill(str(grNr),4)+'.tif')

	full_recon = np.max(all_recons,axis=0)
	max_id = np.argmax(all_recons,axis=0).astype(np.int32)
	Image.fromarray(max_id).save('Recons/Full_recon_max_project_grID.tif')
	Image.fromarray(full_recon).save('Recons/Full_recon_max_project.tif')
	im_list[0].save('Recons/all_recons_together.tif',compression="tiff_deflate",save_all=True,append_images=im_list[1:])

	# Now we use max_id to compare with the results that we had to see which orientations were found
	max_id = np.array(Image.open('Recons/Full_recon_max_project_grID.tif'))
	max_id2 = np.flipud(max_id)
	max_id2 = max_id2.flatten()
	IDArr = []
	for grNr in range(nGrs):
		grVoxels = (max_id2==grNr).nonzero()[0]
		line = lines[grNr].split('\n')[0]
		orient = np.array([float(val) for val in line.split()[-9:]])
		nrVox = 0
		# Now we know the voxels belonging to this grain, go through the files list, read the BestPos_voxNr_*.csv and get the correct orientation files
		for voxNr in grVoxels:
			bestConf = -1
			bestID = -1
			blurb = '_'+str.zfill(str(voxNr),6)+'_'
			fns = [fn for fn in files if blurb in fn]
			for fn in fns:
				f = open(fn)
				str1= f.readline()
				str1= f.readline()
				line = f.readline().split()
				f.close()
				OMthis = [float(a[:-1]) for a in line[1:10]]
				nExp = float(line[-2][:-1])
				nObs = float(line[-1][:-1])
				ConfThis = nObs/nExp
				ang = rad2deg*GetMisOrientationAngleOM(orient,OMthis,sgnum)[0]
				if ang < maxang and ConfThis > bestConf:
					bestConf = ConfThis
					bestID = int((fn.split('.')[-2]).split('_')[-1])
			if bestID != -1:
				nrVox+=1
				IDArr.append([bestID,voxNr])
		print(grVoxels.shape[0],nrVox)

	print(len(IDArr))
	np.savetxt('SpotsToIndex.csv',IDArr,fmt="%d %d")
	nIDs = len(IDArr)
	os.makedirs('Results',exist_ok=True)
	subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/FitOrStrainsScanningOMP")+' paramstest.txt 0 1 '+ str(nIDs)+' '+str(numProcs),shell=True)
	NrSym,Sym = MakeSymmetries(sgnum)
	# go through the output
	files2 = glob.glob(folder+'/Results/*.csv')
	filesdata = np.zeros((len(files2),43))
	i=0
	for fileN in files2:
		f = open(fileN)
		str1 = f.readline()
		data = f.readline().split()
		for j in range(len(data)):
			filesdata[i][j] = float(data[j])
		OM = filesdata[i][1:10]
		quat = BringDownToFundamentalRegionSym(OrientMat2Quat(OM),NrSym,Sym)
		filesdata[i][39:43] = quat
		i+=1
		f.close()
	np.savetxt('microstrFull.csv',filesdata,fmt='%.6f',delimiter=',',header='SpotID,O11,O12,O13,O21,O22,O23,O31,O32,O33,SpotID,x,y,z,SpotID,a,b,c,alpha,beta,gamma,SpotID,PosErr,OmeErr,InternalAngle,Radius,Completeness,E11,E12,E13,E21,E22,E23,E31,E32,E33,Eul1,Eul2,Eul3,Quat1,Quat2,Quat3,Quat4')
