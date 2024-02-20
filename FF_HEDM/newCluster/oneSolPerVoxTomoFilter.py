import glob
import os,sys
utilsDir = os.path.expanduser('~/opt/MIDAS/utils/')
sys.path.insert(0,utilsDir)
from calcMiso import *
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.transform import iradon
from PIL import Image
import subprocess
import warnings
from skimage.filters import threshold_otsu
from skimage.morphology import reconstruction
import h5py
warnings.filterwarnings('ignore')
# np.set_printoptions(suppress=True,precision=3,threshold=sys.maxsize)

def runRecon(folder,startFNr,nScans,nFrames,sgnum,numProcs,nrFilesPerSweep=1,removeDuplicates=0,maxang=1,tol_ome=1,tol_eta=1,findUniques=1,thresh_reqd=0,draw_sinos=0,normalize=1):
	uniqueOrients = []
	bestConfs = []
	uniqueFNames = []
	os.chdir(folder)
	files = glob.glob(folder+'/Output/*.csv')
	nVoxels = nScans*nScans

	if findUniques==1:
		print("Finding unique grains, will print Orientation, Confidence, fileName for each unique grain.")
		print("As a better solution for that grain is found, it will be printed again.")
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
		print("Final best unique grain list: ")
		print("FileName\t\t\tBestConfidence\t\t\tOrientationMatrix")
		for lineNr in range(len(uniqueFNames)):
			line = uniqueFNames[lineNr]
			orient = uniqueOrients[lineNr]
			printVal = line+' '+str(bestConfs[lineNr])+' '+str(orient[0])+' '+str(orient[1])
			printVal += ' '+str(orient[2])+' '+str(orient[3])+' '+str(orient[4])+' '+str(orient[5])
			printVal += ' '+str(orient[6])+' '+str(orient[7])+' '+str(orient[8])
			print(printVal)
			f.write(printVal+'\n')
		f.close()

	lines = open('fNamesUniqueOrients.csv','r').readlines()
	IDsMergedScanning = np.genfromtxt('IDsMergedScanning.csv',skip_header=1,delimiter=',')
	nrhkls = 2 * (np.genfromtxt('hkls.csv',skip_header=1)).shape[0]
	nGrs = len(lines)
	pos_arr = np.zeros((nGrs*nrhkls,6))
	nrSps = np.zeros(nGrs)
	spDone = 0
	print("Now finding spots for each unique grain.")
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
			fNr = startFNr+scanNrThis*nrFilesPerSweep
			if (len(scanNrThis) == 0): 
				continue
			fNr = int(fNr)
			idsKey = np.genfromtxt(str(fNr)+'/IDRings.csv',skip_header=1)
			orig_ID = idsKey[idsKey[:,2]==origIDThis,1]
			radius_info = np.genfromtxt(str(fNr)+'/Radius_StartNr_1_EndNr_'+str(nFrames)+'.csv',skip_header=1)
			omega = radius_info[radius_info[:,0]==orig_ID,2]
			# Take uncorrected position, this will be easier to match.
			ringNr = radius_info[radius_info[:,0]==orig_ID,13]
			eta = radius_info[radius_info[:,0]==orig_ID,10]
			pos_arr[spDone][0] = omega[0]
			pos_arr[spDone][1] = eta[0]
			pos_arr[spDone][2] = ringNr[0]
			pos_arr[spDone][3] = lineNr
			pos_arr[spDone][4] = hklnr
			pos_arr[spDone][5] = spDone
			hklnr += 1
			spDone += 1
		nrSps[lineNr] = hklnr
	print(f"Total unique spots: {spDone}")

	pos_arr = pos_arr[:spDone,:]
	if removeDuplicates == 1:
		print("We were requested to remove duplicate spots.")
		dupArr = np.zeros(spDone)
		# Find duplicate and remove spots
		for sp in range(spDone):
			if dupArr[sp] == 1: continue
			pa = pos_arr[dupArr==0,:]
			pa2 = pa[sp+1:,:]
			sub_arr = pa2[pa2[:,2]==pos_arr[sp,2],:] # same ring
			sub_arr2 = sub_arr[np.fabs(sub_arr[:,0]-pos_arr[sp,0])<tol_ome,:]
			if (len(sub_arr2) ==0): continue
			sub_arr2 = sub_arr2[np.fabs(sub_arr2[:,1]-pos_arr[sp,1])<tol_eta,:]
			if (len(sub_arr2)==0): continue
			sub_arr2 = sub_arr2[0]
			dupArr[sp] = 1
			dupArr[int(sub_arr2[5])] = 1
		pos_arr = pos_arr[dupArr==0,:]
		spDone = pos_arr.shape[0]
		print(f"Total unique spots after removing duplicates: {spDone}")
		# We need to change hklnrs appropriately
		for grNr in range(nGrs):
			idxs = pos_arr[:,3].astype(np.int32) == grNr
			nSpsThisGr = np.sum(idxs)
			pos_arr[idxs,4] = np.arange(nSpsThisGr)
			nrSps[grNr] = nSpsThisGr

	np.savetxt('spot_position_arr_unique_grains.txt',pos_arr,fmt="%.6f %.6f %d %d %d %d")
	# Generate a normalization array
	if normalize==1:
		print("Generating a normalization array for sinogram intensities.")
		hkls = np.genfromtxt('hkls.csv',skip_header=1)
		nRings = int(np.max(hkls[:,4])) + 1
		norm_arr = np.zeros((nScans,nRings))
		for scanNr in range(nScans):
			fNr = startFNr + scanNr*nrFilesPerSweep
			radius_info = np.genfromtxt(str(fNr)+'/Radius_StartNr_1_EndNr_'+str(nFrames)+'.csv',skip_header=1)
			if len(radius_info) == 0: continue
			if len(radius_info.shape) == 1: continue
			for ringNr in range(nRings):
				spotsThisRing = radius_info[radius_info[:,13]==ringNr,:]
				if len(spotsThisRing) == 0: continue
				norm_arr[scanNr,ringNr] = spotsThisRing[0,16]
		norm_arr = np.where(np.max(norm_arr, axis=0)==0, norm_arr, norm_arr*1./np.max(norm_arr, axis=0))
		np.savetxt('intensity_normalization_array.txt',norm_arr,fmt="%.6f")
	else:
		norm_arr = np.ones((nScans,nScans))

	# Now go through each Radius*.csv file, find matching spots and save them in the sinogram for each grain
	print("Generating sinograms, doing a tomo recon for each grain.")
	Sinos = np.zeros((nGrs,nrhkls,nScans))
	omegas = np.zeros((nGrs,nrhkls))
	grainSpots = nrSps.astype(np.int32)
	for scanNr in range(nScans):
		fNr = startFNr + scanNr*nrFilesPerSweep
		radius_info = np.genfromtxt(str(fNr)+'/Radius_StartNr_1_EndNr_'+str(nFrames)+'.csv',skip_header=1)
		if len(radius_info)==0: continue
		if len(radius_info.shape) == 1: continue
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
				# eta_f = spots_filtered_eta[10]
			else:
				bestRow = np.argmax(spots_filtered_eta[:,15])
				intensity = spots_filtered_eta[bestRow,15]
				omega_f = spots_filtered_eta[bestRow,2]
				# eta_f = spots_filtered_eta[bestRow,10]
			intensity *= norm_arr[scanNr,int(ringNr)]
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
		if thresh_reqd == 1:
			#Thresholding
			thresh = threshold_otsu(recon)
			binary = recon <= thresh
			_,axs = plt.subplots(ncols=4, figsize=(10, 2))
			if draw_sinos==1:
				axs[0].imshow(np.transpose(recon),cmap=plt.cm.gray)
				axs[1].hist(recon.ravel(),bins=256);
				axs[1].axvline(thresh,color='r')
			recon[binary] = 0
			seed = np.copy(recon)
			seed[1:-1, 1:-1] = recon.max()
			mask = recon
			filled = reconstruction(seed, mask, method='erosion')
			if draw_sinos==1:
				axs[2].imshow(np.transpose(recon),cmap=plt.cm.gray)
				axs[3].imshow(np.transpose(filled),cmap=plt.cm.gray)
				plt.show()
			recon = filled
		all_recons[grNr,:,:] = recon
		im_list.append(Image.fromarray(recon))
		Image.fromarray(recon).save('Recons/recon_grNr_'+str.zfill(str(grNr),4)+'.tif')

	full_recon = np.max(all_recons,axis=0)
	print("Done with tomo recon, now running the optimization.")
	max_id = np.argmax(all_recons,axis=0).astype(np.int32)
	max_id[full_recon==0] = -1
	if draw_sinos == 1:
		plt.imshow(np.transpose(full_recon),cmap='gray'); plt.title('Final reconstruction.'); plt.show()
		plt.imshow(np.transpose(max_id),cmap='gray'); plt.title('GrainID.'); plt.show()
	Image.fromarray(max_id).save('Recons/Full_recon_max_project_grID.tif')
	Image.fromarray(full_recon).save('Recons/Full_recon_max_project.tif')
	im_list[0].save('Recons/all_recons_together.tif',compression="tiff_deflate",save_all=True,append_images=im_list[1:])

	# Now we use max_id to compare with the results that we had to see which orientations were found
	max_id = np.array(Image.open('Recons/Full_recon_max_project_grID.tif'))
	max_id2 = np.flipud(max_id)
	max_id2 = max_id2.flatten()
	IDArr = []
	print('Filtering voxels to generate reconstruction map.')
	print('OrigNVoxels\tFilteredNVoxels')
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
				_ = f.readline()
				_ = f.readline()
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
				if bestConf == 1.0: break
			if bestID == -1: max_id2[voxNr] = -1
			if bestID != -1:
				nrVox+=1
				IDArr.append([bestID,voxNr])
		print(grVoxels.shape[0],nrVox)
	# _,ax = plt.subplots(ncols=2)
	# ax[0].imshow(np.flipud(max_id))
	# ax[1].imshow(max_id2.reshape((nScans,nScans)))
	# plt.show()

	np.savetxt('SpotsToIndex.csv',IDArr,fmt="%d %d")
	nIDs = len(IDArr)
	os.makedirs('Results',exist_ok=True)
	subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/FitOrStrainsScanningOMP")+' paramstest.txt 0 1 '+ str(nIDs)+' '+str(numProcs),shell=True)
	
	NrSym,Sym = MakeSymmetries(sgnum)
	print(f"Filtering the final output. Will be saved to {folder}/Recons/microstrFull.csv and {folder}/Recons/microstructure.hdf")
	# go through the output
	files2 = glob.glob(folder+'/Results/*.csv')
	filesdata = np.zeros((len(files2),43))
	i=0
	info_arr = np.zeros((23,nScans*nScans))
	info_arr[:,:] = np.nan
	for fileN in files2:
		f = open(fileN)
		voxNr = int(fileN.split('.')[-2].split('_')[-2])
		_ = f.readline()
		data = f.readline().split()
		for j in range(len(data)):
			filesdata[i][j] = float(data[j])
		OM = filesdata[i][1:10]
		quat = BringDownToFundamentalRegionSym(OrientMat2Quat(OM),NrSym,Sym)
		filesdata[i][39:43] = quat
		info_arr[:,voxNr] = filesdata[i][[0,-4,-3,-2,-1,11,12,15,16,17,18,19,20,22,23,24,26,27,28,29,31,32,35]]
		i+=1
		f.close()
	head = 'SpotID,O11,O12,O13,O21,O22,O23,O31,O32,O33,SpotID,x,y,z,SpotID,a,b,c,alpha,beta,gamma,SpotID,PosErr,OmeErr,InternalAngle,'
	head += 'Radius,Completeness,E11,E12,E13,E21,E22,E23,E31,E32,E33,Eul1,Eul2,Eul3,Quat1,Quat2,Quat3,Quat4'
	np.savetxt('Recons/microstrFull.csv',filesdata,fmt='%.6f',delimiter=',',header=head)
	f = h5py.File('Recons/microstructure.hdf','w')
	micstr = f.create_dataset(name='microstr',dtype=np.double,data=filesdata)
	micstr.attrs['Header'] = np.string_(head)
	info_arr = info_arr.reshape((23,nScans,nScans))
	info_arr = np.flip(info_arr,axis=(1,2))
	info_arr = info_arr.transpose(0,2,1)
	imgs = f.create_dataset(name='images',dtype=np.double,data=info_arr)
	imgs.attrs['Header'] = np.string_('ID,Quat1,Quat2,Quat3,Quat4,x,y,a,b,c,alpha,beta,gamma,posErr,omeErr,InternalAngle,Completeness,E11,E12,E13,E22,E23,E33')
	f.close()

# runRecon('/local/s1iduser/borbely_apr17_midas',66,163,1440,225,96,nrFilesPerSweep=8,removeDuplicates=1,maxang=3,tol_ome=3,tol_eta=3,findUniques=0,draw_sinos=0,thresh_reqd=1,normalize=0)
# runRecon('/local/s1iduser/bucsek_jul22_midas/L1_new',584,117,1800,194,96,nrFilesPerSweep=1,removeDuplicates=0,maxang=3,tol_ome=3,tol_eta=3,findUniques=0,draw_sinos=0,thresh_reqd=0,normalize=1)