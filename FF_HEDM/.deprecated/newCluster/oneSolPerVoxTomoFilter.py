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
import argparse

class MyParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

def runRecon(folder,nScans,sgnum,numProcs,numProcsLocal,maxang=1,tol_ome=1,tol_eta=1,thresh_reqd=0,draw_sinos=0,nNodes=1,machineName='local'):

	subprocess.call(os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/findSingleSolutionPF')+f' {folder} {sgnum} {maxang} {nScans} {numProcsLocal} {tol_ome} {tol_eta}',cwd=folder,shell=True)

	sinoFN = glob.glob("sinos_*.bin")[0]
	nGrs = int(sinoFN.split('_')[1])
	maxNHKLs = int(sinoFN.split('_')[2])
	Sinos = np.fromfile(sinoFN,dtype=np.double,count=nGrs*maxNHKLs*nScans).reshape((nGrs,maxNHKLs,nScans))
	omegas = np.fromfile(f"omegas_{nGrs}_{maxNHKLs}.bin",dtype=np.double,count=nGrs*maxNHKLs).reshape((nGrs,maxNHKLs))
	grainSpots = np.fromfile(f"nrHKLs_{nGrs}.bin",dtype=np.int32,count=nGrs)

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
				axs[1].hist(recon.ravel(),bins=256)
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

	# Now we have the IDs for each grain, let's find out the orientations, generate info for refinement
	# Tahe max_id, go through each voxel and find the ones that should be done for each voxel
	# write a SpotsToIndex.csv with the full info
	uniqueOrientations = np.genfromtxt(f'{folder}/UniqueOrientations.csv',delimiter=' ')
	max_id2 = np.flipud(max_id)
	max_id2 = max_id2.flatten() # max_id2 has the rowNr, starting from 0. TODO: confirm this
	f = open(f'{folder}/SpotsToIndex.csv','w')
	for voxNr in range(nScans*nScans):
		if max_id2[voxNr] == -1:
			continue
		orientThis = uniqueOrientations[max_id2[voxNr],5:14]
		if os.path.isfile(f'{folder}/Output/UniqueIndexKeyOrientAll_voxNr_{voxNr}.txt'):
			with open(f'{folder}/Output/UniqueIndexKeyOrientAll_voxNr_{voxNr}.txt','r') as f:
				lines = f.readlines()
			for line in lines:
				orientInside = [float(val) for val in line.split()[5:14]]
				ang = rad2deg*GetMisOrientationAngleOM(orientThis,orientInside,sgnum)[0]
				if ang < maxang:
					print(line)
					f.write(line)
					break
	f.close()

	IDs = np.genfromtxt(f"{folder}/SpotsToIndex.csv")
	nIDs = IDs.shape[0]
	os.makedirs('Results',exist_ok=True)
	swiftcmdIdx = os.path.expanduser('~/.MIDAS/swift/bin/swift') + ' -config ' + os.path.expanduser('~/opt/MIDAS/FF_HEDM/newCluster/sites.conf') 
	swiftcmdIdx += ' -sites ' + machineName + ' ' + os.path.expanduser('~/opt/MIDAS/FF_HEDM/newCluster/runRefiningScanning.swift') + ' -folder=' 
	swiftcmdIdx += folder + ' -nrNodes=' + str(nNodes) + ' -nIDs=' + str(nIDs) + ' -numProcs='+ str(numProcs)
	print(swiftcmdIdx)
	subprocess.call(swiftcmdIdx,shell=True)
	
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

if __name__ == "__main__":
	parser = MyParser(description='''Find one solution per voxel, then refine''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-resultFolder', type=str, required=True, help='Working folder.')
	parser.add_argument('-nScans', type=int, required=True, help='Number of scans.')
	parser.add_argument('-positionsFile', type=str, required=False, default='positions.csv', help='Positions file.')
	parser.add_argument('-machineName', type=str, required=False, default='local', help='Machine to run analysis on.')
	parser.add_argument('-sgNum', type=int, required=True, help='SpaceGroup number.')
	parser.add_argument('-nCPUs', type=int, required=True, help='Number of CPU cores to use.')
	parser.add_argument('-startFNr', type=int, required=True, help='Start file number for the first scan. This is used to compare peaksearch results.')
	parser.add_argument('-nFrames', type=int, required=True, help='Number of frames per scan.')
	parser.add_argument('-nrFilesPerSweep', type=int, required=False, default=1, help='Difference in fileNr between scans.')
	parser.add_argument('-removeDuplicates', type=int, required=False, default=0, help='If you want to remove duplicate spots or not. This is useful for removing shared spots for eg. twins.')
	parser.add_argument('-findUniques', type=int, required=False, default=1, help='If you want to skip finding unique grains again, if already done, put to 0.')
	parser.add_argument('-threshReqd', type=int, required=False, default=0, help="If you want to do thresholding in sinograms using Otsu's method.")
	parser.add_argument('-drawSinos', type=int, required=False, default=0, help='If you want to plot the Sinos to check interactively.')
	parser.add_argument('-normalize', type=int, required=False, default=1, help='If you want to normalize the intensity of sinograms using powder intensity for rings.')
	parser.add_argument('-nNodes', type=int, required=False, default=1, help='Number of nodes to use, if not using local.')
	parser.add_argument('-maxAng', type=float, required=False, default=1.0, help='Maximum angle in degrees to qualify as a different orientation.')
	parser.add_argument('-tolOme', type=float, required=False, default=1.0, help='Maximum tolerance in omega when looking for spots to make sinograms.')
	parser.add_argument('-tolEta', type=float, required=False, default=1.0, help='Maximum tolerance in eta when looking for spots to make sinograms.')
	args, unparsed = parser.parse_known_args()
	resultDir = args.resultFolder
	nScans = args.nScans
	sgnum = args.sgNum
	numProcs = args.nCPUs
	maxang = args.maxAng
	tol_ome = args.tolOme
	tol_eta = args.tolEta
	thresh_reqd = args.threshReqd
	draw_sinos = args.drawSinos
	nNodes = args.nNodes
	machineName = args.machineName
	numProcsLocal = 3
	runRecon(resultDir,nScans,sgnum,numProcs,numProcsLocal,maxang,tol_ome,tol_eta,thresh_reqd,draw_sinos,nNodes,machineName)


# runRecon('/local/s1iduser/borbely_apr17_midas',66,163,1440,225,96,nrFilesPerSweep=8,removeDuplicates=1,maxang=3,tol_ome=3,tol_eta=3,findUniques=0,draw_sinos=0,thresh_reqd=1,normalize=0)
# runRecon('/local/s1iduser/bucsek_jul22_midas/L1_new',584,117,1800,194,96,nrFilesPerSweep=1,removeDuplicates=0,maxang=3,tol_ome=3,tol_eta=3,findUniques=0,draw_sinos=0,thresh_reqd=0,normalize=1)