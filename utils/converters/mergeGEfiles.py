import numpy as np
import os

folder = '/home/beams/S1IDUSER/mnt/s1c/smaddali_jun21/ge5/'
fileStem = 'test_sam1'
startNr = 108
nrFiles = 96
padding = 6
ext = '.ge5'

outfn = folder + fileStem + '_Merged_' + str(startNr).zfill(padding) + ext
outfile = open(outfn,'wb')
header = np.fromfile(folder+fileStem+'_'+str(startNr).zfill(padding)+ext,dtype=np.uint8,count=8192)
np.array(header).tofile(outfile)

for fNr in range(startNr,startNr+nrFiles):
	fName = folder + fileStem + '_' + str(fNr).zfill(padding) + ext
	print(fNr)
	f = open(fName,'rb')
	f.seek(8192,os.SEEK_SET)
	thisData = np.fromfile(f,dtype=np.uint16)
	f.close()
	np.array(thisData).tofile(outfile)

outfile.close()
