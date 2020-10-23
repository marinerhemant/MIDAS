import numpy as np
import os

folder = '/home/beams/S1IDUSER/mnt/s1c/smaddali_oct20/ge3/trial/'
fileStem = 'Si_shard2_trench1_ff_at_Pt_nightscan'
startNr = 197
nrFiles = 4
nrFrames = 1441
padding = 6
ext = '.ge3'

outfn = folder + fileStem + '_Merged_' + str(startNr).zfill(padding) + ext
outfile = open(outfn,'wb')
header = np.fromfile(folder+fileStem+'_'+str(startNr).zfill(padding)+ext,dtype=np.uint8,count=8192)
np.array(header).tofile(outfile)

for fNr in range(startNr,startNr+nrFiles):
	fName = folder + fileStem + '_' + str(fNr).zfill(padding) + ext
	f = open(fName,'rb')
	f.seek(8192,os.SEEK_SET)
	thisData = np.fromfile(f,dtype=np.uint16)
	f.close()
	np.array(thisData).tofile(outfile)

outfile.close()
