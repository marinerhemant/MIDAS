import numpy as np

gFile = '/home/hemantmariner/Downloads/Zr0_1Fe_3dpa_middle_mergered_filtered_globals19.gff'
grFile = '/home/hemantmariner/Downloads/Grains_3dpa_15um.csv'
grMat = np.genfromtxt(gFile,skip_header=1)
OMs = np.zeros((grMat.shape[0],10))
OMs[:,0] = np.arange(1,grMat.shape[0]+1)
OMs[:,1:] = grMat[:,9:18]
head = 'NumGrains '+str(OMs.shape[0])+'\nBeamCenter 0\nBeamCenter 0\nBeamCenter 0\nBeamCenter 0\nBeamCenter 0\nBeamCenter 0\nBeamCenter 0\nBeamCenter 0'
np.savetxt(grFile,OMs,fmt='%d %2.12f %2.12f %2.12f %2.12f %2.12f %2.12f %2.12f %2.12f %2.12f',delimiter='\t',newline='\n',header=head)
