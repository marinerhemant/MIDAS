#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

import sys
from subprocess import call


binFolder = sys.argv[1]
ringfile = sys.argv[2]
paramfile = sys.argv[3]
fnr = sys.argv[4]

ringsinfo = open(ringfile,'r').readlines()
paramsinfo = open(paramfile,'r').readlines()
for idx,line in enumerate(ringsinfo):
	ringnr = line.rstrip()
	paramfilename = paramsinfo[idx].rstrip()
	call([binFolder+'/PeaksFittingPerFile',paramfilename,fnr,ringnr])
	
