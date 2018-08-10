#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

NF=NF_HEDM/
FF=FF_HEDM/

NLOPT=""
ifneq ($(NLOPT),"")
	FLAGSNLOPT=NLOPT="$(NLOPT)"
else
	FLAGSNLOPT=
endif
TIFF=""
ifneq ($(TIFF),"")
	FLAGSTIFF=TIFF="$(TIFF)"
else
	FLAGSTIFF=
endif

nall: helpall
	@echo Successfully compiled.

makenf:
	cd $(NF) && $(MAKE) clean
	@echo $(TIFF) $(NLOPT)
	cd $(NF) && $(MAKE) all $(FLAGSNLOPT) $(FLAGSTIFF)

makeff:
	cd $(FF) && $(MAKE) clean
	@echo $(NLOPT)
	cd $(FF) && $(MAKE) all $(FLAGSNLOPT)

help: helpall

helpall:
	@echo
	@echo .......................................\"Global Help\"..........................................
	@echo ....... Either install individually by going into subfolders, or install directly.............
	@echo ............................ \"make all\" to compile both codes................................. 
	@echo ........................Need the NLOPT package and libtiff5-dev package.......................
	@echo ....................Provide paths for NLOPT and TIFF if not installed as su...................
	@echo .............eg. on orthros: NLOPT=\"/clhome/TOMO1/PeaksAnalysisHemant/NF_HEDM/NLOPT/\".........
	@echo

helpnf: $(NF)/Makefile
	cd $(NF) && $(MAKE) help

helpff: $(FF)/Makefile
	cd $(FF) && $(MAKE) help
