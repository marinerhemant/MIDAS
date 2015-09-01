NF=NF_HEDM/
FF=FF_HEDM/

all: 

help: helpall helpnf helpff

helpall:
	@echo .......................................\"Global Help\"..........................................
	@echo 

helpnf: $(NF)/Makefile
	cd $(NF) && $(MAKE) help

helpff: $(FF)/Makefile
	cd $(FF) && $(MAKE) help
