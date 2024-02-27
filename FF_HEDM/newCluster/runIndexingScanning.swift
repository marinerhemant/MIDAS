#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

type file;

app (file out) runIndexingScanning (string folder, int blockNr, int numBlocks, int numScans, int numProcs){
	indexScanning folder blockNr numBlocks numScans numProcs stdout=filename(out);
}

# Parameters to be supplied ###
string folder = arg("folder","");
int nrNodes = toInt(arg("nrNodes","11"));
int nScans = toInt(arg("nScans","117"));
int numProcs = toInt(arg("numProcs","32"));
# End Parameters ##############

file indexings[];
foreach nodeNr in [0:nrNodes-1] {
	file indexing<simple_mapper;location=strcat(folder,"/output"),prefix=strcat("Indexing_",nodeNr,"_"),suffix=".out">;
	indexing = runIndexingScanning(folder,nodeNr,nrNodes,nScans,numProcs);
	indexings[nodeNr] = indexing;
}