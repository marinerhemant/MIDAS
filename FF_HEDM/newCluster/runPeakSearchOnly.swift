#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

type file;

app (file out) runPeaks (string folder, string paramfn, int blockNr, int numBlocks, int nFrames, int numProcs){
	peaks folder paramfn blockNr numBlocks nFrames numProcs stdout=filename(out);
}

# Parameters to be supplied ###
string folder = arg("folder","");
string paramfn = arg("paramfn","");
int nrNodes = toInt(arg("nrNodes","11"));
int nFrames = toInt(arg("nFrames","1440"));
int numProcs = toInt(arg("numProcs","32"));
# End Parameters ##############

file peaks[];
foreach nodeNr in [0:nrNodes-1] {
	file peak<simple_mapper;location=strcat(folder,"/output"),prefix=strcat("Peaks_",nodeNr,"_"),suffix=".out">;
	peak = runPeaks(folder,paramfn,nodeNr,nrNodes,nFrames,numProcs);
	peaks[nodeNr] = peak;
}
