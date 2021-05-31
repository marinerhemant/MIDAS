#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

type file;

app (file out) runPeaks (string folder, string paramfn, int blockNr, int numBlocks, int nFrames, int numProcs){
	peaks folder paramfn blockNr numBlocks nFrames numProcs stdout=filename(out);
}

app (file out) runPostPeaks (string folder, string paramfn, file DummyA[]){
	postPeaks folder paramfn stdout=filename(out);
}

app (file out) runIndexRefine (string folder, int blockNr, int numBlocks, int numProcs, file dummy){
	indexRefine folder blockNr numBlocks numProcs stdout=filename(out);
}

app (file out) runProcessGrains (string folder, string paramfn, file dummyB[]){
	processGrains folder paramfn stdout=filename(out);
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
file postpeak<single_file_mapper;file=strcat(folder,"/output/PostPeaks.out")>;
postpeak = runPostPeaks(folder,paramfn,peaks);
file indexrefines[];
foreach nodeNr in [0:nrNodes-1] {
	file indexrefine<simple_mapper;location=strcat(folder,"/output"),prefix=strcat("IndexRefine_",nodeNr,"_"),suffix=".out">;
	indexrefine = runIndexRefine(folder,nodeNr,nrNodes,numProcs,postpeak);
	indexrefines[nodeNr] = indexrefine;
}
file processGrain<simple_mapper;location=strcat(folder,"/output"),prefix="ProcessGrains",suffix=".out">;
processGrain = runProcessGrains(folder,paramfn,indexrefines);
