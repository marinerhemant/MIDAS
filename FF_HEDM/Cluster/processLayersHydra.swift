#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

type file;

app (file ep) runPeaks (string paramsfn, int fnr, int ringnr)
{
	peaks paramsfn fnr ringnr stdout=filename(ep);
}

app (file err) runProcessPeaks (string paramsf, int RNr, file DummyA[])
{
	processPeaks paramsf RNr stdout=filename(err);
}

app (file err) mergerings (string pfname, file dummy[])
{
	mergeRings pfname stdout=filename(err);
}

app (file err, file spotsfile) postpeaks (string foldername, string pfname, file dummy)
{
	postPeaksHydra foldername pfname filename(spotsfile) stderr=filename(err);
}

app (file err) indexrefine (string foldername, int spotsinput, file dm)
{
	indexstrains spotsinput foldername stderr=filename(err);
}

app (file out) processgrains (string foldername, string pfname, file dummy[])
{
	processGrains foldername pfname stdout=filename(out);
}

app (file err) mergedetectors (string foldername, string layernr, file dummy[])
{
	mergeDetectors foldername layernr stderr=filename(err);
}

# Parameters to be modified #############

int startnr = toInt(arg("startnr","1"));
int endnr = toInt(arg("endnr","600"));
int startlayernr = toInt(arg("startLayer","1"));
int endlayernr = toInt(arg("endLayer","1"));
string ringfile = arg("ringfile","RingInfo.txt");
string seedfolder = arg("SeedFolder","/clhome/FolderNames.txt");

# End parameters ########################

int rings[] = readData(ringfile);
string folderNames[] = readData(strcat(seedfolder,"/FolderNames.txt"));

iterate ix {
	string foldername = folderNames[ix];
	int layernr = ix + startlayernr;
	tracef("Layer %d\n",layernr);
	file simAerr[];
	foreach detnr in [1:4]{
		string paramfilenamefile = strcat(foldername,"/Detector",detnr,"/ParamFileNames.txt");
		string paramFileNames[] = readData(paramfilenamefile);
		foreach Ring,idx in rings {
			string parameterfilename = paramFileNames[idx];
			string PreFix1 = strcat("PeaksPerFile_",Ring);
			foreach i in [startnr:endnr] {
				file simx<simple_mapper;location=strcat(foldername,"/Detector",detnr,"/output"),prefix=strcat(PreFix1,"_",i,"_"),suffix=".err">;
				simx = runPeaks(parameterfilename,i,Ring);
				if (i %% 100 == 0){
					int simAidx = (detnr-1)*length(rings)*(endnr%/100) + idx*(endnr%/100) + (i%/100);
					simAerr[simAidx] = simx;
				}
			}
		}
	}
	# Hack to get things to be 
	file simBerr[]<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("ProcessPeaks_",ix,"_"),suffix=".err">;
	file simCerr[]<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("MergeRings_",ix,"_"),suffix=".err">;
	foreach detnr2 in [1:4]{
		string paramfilenamefile2 = strcat(foldername,"/Detector",detnr2,"/ParamFileNames.txt");
		string paramFileNames2[] = readData(paramfilenamefile2);
		foreach Ring2,idx2 in rings {
			string parameterfilename2 = paramFileNames2[idx2];
			simBerr[idx2+(detnr2*length(rings))] = runProcessPeaks(parameterfilename2,Ring2,simAerr);
		}
	}
	foreach detnr3 in [1:4]{
		string pfname = strcat(foldername,"/Detector",detnr3,"/Layer",layernr,"_MultiRing_ps.txt");
		simCerr[detnr3] = mergerings(pfname, simBerr);
	}
	# Now merge peaks from the detectors
	file simDerr<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("MergeDetectors",ix),suffix=".err">;
	simDerr = mergedetectors(foldername,layernr,simCerr);
	# SHM
	file simEerr<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("PostPeaksSHM_",ix),suffix=".err">;
	file simCatOut<single_file_mapper;file=strcat(foldername,"/SpotsToIndexSwift.csv")>;
	(simEerr,simCatOut) = postpeaks(foldername,"hydra",simDerr);
	int spots[] = readData(simCatOut);
	tracef("Total number of remaining jobs: %d\n",length(spots));
	foreach spotnr in spots {
		file simFerr<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("IndexRefine_",ix,"_",spotnr),suffix=".err">;
		simEerr = indexrefine(foldername,spotnr,simCatOut);
	}
}until (ix == length(folderNames));
