#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

type file;

app (file ep) runPeaks (string ringfile, string paramfile, int fnr)
{
	peaks ringfile paramfile fnr stderr=filename(ep);
}

app (file err) runProcessPeaks (string paramsf, int RNr, file DummyA[])
{
	processPeaks paramsf RNr stderr=filename(err);
}

app (file err) mergerings (string pfname, file dummy[])
{
	mergeRings pfname stderr=filename(err);
}

app (file err, file spotsfile) postpeaks (string foldername, string pfname, file dummy, string mn)
{
	postPeaks foldername pfname filename(spotsfile) mn stderr=filename(err);
}

app (file err, file spotsfile) postpeaks2 (string foldername, string pfname)
{
	postPeaks foldername pfname filename(spotsfile) stderr=filename(err);
}

app (file err) indexrefine (string foldername, int spotsinput, file dm)
{
	indexstrains spotsinput foldername stderr=filename(err);
}

app (file err) indexrefine2 (string foldername, int spotsinput)
{
	indexstrains spotsinput foldername stderr=filename(err);
}

# Parameters to be modified #############

int startnr = toInt(arg("startnr","1"));
int endnr = toInt(arg("endnr","600"));
string ringfile = arg("ringfile","RingInfo.txt");
string seedfolder = arg("SeedFolder","/clhome/FolderNames.txt");
int dopeaksearch = toInt(arg("DoPeakSearch","1"));
string MachineName = arg("MachineName","orthrosnew");

# End parameters ########################

if (dopeaksearch == 1) {
	string folderNames[] = readData(strcat(seedfolder,"/FolderNames.txt"));
	string PFNames[] = readData(strcat(seedfolder,"/PFNames.txt"));
	int rings[] = readData(ringfile);
	iterate ix {
		string foldername = folderNames[ix];
		string paramfilenamefile = strcat(foldername,"/ParamFileNames.txt");
		string paramFileNames[] = readData(paramfilenamefile);
		file simBerr[]<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("ProcessPeaks_",ix,"_"),suffix=".err">;
		file simAerr[];
		tracef("Total number of jobs for PeakSearch: %d\n",endnr);
		foreach i in [startnr:endnr]{
			file simx<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("PeaksPerFile_",i,"_"),suffix=".err">;
			simx = runPeaks(ringfile, paramfilenamefile, i);
			if (i %% 100 == 0){
				int simAidx = (i%/100);
				simAerr[simAidx] = simx;
			}
		}
		foreach Ring2,idx2 in rings {
			string parameterfilename = paramFileNames[idx2];
			simBerr[idx2] = runProcessPeaks(parameterfilename,Ring2,simAerr);
		}
		string pfname = PFNames[ix];
		file simCerr<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("MergeRings_",ix),suffix=".err">;
		simCerr = mergerings(pfname, simBerr);
		file simDerr<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("PostPeaksSHM_",ix),suffix=".err">;
		file simCatOut<single_file_mapper;file=strcat(foldername,"/SpotsToIndexSwift.csv")>;
		(simDerr,simCatOut) = postpeaks(foldername,pfname,simCerr,MachineName);
		int spots[] = readData(simCatOut);
		tracef("Total number of remaining jobs: %d\n",length(spots));
		foreach spotnr in spots {
			file simEerr<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("IndexRefine_",ix,"_",spotnr),suffix=".err">;
			simEerr = indexrefine(foldername,spotnr,simCatOut);
		}
	} until (ix == length(folderNames));
} else {
	string folderNames[] = readData(strcat(seedfolder,"/FolderNames.txt"));
	string PFNames[] = readData(strcat(seedfolder,"/PFNames.txt"));
	tracef("%s\n",folderNames[0]);
	tracef("%s\n",PFNames[0]);
	string foldername = folderNames[0];
	string pfname = PFNames[0];
	int spots[] = readData(strcat(foldername,"/SpotsToIndex.csv"));
	tracef("Total number of remaining jobs: %d\n",length(spots));
	foreach i in spots {
		file simEerr<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("IndexRefine_",i),suffix=".err">;
		simEerr = indexrefine2(foldername,i);
	}
}
