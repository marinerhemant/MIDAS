#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

type file;

app (file out, file err) Medians (string pf, int layernr, file trial)
{
	runmedianparallel pf layernr stdout=@filename(out) stderr=@filename(err);
}

app (file outm, file errm) Images (string paramf, int layern, int filenr, file inp)
{
	runimageprocessingparallel paramf layern filenr stdout=@filename(outm) stderr=@filename(errm);
}

app (file done) PlaceHolder (string prefix, file out[])
{
	echo prefix stdout=@filename(done);
}

app (file done) PlaceHolder2 (string prefix)
{
	echo prefix stdout=@filename(done);
}

app (file outf, file errf) runfitorientation (string pf, int nr, file mmapdone)
{
	fitorientation pf nr stdout=@filename(outf) stderr=@filename(errf);
}

app (file mmapdone) mmapcode (string paramfn, file imagedone)
{
	mmaps paramfn dire stdout=@filename(mmapdone);
}

app parsemic (string paramfn, file trial[])
{
	micparser paramfn;
}

app (file done) initialsetup ( string paramfn, int ffseed)
{
	setupNF paramfn ffseed stdout=@filename(done);
}

type BulkNames{
	string paramfn;
}

# Parameters to be modified ############

string paramf = arg("FileData","/data/tomo1/NFTest/ParametersGoldApril14.txt");
int NrLayers = toInt(arg("NrDistances","3"));
int NrFilesPerLayer = toInt(arg("NrFilesPerDistance","180"));
int startnr = toInt(arg("StartNumber","1"));
int endnr = toInt(arg("EndNumber","2000"));
int DoPeakSearch = toInt(arg("DoPeakSearch","1"));
int ffseed = toInt(arg("FFSeedOrientations","1"));

# End parameters #######################

# Read data
Bulknames NameData[] = readData(paramf);

foreach dat in NameData {
	string paramfile = dat.paramfn;
	
	# Do the initial setup
	file setupdone <"output/initialsetup.csv">;
	setupdone = initialsetup(paramfile,ffseed);
	
	## Whether do peak search or not
	file imagesdone <"output/imageprocessing.txt">;
	if (DoPeakSearch == 1){
		trace("Doing peaksearch.\n");
		string prefix2 = strcat("ImageProcessing_");
		file simBout[]<simple_mapper;location="output",prefix=prefix2,suffix=".out">;
		file simBerr[]<simple_mapper;location="output",prefix=prefix2,suffix=".err">;
		foreach layer in [1:NrLayers] {
			string prefix1 = strcat("Median_",layer);
			file simAout <simple_mapper;location="output",prefix=prefix1,suffix=".out">;
			file simAerr <simple_mapper;location="output",prefix=prefix1,suffix=".err">;
			(simAout,simAerr) = Medians(paramfile,layer,setupdone);
			foreach FileNr in [0:(NrFilesPerLayer-1)]{
				(simBout[(layer-1)*NrFilesPerLayer + FileNr],simBerr[(layer-1)*NrFilesPerLayer + FileNr]) = Images(paramfile, layer, FileNr,simAout);
			}
		}
		imagesdone = PlaceHolder("Done",simBout);
	} else {
		trace("Not doing peaksearch.\n");
		string prefix2 = "ImageProcessing was not done";
		imagesdone = PlaceHolder2(prefix2);
	}
	
	## Now MMap Images
	tracef("%s\n",imagesdone);
	file mmapdone <"output/mmapdone.txt">;
	mmapdone = mmapcode(paramfile,imagesdone);
	
	## Now do FitOrientation
	file errfit[]<simple_mapper;location="output",prefix="fitorient",suffix=".err">;
	file outfit[]<simple_mapper;location="output",prefix="fitorient",suffix=".out">;
	foreach i in [startnr:endnr] {
		(outfit[i],errfit[i])runfitorientation(paramfile,i,mmapdone);
	}
	
	# Now parse mic file
	parsemic(paramfile,outfit)
}
