#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

type file;

app runPeaks (string paramsfn, int fnr, int ringnr)
{
 peakstracking paramsfn fnr ringnr;
}

# Parameters to be modified #############

int startnr = toInt(arg("startnr","1"));
int endnr = toInt(arg("endnr","600"));
string parameterfilestem = arg("paramsfile","/clhome/TOMO1/PeaksAnalysisHemant/PeaksFittingCode/90_33ParamsFile1.txt");
string ringfile = arg("ringfile","RingInfo.txt");
string fstm = arg("fstm","PS.txt");

# End parameters ########################

file hkl <"hkls.csv">;

int rings[] = readData(ringfile);

foreach Ring in rings {
    string PFst1 = strcat(parameterfilestem,Ring);
    string parameterfilename = strcat(PFst1,"_",fstm);
    tracef("%s\n",parameterfilename);
    foreach i in [startnr:endnr] {
       runPeaks(parameterfilename,i,Ring);
    }
}
