#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

type file;
app (file out) processgrains (string foldername, string pfname)
{
	processGrains foldername pfname stdout=filename(out);
}

string seedfolder = arg("SeedFolder","/clhome/FolderNames.txt");

string folderNames[] = readData(strcat(seedfolder,"/FolderNames.txt"));
string PFNames[] = readData(strcat(seedfolder,"/PFNames.txt"));

iterate ix {
	string foldername = folderNames[ix];
	string pfname = PFNames[ix];
	file processgrainsout <single_file_mapper;file=strcat(foldername,"/output/processgrains.txt")>;
	processgrainsout = processgrains(foldername,pfname);
} until (ix == length(folderNames));
