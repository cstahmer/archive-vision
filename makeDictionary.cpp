/*
 * ArchiveVision: Compter Vision Engine for Digital Archives
 * 
 * file: makeDictionary.cpp
 *
 * Contains classes and functions for building a Visual Word Dictionary
 * using a collecitons files in a command line argument designated directory.
 * Size of dictionary and output dictionary name also delivered as command line arguments
 *
 * Copyright (C) 2012 Carl Stahmer (cstahmer@gmail.com) 
 * Early Modern Center, University of California, Santa Barbara
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the Creative Commons licence, version 3.
 *    
 * See http://creativecommons.org/licenses/by/3.0/legalcode for the 
 * complete licence.
 *    
 * This program is distributed WITHOUT ANY WARRANTY; 
 * without even the implied warranty of MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for 
 * more details.
 * 
 */

#include "makeDictionary.h"

using namespace cv; 
using namespace std;

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::string;


char ch[30];


int main(int argc, char* argv[]) {
	
	Helper helper;
	string event;
	bool runInBackground = RUN_IN_BACKGROUND;
	bool writelog = WRITE_LOG;
	
	//--------Using SURF as feature extractor and FlannBased for assigning a new point to the nearest one in the dictionary
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
	Ptr<DescriptorExtractor> extractor = new SurfDescriptorExtractor();
	SurfFeatureDetector detector(300); // original value 500 documentation says somewhere between 300-500 is good depending on sharpness and contrast

	//---Initialize various objects and parameters with base values
	int dictionarySize = 2000; // originally set to 1500
	TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
	int retries = 1;
	int flags = KMEANS_PP_CENTERS;
	string trainingDirectory = TRAINING_DIR;
	string dictionaryFileName = "dictionary";

	Mat dictionary;
	
	//"Usage is -d <dirctory of training files> -n <name of dictionary output file> -s <size of dictionary> \n"
	
    for (int i = 1; i < argc; i++) { 
    	string arument = argv[i];
        if (arument == "-d") {
        	trainingDirectory = argv[i + 1];
        }
        if (arument == "-n") {
        	dictionaryFileName = argv[i + 1];
        }
        if (arument == "-s") {
        	dictionarySize = atoi(argv[i + 1]);
        }
        if (arument == "-back") {
        	runInBackground = true;
        }
        if (arument == "-log") {
        	writelog = true;
        }
        if (arument == "-help") {
            cout << "Usage is -d <dirctory of training files> -n <name of dictionary output file> -s <size of dictionary> -back [flag to run in backbround mode] -log [flag to run in log mode]"<<endl;
            exit(0);
        } 
    }
    
	string fullDictionaryFileName = dictionaryFileName + ".yml";
	
	event = "Starting makeDictionary execuatable.";
	helper.logEvent(event, 2, runInBackground, writelog);
    event = "Training Directory: " + trainingDirectory;
    helper.logEvent(event, 2, runInBackground, writelog);
    event = "Filename to use when saving dictionary: " + fullDictionaryFileName;
    helper.logEvent(event, 2, runInBackground, writelog);
    string strDictSize = static_cast<ostringstream*>( &(ostringstream() << (dictionarySize)) )->str();
    event = "Size of dictionary: " + strDictSize;
    helper.logEvent(event, 2, runInBackground, writelog);

	
	// configure BOW trainer and extractor according to set paramaters
	BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
	BOWImgDescriptorExtractor bowDE(extractor, matcher);
	
	//make sure that the training directory exists
	bool isDir = false;
	struct stat sb;
	const char * fc = trainingDirectory.c_str();
	if (stat(fc, &sb) == 0 && S_ISDIR(sb.st_mode)) {
		isDir = true;
	}
	if (!isDir) {
		event = trainingDirectory + " is not a valid directory or you do not have the correct permissions!";
	    helper.logEvent(event, 0, runInBackground, writelog);
	    event = "Operation Aborted!";
	    helper.logEvent(event, 0, runInBackground, writelog);
		exit(0);
	}
	
	event = "Calculating Centroids...";
	helper.logEvent(event, 2, runInBackground, writelog);
	// Call the collectclasscentroids function from imgextract.cpp
	collectclasscentroids(detector, extractor, bowTrainer, trainingDirectory, runInBackground, writelog);
	
	string strDescriptorCount = static_cast<ostringstream*>( &(ostringstream() << (bowTrainer.descripotorsCount())) )->str();
	event = "Clustering " + strDescriptorCount + " features.";
	helper.logEvent(event, 2, runInBackground, writelog);
	//cluster the descriptors int a dictionary
	dictionary = bowTrainer.cluster();
	
	
	event = "Saving Dictionary File";
	helper.logEvent(event, 2, runInBackground, writelog);
	helper.WriteToFile(fullDictionaryFileName, dictionary, dictionaryFileName);
	
	event = "Dictionary saved as " + fullDictionaryFileName + ".";
	helper.logEvent(event, 4, runInBackground, writelog);

	event = "makeDictionary Process Completed.";
	helper.logEvent(event, 2, runInBackground, writelog);
	event = "Bye, Bye!";
	helper.logEvent(event, 2, runInBackground, writelog);
	
	return 0;

}