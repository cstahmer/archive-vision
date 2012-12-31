/*
 * ArchiveVision: Compter Vision Engine for Digital Archives
 * 
 * file: imageextract.cpp
 *
 * Contains classes and functions for extracting descriptor
 * information from images
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

#include "imgextract.h"

using namespace std;
using namespace cv;

using std::vector;


void collectclasscentroids(SurfFeatureDetector &detector, Ptr<DescriptorExtractor> &extractor, BOWKMeansTrainer &bowTrainer, string trainingDir, bool runInBackground, bool writelog) {
	
	IplImage *img;
	vector<string> files = vector<string>();
	Helper helper;
	string event;
	char ch[30];
	
	// should put error correction here to check if directory exists
	
	helper.GetFileList(trainingDir, files);

    for (unsigned int iz = 0;iz < files.size();iz++) {
    	int isImage = helper.instr(files[iz], "jpg", 0, true);
        if (isImage > 0) {
        	
   
        	string sFileName = trainingDir;
        	string sFeaturesDir = "/usr/local/share/archive-vision/build/features/";
        	string sOutputImageFilename = "/usr/local/share/archive-vision/build/feature_point_images/";
        	sFileName.append(files[iz]);
        	sOutputImageFilename.append(files[iz]);
        	sFeaturesDir.append(files[iz]);
        	sFeaturesDir.append(".txt");
        	const char * imageName = sFileName.c_str ();
        	
			img = cvLoadImage(imageName,0);
			if (img) {
				string workingFile = files[iz];
				vector<KeyPoint> keypoint;
				detector.detect(img, keypoint);
				if (keypoint.size()) {
					Mat features;
					extractor->compute(img, keypoint, features);
					
					event = "Processing " + workingFile;
					helper.logEvent(event, 2, runInBackground, writelog);
	
					
					//try to write out an image with the features highlighted
					// Add results to image and save.
	//				Mat output;
	//				drawKeypoints(img, keypoint, output, Scalar(0, 128, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//				imwrite(sOutputImageFilename, output);
					
	
					
					
					// try writing out all the feature, each to its own YML file and see what
					// they look like 
	//				helper.WriteToFile(sFeaturesDir, features, "features");
					
					bowTrainer.add(features);
				} else {
					event = workingFile + "contains no keypoints.";
					helper.logEvent(event, 1, runInBackground, writelog);
				}
			}
			
			
        }
    }
	return;
}


/*
 * Note, if I want to class images, I need to do some work on how lables are generated.  Perhaps lookup
 * in a db a class (woodcut_group) for each file name and submit this as the label
 */

vector<Mat> getHistAndLabels(SurfFeatureDetector &detector, BOWImgDescriptorExtractor &bowDE, int dictionarySize) {
	
	// setup variable and object I need
	IplImage *img2;
	Mat labels(0, 1, CV_32FC1);
	Mat trainingData(0, dictionarySize, CV_32FC1);
	vector<KeyPoint> keypoint1;
	Mat bowDescriptor1;
	Helper helper;
	vector<string> files = vector<string>();
	
	helper.GetFileList(EVAL_DIR, files);

	float labelVal;

    for (unsigned int iz = 0;iz < files.size();iz++) {
    	int isImage = helper.instr(files[iz], "jpg", 0, true);
        if (isImage > 0) {
        	string sFileName = TRAINING_DIR;
        	sFileName.append(files[iz]);
        	const char * imageName = sFileName.c_str ();
        	
			img2 = cvLoadImage(imageName,0);
			if (img2) {
				detector.detect(img2, keypoint1);
				bowDE.compute(img2, keypoint1, bowDescriptor1);
				trainingData.push_back(bowDescriptor1);
				labelVal = iz+1;
				labels.push_back(labelVal);
			}
			
			
        }
    }	
	
	vector<Mat> retVec;
	retVec.push_back(trainingData);
	retVec.push_back(labels);
	return retVec;
	
}


Mat getHistograms(SurfFeatureDetector &detector, BOWImgDescriptorExtractor &bowDE, int dictionarySize, vector<string> &collectionFilenames, string evalDir) {
	
	// setup variable and object I need
	IplImage *img2;
	Mat trainingData(0, dictionarySize, CV_32FC1);
	vector<KeyPoint> keypoint1;
	Mat bowDescriptor1;
	Helper helper;
	vector<string> files = vector<string>();	
	
	helper.GetFileList(evalDir, files);
	
	cout << "Number of Collection Files to Process: " << files.size()-2 << endl;

    for (unsigned int iz = 0;iz < files.size();iz++) {
    	int isImage = helper.instr(files[iz], "jpg", 0, true);
        if (isImage > 0) {
        	cout << "     Processing " << files[iz] << endl;
        	
        	collectionFilenames.push_back(files[iz]);
        	string sFileName = EVAL_DIR;
        	sFileName.append(files[iz]);
        	const char * imageName = sFileName.c_str ();
        	
			img2 = cvLoadImage(imageName,0);
			if (img2) {
				detector.detect(img2, keypoint1);
				bowDE.compute(img2, keypoint1, bowDescriptor1);
				trainingData.push_back(bowDescriptor1);
			}
			
			
        }
    }	
	
	return trainingData;	
}
Mat getSingleImageHistogram(SurfFeatureDetector &detector, BOWImgDescriptorExtractor &bowDE, string evalFile) {
	
	// setup variable and object I need
	IplImage *img2;
	vector<KeyPoint> keypoint1;
	Mat bowDescriptor1;
	Helper helper;

    
	int isImage = helper.instr(evalFile, "jpg", 0, true);
    if (isImage > 0) {
 
    	const char * imageName = evalFile.c_str ();
		img2 = cvLoadImage(imageName,0);
		if (img2) {
			detector.detect(img2, keypoint1);
			bowDE.compute(img2, keypoint1, bowDescriptor1);
		}	
    }	
	
	return bowDescriptor1;	
}


float getClassMatch(SurfFeatureDetector &detector, BOWImgDescriptorExtractor &bowDE, IplImage* &img2, int dictionarySize, string sFileName, CvSVM &svm) {
	float response;
	
	vector<KeyPoint> keypoint2;
	Mat bowDescriptor2;
	Mat evalData(0, dictionarySize, CV_32FC1);
	Mat groundTruth(0, 1, CV_32FC1);
	Mat results(0, 1, CV_32FC1);
	
	
	detector.detect(img2, keypoint2);
	bowDE.compute(img2, keypoint2, bowDescriptor2);
	
	
	//evalData.push_back(bowDescriptor2);
	//groundTruth.push_back((float) classID);
	response = svm.predict(bowDescriptor2);
	//results.push_back(response);

	
	return response;
}