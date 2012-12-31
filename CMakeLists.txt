cmake_minimum_required(VERSION 2.6)
project(archive-vision)


#Make the executables
#add_executable(archv-test archv-test.cpp)
#add_executable(visualWordGenerator visualWordGenerator.cpp)
#add_executable(match matching_to_many_images.cpp)
add_executable(makeContourImages makeContourImages.cpp)
add_executable(makeDictionary makeDictionary.cpp)
add_executable(makeWordHistograms makeWordHistograms.cpp)
#add_executable(peopledetect peopledetect.cpp)
#add_executable(mser mser_sample.cpp)


#try go get the link to the opencv library
FIND_PACKAGE( OpenCV REQUIRED )
#TARGET_LINK_LIBRARIES( archv-test ${OpenCV_LIBS} )
#TARGET_LINK_LIBRARIES( visualWordGenerator ${OpenCV_LIBS} )
#TARGET_LINK_LIBRARIES( match ${OpenCV_LIBS} )
TARGET_LINK_LIBRARIES( makeContourImages ${OpenCV_LIBS} )
TARGET_LINK_LIBRARIES( makeDictionary ${OpenCV_LIBS} )
TARGET_LINK_LIBRARIES( makeWordHistograms ${OpenCV_LIBS} )
#TARGET_LINK_LIBRARIES( peopledetect ${OpenCV_LIBS} )
#TARGET_LINK_LIBRARIES( mser ${OpenCV_LIBS} )