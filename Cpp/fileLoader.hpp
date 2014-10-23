#ifndef FILE_LOADER
#define FILE_LOADER

// loads all files of a given name and extension
#include "convertAhandaPovRayToStandard.h"
void loadAhanda(const char * rootpath,
                int imageNumber,
                cv::Mat& image,
                cv::Mat& d,
                cv::Mat& cameraMatrix,
                cv::Mat& R,
                cv::Mat& T);

#endif