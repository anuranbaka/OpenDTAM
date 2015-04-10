#ifndef TRACK_HPP
#define TRACK_HPP
#include <opencv2/opencv.hpp>
#include <CostVolume/Cost.h>
#include <CostVolume/CostVolume.hpp>
#include <DepthmapDenoiseWeightedHuber/DepthmapDenoiseWeightedHuber.hpp>

class Track{
public:
    void align();
    void align_gray(cv::Mat& base, cv::Mat& depth, cv::Mat& input);
    cv::Mat cameraMatrix;
    int rows;
    int cols;
    cv::Mat dMdp;
    cv::Mat gradBase;
    cv::Mat baseImage;
    cv::Mat basePose;
    cv::Mat depth;
    cv::Mat pose;
    cv::Mat thisFrame;
    cv::Mat lastFrame;
    
    Track(Cost cost);
    Track(CostVolume cost);
    void addFrame(cv::Mat frame);
    void ESM();
    void cacheDerivatives();

private:
    //Alignment Functions
    
    //Large deformation, forward mapping, 6DoF
    bool align_level_largedef_gray_forward(const cv::Mat& T,
                                           const cv::Mat& d,
                                           const cv::Mat& _I,
                                           const cv::Mat& cameraMatrix,//Mat_<double>
                                           const cv::Mat& _p,                //Mat_<double>
                                           int mode,
                                           float threshold,
                                           int numParams);
    
};




#endif
