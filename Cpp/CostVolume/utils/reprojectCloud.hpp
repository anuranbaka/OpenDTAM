#ifndef REPROJECT_CLOUD_HPP
#define REPROJECT_CLOUD_HPP
#include <opencv2/core/core.hpp>
struct MapInfo3d{
    cv::Mat xyin;
    cv::Mat xyout;
};

cv::Mat diagnosticInfo(const cv::Mat comparison,const cv::Mat im, const cv::Mat _depth,const cv::Mat _oldPose, const cv::Mat _newPose, const cv::Mat _cameraMatrix);
MapInfo3d reprojectCloud(const cv::Mat _depth,const cv::Mat _oldPose, const cv::Mat _newPose, const cv::Mat _cameraMatrix);



#endif