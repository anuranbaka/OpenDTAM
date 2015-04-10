#ifndef REPROJECT_CLOUD_HPP
#define REPROJECT_CLOUD_HPP
#include <opencv2/core/core.hpp>


cv::Mat reprojectCloud(const cv::Mat comparison,const cv::Mat im, const cv::Mat _depth,const cv::Mat _oldPose, const cv::Mat _newPose, const cv::Mat _cameraMatrix);

#endif