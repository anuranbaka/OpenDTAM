#ifndef REPROJECT_HPP
#define REPROJECT_HPP   
#include <opencv2/core/core.hpp>
template <class T>
void reproject( const cv::Mat_<T>& src,
                const cv::Matx33d& cameraMatrix,
                const cv::Matx44d& cameraAffinePoseBase,
                const cv::Matx44d& cameraAffinePoseAlternate,
                float invDepth,
                cv::Mat_<T>& dst,
                cv::Mat_<bool>& mask);
void reproject( const cv::Mat& src,
                const cv::Matx33d& cameraMatrix,
                const cv::Mat& cameraAffinePoseBase,
                const cv::Mat& cameraAffinePoseAlternate,
                float invDepth,
                cv::Mat& dst,
                cv::Mat& mask);
#endif