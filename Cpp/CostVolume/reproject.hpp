#ifndef REPROJECT_HPP
#define REPROJECT_HPP
#include <opencv2/core/core.hpp>



void reproject( const cv::Mat& src,
                const cv::Mat& cameraMatrix,
                const cv::Mat& cameraAffinePoseBase,
                const cv::Mat& cameraAffinePoseAlternate,
                float invDepth,
                cv::Mat& dst,
                cv::Mat& mask);

template <class T>//fully qualified variant
inline void reproject( const cv::Mat_<T>& src,
                const cv::Matx33d& cameraMatrix,
                const cv::Matx44d& cameraAffinePoseBase,
                const cv::Matx44d& cameraAffinePoseAlternate,
                float invDepth,
                cv::Mat_<T>& dst,
                cv::Mat_<uchar>& mask)
{
    const cv::Mat m1(src);
    const cv::Mat m2(cameraMatrix);
    const cv::Mat m3(cameraAffinePoseBase);
    const cv::Mat m4( cameraAffinePoseAlternate);


    reproject(  m1,m2,m3,m4,
                invDepth,
                (cv::Mat&)( dst),
                (cv::Mat&)(mask));
}
//Template version:
/*
template <class T>
void reproject( const cv::Mat_<T>& src,
                const cv::Matx33d& cameraMatrix,
                const cv::Matx44d& cameraAffinePoseBase,
                const cv::Matx44d& cameraAffinePoseAlternate,
                float invDepth,
                cv::Mat_<T>& dst,
                cv::Mat_<uchar>& mask);
                */
#endif
