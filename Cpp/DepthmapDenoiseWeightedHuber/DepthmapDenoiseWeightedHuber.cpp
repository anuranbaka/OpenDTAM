// Free for non-commercial, non-military, and non-critical 
// use unless incorporated in OpenCV. 
// Inherits OpenCV Licence if in OpenCV.


#include "DepthmapDenoiseWeightedHuber.hpp"
#include "DepthmapDenoiseWeightedHuber.cuh"
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::cuda;
static void memZero(GpuMat& in,Stream& cvStream);

static void memZero(GpuMat& in,Stream& cvStream){
    cudaSafeCall(cudaMemsetAsync(in.data,0,in.rows*in.cols*sizeof(float),cv::cuda::StreamAccessor::getStream(cvStream)));
}
DepthmapDenoiseWeightedHuber::DepthmapDenoiseWeightedHuber(int rows,
                                                           int cols,
                                                           const cv::cuda::GpuMat& visibleLightImage,
                                                           Stream cvStream) : 
                                                           rows(rows),
                                                           cols(cols),
                                                           visibleLightImage(visibleLightImage), 
                                                           cvStream(cvStream)
{
    //For performance reasons, OpenDTAM only supports multiple of 32 image sizes with cols >= 64
    CV_Assert(rows % 32 == 0 && cols % 32 == 0 && cols >= 64);
    alloced=0;
    cachedG=0;
}

#define FLATALLOC( n) n.create(1,cv.rows*cv.cols, CV_32FC1); n=n.reshape(0,cv.rows);CV_Assert(n.isContinuous())




