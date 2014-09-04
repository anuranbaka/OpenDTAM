// Free for non-commercial, non-military, and non-critical 
// use unless incorporated in OpenCV. 
// Inherits OpenCV Licence if in OpenCV.


#include "Optimizer.hpp"
#include "Optimizer.cuh"
#include <opencv2/gpu/stream_accessor.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace gpu;
static void memZero(GpuMat& in,Stream& cvStream);

void Optimizer::setDefaultParams(){
    
    float off=cv.layers;
    thetaStart =    200.0;
    thetaMin   =     0.1;
    thetaStep  =      .97;
    epsilon    =       .1;
    lambda     =       .01;
}

static void memZero(GpuMat& in,Stream& cvStream){
    cudaSafeCall(cudaMemsetAsync(in.data,0,in.rows*in.cols*sizeof(float),cv::gpu::StreamAccessor::getStream(cvStream)));
}

Optimizer::Optimizer(CostVolume& cv) : cv(cv), cvStream(cv.cvStream)
{
    //For performance reasons, OpenDTAM only supports multiple of 32 image sizes with cols >= 64
    CV_Assert(cv.rows % 32 == 0 && cv.cols % 32 == 0 && cv.cols >= 64);
    allocate();
    setDefaultParams();
    stableDepthEnqueued=haveStableDepth=0;
    
}
void Optimizer::attach(CostVolume& cv){
    this->cv=cv;
    cvStream=cv.cvStream;
}
#define FLATALLOC( n) n.create(1,cv.rows*cv.cols, CV_32FC1); n=n.reshape(0,cv.rows);CV_Assert(n.isContinuous())

void Optimizer::allocate(){
    FLATALLOC(_a);
    FLATALLOC(_d);
}

void Optimizer::initOptimization(){
    theta=thetaStart;
    initA();
}

void Optimizer::initA() {
//     cv.loInd.copyTo(_a,cvStream);
    cvStream.enqueueCopy(cv.loInd,_a);
}

bool Optimizer::optimizeA(const cv::gpu::GpuMat _d,cv::gpu::GpuMat _a){
    using namespace cv::gpu::device::dtam_optimizer;
    localStream = cv::gpu::StreamAccessor::getStream(cvStream);
    this->_a=_a;

    Mat tmp(cv.rows,cv.cols,CV_32FC1);
    bool doneOptimizing = theta <= thetaMin;
    int layerStep = cv.rows * cv.cols;
    float* d = (float*) _d.data;
    float* a = (float*) _a.data;

   loadConstants(cv.rows, cv.cols, cv.layers, layerStep, a, d, cv.data, (float*)cv.lo.data,
           (float*)cv.hi.data, (float*)cv.loInd.data);
    minimizeACaller  ( cv.data, a, d, cv.layers, theta,lambda);
    theta*=thetaStep;
    if (doneOptimizing){
        stableDepthReady=Ptr<char>((char*)(new cudaEvent_t));
        cudaEventCreate((cudaEvent_t*)(char*)stableDepthReady,cudaEventBlockingSync);
//         _a.convertTo(stableDepth,CV_32FC1,cv.depthStep,cv.far,cvStream);
        cvStream.enqueueConvert(_a,stableDepth,CV_32FC1,cv.depthStep,cv.far);
        cudaEventRecord(*(cudaEvent_t*)(char*)stableDepthReady,localStream);
        stableDepthEnqueued = 1;
    }
    return doneOptimizing;
}

const cv::Mat Optimizer::depthMap(){
    //Returns the best available depth map
    // Code should not rely on the particular mapping of true
    // internal data to true inverse depth, as this may change.
    // Currently depth is just a constant multiple of the index, so
    // infinite depth is always represented. This is likely to change.
    Mat tmp(cv.rows,cv.cols,CV_32FC1);
    cv::gpu::Stream str;
    if(stableDepthEnqueued){
        cudaEventSynchronize(*(cudaEvent_t*)(char*)stableDepthReady);
//         stableDepth.download(tmp,str);
        str.enqueueDownload(stableDepth,tmp);
        str.waitForCompletion();
    }else{
//         _a.download(tmp,str);
        str.enqueueDownload(_a,tmp);
        str.waitForCompletion();
        tmp = tmp * cv.depthStep + cv.far;
    }
    return tmp;
}

