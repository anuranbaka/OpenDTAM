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
void Optimizer::setDefaultParams(){
    thetaStart =    20.0;
    thetaMin   =     1.0;
    thetaStep  =      .97;
    epsilon    =       .1;
    lambda     =       .01;
}

Optimizer::Optimizer(CostVolume& cv) : cv(cv), cvStream(cv.cvStream)
{
    //For performance reasons, OpenDTAM only supports multiple of 32 image sizes with cols >= 64
    CV_Assert(cv.rows % 32 == 0 && cv.cols % 32 == 0 && cv.cols >= 64);
    allocate();
    setDefaultParams();
    stableDepthEnqueued=cachedG=haveStableDepth=0;
    
}

void Optimizer::allocate(){
    _a.create(cv.rows,cv.cols,CV_32FC1);assert(_a.isContinuous());
    _d.create(cv.rows,cv.cols,CV_32FC1);assert(_d.isContinuous());
    _qx.create(cv.rows,cv.cols,CV_32FC1);assert(_qx.isContinuous());
    _qy.create(cv.rows,cv.cols,CV_32FC1);assert(_qy.isContinuous());
    _g1.create(cv.rows,cv.cols,CV_32FC1);assert(_g1.isContinuous());
    _gx.create(cv.rows,cv.cols,CV_32FC1);assert(_gx.isContinuous());
    _gy.create(cv.rows,cv.cols,CV_32FC1);assert(_gy.isContinuous());
    stableDepth.create(cv.rows,cv.cols,CV_32FC1);
}

void Optimizer::initOptimization(){
    theta=thetaStart;
    computeSigmas();
    initA();
    initQD();
}

void Optimizer::initQD(){
    cvStream.enqueueCopy(cv.loInd,_d);
    cvStream.enqueueMemSet(_qx,0.0);
    cvStream.enqueueMemSet(_qy,0.0);
    cacheGValues();
}
void Optimizer::initA() {
//     cv.loInd.copyTo(_a);
    cvStream.enqueueCopy(cv.loInd,_a);
}
bool Optimizer::optimizeA(){
    using namespace cv::gpu::device::dtam_optimizer;
    localStream = cv::gpu::StreamAccessor::getStream(cvStream);

    bool doneOptimizing = theta <= thetaMin;
    int layerStep = cv.rows * cv.cols;
    float* d = (float*) _d.data;
    float* a = (float*) _a.data;

//    loadConstants(cv.rows, cv.cols, cv.layers, layerStep, a, d, cv.data, (float*)cv.lo.data,
//            (float*)cv.hi.data, (float*)cv.loInd.data);
    minimizeACaller  ( theta,lambda);

    theta*=thetaStep;
    if (doneOptimizing){
        stableDepthReady=(char*)(new cudaEvent_t);
        cudaEventCreate((cudaEvent_t*)(char*)stableDepthReady,cudaEventBlockingSync);
//         _a.convertTo(stableDepth,CV_32FC1,cv.depthStep,cv.far);
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
        str.enqueueDownload(stableDepth,tmp);
        str.waitForCompletion();
    }else{
        str.enqueueDownload(_a,tmp);
        str.waitForCompletion();
        tmp = tmp * cv.depthStep + cv.far;
    }
    return tmp;
}

