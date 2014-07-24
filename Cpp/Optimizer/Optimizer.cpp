// Free for non-commercial, non-military, and non-critical 
// use unless incorporated in OpenCV. 
// Inherits OpenCV Licence if in OpenCV.


#include "Optimizer.hpp"
#include "Optimizer.cuh"
void Optimizer::setDefaultParams(){
    thetaStart =    500.0;
    thetaMin   =     0.01;
    thetaStep  =      .99;
    epsilon    =       .1;
    lambda     =  .000001;
}

Optimizer::Optimizer(CostVolume cv) : cv(cv)
{
    setDefaultParams();
}

void Optimizer::initOptimization(){
    theta=thetaStart;
    initA();
    initQD();

}
void Optimizer::initQD(){
    _a.copyTo(_d);
    _qx.create(cv.rows,cv.cols,CV_32FC1);
    _qx=0.0;
    _qy.create(cv.rows,cv.cols,CV_32FC1);
    _qy=0.0;
}
void Optimizer::initA() {
    cv.loInd.copyTo(_a);
}
bool Optimizer::optimizeA(){
    using namespace cv::gpu::device::dtam_optimizer;
    bool doneOptimizing = theta <= thetaMin;
    int layerStep = cv.rows * cv.cols;
    float* d = (float*) _d.data;
    float* a = (float*) _a.data;

    loadConstants(cv.rows, cv.cols, cv.layers, layerStep, a, d, cv.data, (float*)cv.lo.data,
            (float*)cv.hi.data, (float*)cv.loInd.data);
    minimizeACaller  ( 0);
    return doneOptimizing;
}

