// Free for non-commercial, non-military, and non-critical 
// use unless incorporated in OpenCV. 
// Inherits OpenCV Licence if in OpenCV.


class Optimizer
{
};

// Free for non-commercial, non-military, and non-critical 
// use unless incorporated in OpenCV. 
// Inherits OpenCV Licence if in OpenCV.


#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "CostVolume/CostVolume.h"

class Optimizer
{
public:
    CostVolume cv;//The cost volume we are attached to

    cv::gpu::GpuMat _qx,_qy,_d,_a,_g,_gu,_gd,_gl,_gr,_gbig;
    cv::gpu::GpuMat stableDepth;
    float theta,thetaStart,thetaStep,thetaMin,epsilon,lambda,sigma_d,sigma_q;
    
    void attach(CostVolume cv);
    void init();
    bool optimizeA();
    bool optimizeQD();
    const cv::Mat depthMap();
    
    //public parameters
    float thetaStart,thetaStep,thetaMin,epsilon,lambda;

private:
    void initA();
    void initQD();
    void computeSigmas();
    void cacheGValues();

    //internal param values
    float theta,sigma_d,sigma_q;

    //flags
    bool cachedG;



};

#endif // OPTIMIZER_H
