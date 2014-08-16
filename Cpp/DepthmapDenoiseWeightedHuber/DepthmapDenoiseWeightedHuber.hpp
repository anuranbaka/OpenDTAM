// Free for non-commercial, non-military, and non-critical 
// use unless incorporated in OpenCV. 
// Inherits OpenCV Licence if in OpenCV.


#ifndef DepthmapDenoiseWeightedHuber_H
#define DepthmapDenoiseWeightedHuber_H
#include <opencv2/core/cuda.hpp>


class DepthmapDenoiseWeightedHuber
{
public:
    //CostVolume cv;//The cost volume we are attached to
    
    DepthmapDenoiseWeightedHuber(int rows,int cols,const cv::cuda::GpuMat& visibleLightImage=cv::cuda::GpuMat(),cv::cuda::Stream cvStream=cv::cuda::Stream::Null());
    cv::cuda::GpuMat operator()(const cv::cuda::GpuMat& ain,
                                float epsilon,
                                float theta);




    


    //public parameters
    float epsilon,theta;

    
    cv::cuda::GpuMat visibleLightImage;
    //buffers
    cv::cuda::GpuMat _qx,_qy,_d,_a,_g,_g1,_gx,_gy;
    cv::cuda::GpuMat stableDepth;
    float getTheta(){return theta;}
    
    //in case you want to do these explicitly
    void allocate(int rows,int cols,const cv::cuda::GpuMat& gxin=cv::cuda::GpuMat(),const cv::cuda::GpuMat& gyin=cv::cuda::GpuMat());
    void cacheGValues();
    
private:
    int rows;
    int cols;
    
    void computeSigmas(float epsilon,float theta);
    

    //internal parameter values
    float sigma_d,sigma_q;

    //flags
    bool cachedG;
    int alloced;
    int dInited;
    

public:
    cv::cuda::Stream cvStream;

    
};

#endif // DepthmapDenoiseWeightedHuber_H
