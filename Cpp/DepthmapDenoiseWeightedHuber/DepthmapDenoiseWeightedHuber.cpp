// Free for non-commercial, non-military, and non-critical
// use unless incorporated in OpenCV.
// Inherits OpenCV Licence if in OpenCV.

//This file does Q and D optimization steps on the GPU
#include "DepthmapDenoiseWeightedHuber.hpp"
#include "DepthmapDenoiseWeightedHuber.cuh"
#include <opencv2/gpu/stream_accessor.hpp>

namespace cv{
    namespace gpu{
using namespace std;
using namespace cv::gpu;

DepthmapDenoiseWeightedHuberImpl::DepthmapDenoiseWeightedHuberImpl(const GpuMat& _visibleLightImage,
                                                        Stream _cvStream) : 
                                                        visibleLightImage(_visibleLightImage), 
                                                        rows(_visibleLightImage.rows),
                                                        cols(_visibleLightImage.cols),
                                                        cvStream(_cvStream)
{
    alloced=0;
    cachedG=0; 
    dInited=0;
}

Ptr<DepthmapDenoiseWeightedHuber>
CV_EXPORTS createDepthmapDenoiseWeightedHuber(InputArray visibleLightImage, Stream cvStream){
    return Ptr<DepthmapDenoiseWeightedHuber>(new DepthmapDenoiseWeightedHuberImpl(visibleLightImage.getGpuMat(),cvStream));
}


#define FLATALLOC(n,cv) n.create(1,cv.rows*cv.cols, CV_32FC1);n=n.reshape(0,cv.rows)
static void memZero(GpuMat& in,Stream& cvStream){
    cudaSafeCall(cudaMemsetAsync(in.data,0,in.rows*in.cols*sizeof(float),cv::gpu::StreamAccessor::getStream(cvStream)));
}

void DepthmapDenoiseWeightedHuberImpl::allocate(int _rows,int _cols,InputArray _gxin,InputArray _gyin){
    const GpuMat& gxin=_gxin.getGpuMat();
    const GpuMat& gyin=_gyin.getGpuMat();
    
    rows=_rows;
    cols=_cols;
    if(!(rows % 32 == 0 && cols % 32 == 0 && cols >= 64)){
        CV_Assert(!"For performance reasons, DepthmapDenoiseWeightedHuber currenty only supports multiple of 32 image sizes with cols >= 64. Pad the image to achieve this.");
    }
    

    if(!_a.data){
        _a.create(1,rows*cols, CV_32FC1);
        _a=_a.reshape(0,rows);
    }
    FLATALLOC(_d, _a);
    cachedG=1;
    if(gxin.empty()||gyin.empty()){
        if(gxin.empty()){
            FLATALLOC(_gx,_d);
            cachedG=0;
        }else{
            _gx=gxin;
        }
        if(gyin.empty()){
            FLATALLOC(_gy,_d);
            cachedG=0;
        }else{
            _gy=gyin;
        }
    }else{
        
        if(!gxin.isContinuous()){
            FLATALLOC(_gx,_d);
//             gxin.copyTo(_gx,cvStream);
            cvStream.enqueueCopy(gxin,_gx);
        }
        if(!gyin.isContinuous()){
            FLATALLOC(_gy,_d);
//             gyin.copyTo(_gy,cvStream);
            cvStream.enqueueCopy(gyin,_gy);
        }
    }
    FLATALLOC(_qx, _d);
    FLATALLOC(_qy, _d);
    FLATALLOC(_g1, _d);
    FLATALLOC(stableDepth,_d);
    memZero(_qx,cvStream);
    memZero(_qy,cvStream);
    alloced=1;
}


void DepthmapDenoiseWeightedHuberImpl::computeSigmas(float epsilon,float theta){
    /*
    //This function is my best guess of what was meant by the line:
    //"Gradient ascent/descent time-steps sigma_q , sigma_d are set optimally
    //for the update scheme provided as detailed in [3]."
    // Where [3] is :
    //A. Chambolle and T. Pock. A first-order primal-dual 
    //algorithm for convex problems with applications to imaging.
    //Journal of Mathematical Imaging and Vision, 40(1):120–
    //145, 2011. 3, 4, 6
    //
    // I converted these mechanically to the best of my ability, but no 
    // explaination is given in [3] as to how they came up with these, just 
    // some proofs beyond my ability.
    //
    // Explainations below are speculation, but I think good ones:
    //
    // L appears to be a bound on the largest vector length that can be 
    // produced by the linear operator from a unit vector. In this case the 
    // linear operator is the differentiation matrix with G weighting 
    // (written AG in the DTAM paper,(but I use GA because we want to weight 
    // the springs rather than the pixels)). Since G has each row sum < 1 and 
    // A is a forward difference matrix (which has each input affecting at most
    // 2 outputs via pairs of +-1 weights) the value is bounded by 4.0.
    //
    // So in a sense, L is an upper bound on the magnification of our step size.
    // 
    // Lambda and alpha are called convexity parameters. They come from the 
    // Huber norm and the (d-a)^2 terms. The convexity parameter of a function 
    // is defined as follows: 
    //  Choose a point on the function and construct a parabola of convexity 
    //    c tangent at that point. Call the point c-convex if the parabola is 
    //    above the function at all other points. 
    //  The smallest c such that the function is c-convex everywhere is the 
    //      convexity parameter.
    //  We can think of this as a measure of the bluntest tip that can trace the 
    //     entire function.
    // This is important because any gradient descent step that would not 
    // cause divergence on the tangent parabola is guaranteed not to diverge 
    // on the base function (since the parabola is always higher(i.e. worse)).
    */
    
        
    float lambda, alpha,gamma,delta,mu,rho,sigma;
    float L=4;//lower is better(longer steps), but in theory only >=4 is guaranteed to converge. For the adventurous, set to 2 or 1.44
    
    lambda=1.0/theta;
    alpha=epsilon;
    
    gamma=lambda;
    delta=alpha;
    
    mu=2.0*std::sqrt(gamma*delta)/L;

    rho= mu/(2.0*gamma);
    sigma=mu/(2.0*delta);
    
    sigma_d = rho;
    sigma_q = sigma;
}

void DepthmapDenoiseWeightedHuberImpl::cacheGValues(InputArray _visibleLightImage){
    using namespace cv::gpu::device::dtam_denoise;
    localStream = cv::gpu::StreamAccessor::getStream(cvStream);
    if (!_visibleLightImage.empty()){
        visibleLightImage=_visibleLightImage.getGpuMat();
        cachedG=0;
    }
    if(cachedG)
        return;//already cached
    if(!alloced)
        allocate(rows,cols);

    
    // Call the gpu function for caching g's
    
    loadConstants(rows, cols, 0, 0, 0, 0, 0, 0,
            0, 0);
    CV_Assert(_g1.isContinuous());
    float* pp = (float*) visibleLightImage.data;//TODO: write a color version.
    float* g1p = (float*)_g1.data;
    float* gxp = (float*)_gx.data;
    float* gyp = (float*)_gy.data;
    computeGCaller(pp,  g1p,  gxp,  gyp, cols);
    cachedG=1;
}

GpuMat DepthmapDenoiseWeightedHuberImpl::operator()(InputArray _ain, float epsilon,float theta){
    const GpuMat& ain=_ain.getGpuMat();
    
    using namespace cv::gpu::device::dtam_denoise;
    localStream = cv::gpu::StreamAccessor::getStream(cvStream);
    
    rows=ain.rows;
    cols=ain.cols;
    
    CV_Assert(ain.cols>0);
    if(!(ain.rows % 32 == 0 && ain.cols % 32 == 0 && ain.cols >= 64)){
        CV_Assert(!"For performance reasons, DepthmapDenoiseWeightedHuber currenty only supports multiple of 32 image sizes with cols >= 64. Pad the image to achieve this.");
    }
    rows=ain.rows;
    cols=ain.cols;
    if(!ain.isContinuous()){
        _a.create(1,rows*cols, CV_32FC1);
        _a=_a.reshape(0,rows);
//         ain.copyTo(_a,cvStream);
        cvStream.enqueueCopy(ain,_a);
    }else{
        _a=ain;
    }
    

    
    if(!alloced){
        allocate(rows,cols);
    } 
    
    if(!visibleLightImage.empty())
        cacheGValues();
    if(!cachedG){
//         _gx.setTo(1,cvStream);
        _gx=1;
//         _gy.setTo(1,cvStream);
        _gy=1;
    }
    if(!dInited){
//         _a.copyTo(_d,cvStream);
        cvStream.enqueueCopy(_a,_d);
        dInited=1;
    }
    
    computeSigmas(epsilon,theta);
    
    float* d = (float*) _d.data;
    float* a = (float*) _a.data;
    float* gxpt = (float*)_gx.data;
    float* gypt = (float*)_gy.data;
    float* gqxpt = (float*)_qx.data;
    float* gqypt = (float*)_qy.data;

    loadConstants(rows, cols, 0, 0, 0, 0, 0, 0,
           0, 0);
    updateQDCaller  ( gqxpt, gqypt, d, a,
            gxpt, gypt, cols, sigma_q, sigma_d, epsilon, theta);
    cudaSafeCall(cudaGetLastError());
    return _d;
}
}  
}


