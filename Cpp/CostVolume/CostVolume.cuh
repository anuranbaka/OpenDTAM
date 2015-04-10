
#ifndef COSTVOLUME_CUH
#define COSTVOLUME_CUH
#include <opencv2/core/cuda/common.hpp>
namespace cv { namespace cuda { namespace dtam_updateCost{
    struct m33{
            float data[9];
        };
        struct m34{
            float data[12];
        };
    extern cudaStream_t localStream;

    void updateCostColCaller( int y, m33 sliceToIm, float weight, uint  rows, uint  cols, uint  layers, uint layerStep, float* hdata, float* cdata, float* lo, float* hi, float* loInd, float3* base,  float* bf, cudaTextureObject_t tex);
    void passThroughCaller( float weight, uint  rows, uint  cols, uint  layers, uint layerStep, float* hdata, float* cdata, float* lo, float* hi, float* loInd, float3* base,  float* bf, cudaTextureObject_t tex);
    void perspCaller(m34 persp, float weight, uint  rows, uint  cols, uint  layers, uint layerStep, float* hdata, float* cdata, float* lo, float* hi, float* loInd, float3* base,  float* bf, cudaTextureObject_t tex);
    void volumeProjectCaller(m34 p, float weight, uint  rows, uint  cols, uint  layers, uint layerStep, float* hdata, float* cdata, float* lo, float* hi, float* loInd, float3* base,  float* bf, cudaTextureObject_t tex);
    void simpleCostCaller(m34 p, float weight, uint  rows, uint  cols, uint  layers, uint layerStep, float* hdata, float* cdata, float* lo, float* hi, float* loInd, float3* base,  float* bf, cudaTextureObject_t tex);
    void globalWeightedCostCaller(m34 p, float weight, uint  rows, uint  cols, uint  layers, uint layerStep, float* hdata, float* cdata, float* lo, float* hi, float* loInd, float3* base,  float* bf, cudaTextureObject_t tex);
    void globalWeightedBoundsCostCaller(m34 p,float weight,uint  rows, uint  cols, uint  layers, uint layerStep, float* hdata, float* cdata, float* lo, float* hi, float* loInd, float3* base,  float* bf, cudaTextureObject_t tex);
    
}}}
#endif
