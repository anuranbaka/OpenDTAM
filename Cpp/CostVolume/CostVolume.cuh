
#ifndef COSTVOLUME_CUH
#define COSTVOLUME_CUH
#include <opencv2/gpu/device/common.hpp>
namespace cv { namespace gpu { namespace device { namespace dtam_updateCost{
    struct m33{
            float data[9];
        };
        struct m34{
            float data[12];
        };
    extern cudaStream_t localStream;
    void loadConstants(int h_layers, int h_layerStep, float3* h_base,
            float* h_hdata, float* h_cdata, float* h_lo, float* h_hi, float* h_loInd,
            uint h_rows, uint h_cols, cudaTextureObject_t h_tex);
    void updateCostColCaller(int cols,int rows, int y, m33 sliceToIm);
    void passThroughCaller(int cols,int rows);
    void perspCaller(int cols,int rows,m34 persp);
    void volumeProjectCaller(int cols,int rows,m34 p);
    void simpleCostCaller(int cols,int rows,m34 p);
    void globalWeightedCostCaller(int cols,int rows,m34 p,float weight);
    void globalWeightedBoundsCostCaller(int cols,int rows,m34 p,float weight);
    
}}}}
#endif
