
#ifndef COSTVOLUME_CUH
#define COSTVOLUME_CUH
#include <opencv2/core/cuda/common.hpp>
namespace cv { namespace cuda { namespace dtam_optimizer{
    struct m33{
            float data[9];
        };
        struct m34{
            float data[12];
        };
        void loadConstants(uint h_rows, uint h_cols, uint h_layers, uint h_layerStep,
                float* h_a, float* h_d, float* h_cdata, float* h_lo, float* h_hi,
                float* h_loInd);
    void minimizeACaller(float*cdata,float*a, float* d, int layers,float theta,float lambda);
    
    extern cudaStream_t localStream;
}}}
#endif
