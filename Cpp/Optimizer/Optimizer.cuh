
#ifndef COSTVOLUME_CUH
#define COSTVOLUME_CUH
#include <opencv2/gpu/device/common.hpp>
namespace cv { namespace gpu { namespace device { namespace dtam_optimizer{
    struct m33{
            float data[9];
        };
        struct m34{
            float data[12];
        };
        void loadConstants(uint h_rows, uint h_cols, uint h_layers, uint h_layerStep,
                float* h_a, float* h_d, float* h_cdata, float* h_lo, float* h_hi,
                float* h_loInd);
    void updateCostColCaller(int y, m33 sliceToIm);
    void minimizeACaller(float theta,float lambda);
    void computeGCaller  (float* pp, float* g1p, float* gxp, float* gyp, float lambda, int cols);
    void updateQDCaller  (float* gqxpt, float* gqypt, float *dpt, float * apt,
                    float *gxpt, float *gypt, float sigma_q, float sigma_d, float epsilon,
                    float theta);
    extern cudaStream_t localStream;
}}}}
#endif
