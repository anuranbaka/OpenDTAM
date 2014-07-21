#include <assert.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#define cudaSafeCall(expr)  __cudaSafeCall(expr, __FILE__, __LINE__, __func__)

void __cudaSafeCall(cudaError_t err, const char* file, const int line, const char* func = "")
{
    if (cudaSuccess != err)
        std::cout<<cudaGetErrorString(err)<<"\nFile:"<<file<<":"<< line<< " IN "<<func<<std::endl;
}
namespace cv { namespace gpu { namespace device {
    namespace dtam_updateCost{

    struct m33{
        float data [9];
    };
//__constant__ float sliceToIm[3 * 3];
__constant__ uint  rows;
__constant__ uint  cols;
__constant__ uint  layers;
__constant__ uint layerStep;
__constant__ float* hdata;
__constant__ float* cdata;
__constant__ float* lo;
__constant__ float* hi;
__constant__ uint* loInd;
__constant__ float3* base;
__constant__ cudaTextureObject_t tex;

__global__ void updateCostCol(m33 sliceToIm, unsigned int yoff);

#define SEND(type,sym) cudaMemcpyToSymbol(sym, &h_ ## sym, sizeof(type));

void loadConstants(int h_layers, int h_layerStep, float3* h_base,
        float* h_hdata, float* h_cdata, float* h_lo, float* h_hi, uint* h_loInd,
        uint h_rows, uint h_cols, cudaTextureObject_t h_tex) {
//    cudaMemcpyToSymbol(c_sliceToIm, h_CVToIm, 3*4*sizeof(float));
    cudaMemcpyToSymbol(layers, &h_layers, sizeof(int));
    cudaMemcpyToSymbol(layerStep, &h_layerStep, sizeof(int));
    cudaMemcpyToSymbol(hdata, &h_hdata, sizeof(float*));
    cudaMemcpyToSymbol(cdata, &h_cdata, sizeof(float*));
    assert(h_lo);
    cudaSafeCall(cudaMemcpyToSymbol(lo, &h_lo, sizeof(float*)));
    cudaMemcpyToSymbol(hi, &h_hi, sizeof(float*));
    cudaMemcpyToSymbol(loInd, &h_loInd, sizeof(uint*));
    SEND(uint,cols);
    SEND(uint,rows);
    SEND(float3*,base);
    SEND(cudaTextureObject_t,tex);
}

//void loadS2I(float* h_sliceToIm){
//    cudaMemcpyToSymbol(sliceToIm, h_sliceToIm, 3*3*sizeof(float));
//}


void updateCostColCaller(int cols,int rows, int y, m33 sliceToIm){
    static cudaStream_t strs [15]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    static int num=0;
    int here=(num++%14)+1;
    if(!strs[here])
        cudaStreamCreate(&(strs[here]));
   dim3 dimBlock(64,1);
   dim3 dimGrid((cols  + dimBlock.x - 1) / dimBlock.x,
                (rows + dimBlock.y - 1) / dimBlock.y);
   updateCostCol<<<dimGrid, dimBlock,1,strs[here]>>>(sliceToIm,y*cols);
}

__global__ void updateCostCol(m33 c_sliceToIm,unsigned int yoff)
{
    //per thread:
        //find projection from column to image (3x2)
        //for each pixel:
            //finds L1 error
            //blend in with old value
            //if low
                //update low index
                //update high value
            //if high
                //update high value
        //save results
    //    x  y  z  1 
    // x  0     1  2
    // y  3     4  5
    // z  6     7  8

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int offset=x+yoff;
    
    
    // Find the current base color
    float3 B = base[offset];

    float3 Z;
    float3 W;
    Z.x = c_sliceToIm.data[1];
    Z.y = c_sliceToIm.data[4];
    Z.z = c_sliceToIm.data[7];

    //add in the offset for the current column
    W.x = c_sliceToIm.data[2] + c_sliceToIm.data[0] * x;
    W.y = c_sliceToIm.data[5] + c_sliceToIm.data[3] * x;
    W.z = c_sliceToIm.data[8] + c_sliceToIm.data[6] * x;
    float *hp = hdata+offset;
    float *cp = cdata+offset;
    float minv = 1000;
    float maxv = 0;
    uint mini=0;
    uint end=layers*layerStep;
    for (uint z = 0; z < end; z+=layerStep, W.x += Z.x, W.y += Z.y, W.z += Z.z) {
        //Texture Sample
            float xi, yi, wi;
            wi = W.z;
            xi = W.x / wi;
            yi = W.y / wi;
            float4 c = tex2D<float4>(tex, xi, yi);

        {//Shader core
            float v1 = fabsf(c.x - B.x);
            float v2 = fabsf(c.y - B.y);
            float v3 = fabsf(c.z - B.z);
            float h = hp[z] + 1;
            float ns = cp[z] * (1 - 1 / h) + (v1 + v2 + v3) / h;
            if (ns < minv) {
                minv = ns;
                mini = z;
            }
            if (ns > maxv) {
                maxv = ns;
            }

            hp[z] = h;
            cp[z] = ns;
        }
    }
    lo[offset]=minv;
    hi[offset]=maxv;
    loInd[offset]=mini;
}

}}}}




