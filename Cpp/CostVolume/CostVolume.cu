#include <assert.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "CostVolume.cuh"



namespace cv { namespace gpu { namespace device {
    namespace dtam_updateCost{

cudaStream_t localStream;

#define CONSTT uint  rows, uint  cols, uint  layers, uint layerStep, float* hdata, float* cdata, float* lo, float* hi, float* loInd, float3* base,  float* bf, cudaTextureObject_t tex
#define CONSTS rows,  cols,  layers, layerStep,  hdata,  cdata,  lo, hi,  loInd,  base,   bf,  tex

#define BLOCK_X 64
#define BLOCK_Y 4
__global__ void globalWeightedBoundsCost(m34 p,float weight, CONSTT);
void globalWeightedBoundsCostCaller(m34 p,float weight,CONSTT){
   dim3 dimBlock(BLOCK_X,BLOCK_Y);
   dim3 dimGrid((cols  + dimBlock.x - 1) / dimBlock.x,
                (rows + dimBlock.y - 1) / dimBlock.y);
   globalWeightedBoundsCost<<<dimGrid, dimBlock, 0, localStream>>>(p, weight,CONSTS);
   assert(localStream);
   cudaSafeCall( cudaGetLastError() );
}


__global__ void globalWeightedBoundsCost(m34 p,float weight, CONSTT)
{
    //float*bf=(float*)base;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//    //unsigned int topleft=3*blockDim.x * blockIdx.x + 3*cols*blockDim.y*blockIdx.y;
//    //unsigned int loff=threadIdx.x+3*cols*threadIdx.y;
//    unsigned int sooff=threadIdx.x+threadIdx.y*blockDim.x;
//    //unsigned int sioff=threadIdx.x+3*threadIdx.y*blockDim.x;
//    unsigned int sioff=3*sooff-2*threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    //unsigned int gindex=topleft+loff;

    float xf=x;
    float yf=y;
    unsigned int offset=x+y*cols;
//    unsigned int gindex=offset*3-2*threadIdx.x;
//    const unsigned int mul=BLOCK_X;
//    __shared__ float buff[BLOCK_X*BLOCK_Y*3];
//
//
//    buff[sioff] = bf[gindex];
//    buff[sioff+mul] = bf[gindex+mul];
//    buff[sioff+mul*2] = bf[gindex+mul*2];
//    //__syncthreads();
//    float3 B =((float3*)buff)[sooff];


    float3 B = base[x+y*cols];//Known bug:this requires 12 loads instead of 4 because of stupid memory addressing, can't really fix
    float wi = p.data[8]*xf + p.data[9]*yf + p.data[11];
    float xi = (p.data[0]*xf + p.data[1]*yf + p.data[3]);
    float yi = (p.data[4]*xf + p.data[5]*yf + p.data[7]);
    float minv=1000.0,maxv=0.0;
    float mini=0;
    for(unsigned int z=0;z<layers;z++){
        float c0=cdata[offset+z*layerStep];
        float wiz = wi+p.data[10]*z;
        float xiz = xi+p.data[2] *z;
        float yiz = yi+p.data[6] *z;
        float4 c = tex2D<float4>(tex, xiz/wiz, yiz/wiz);
        float v1 = fabsf(c.x - B.x);
        float v2 = fabsf(c.y - B.y);
        float v3 = fabsf(c.z - B.z);
        float del=v1+v2+v3;
        float ns;
//         if(del>.03){
//             del=0;
//         }
//         del=sqrt(del);
        del=.0001*del + fminf(del,.01f)*1.0f/.01f;
        ns=c0*weight+(del)*(1-weight);
//         ns=del;
        cdata[offset+z*layerStep]=ns;
        if (ns < minv) {
        minv = ns;
        mini = z;
        }
        maxv=fmaxf(ns,maxv);
    }
    lo[offset]=minv;
    loInd[offset]=mini;
    hi[offset]=maxv;
}


// 
// //__constant__ float sliceToIm[3 * 3];
// __constant__ uint  rows;
// __constant__ uint  cols;
// __constant__ uint  layers;
// __constant__ uint layerStep;
// __constant__ float* hdata;
// __constant__ float* cdata;
// __constant__ float* lo;
// __constant__ float* hi;
// __constant__ float* loInd;
// __constant__ /*const __restrict__*/ float3* base;
// __constant__ /*const __restrict__*/ float* bf;
// __constant__ cudaTextureObject_t tex;
// 
// __global__ void updateCostCol(m33 sliceToIm, unsigned int yoff);
// 
// #define SEND(type,sym) cudaMemcpyToSymbolAsync(sym, &h_ ## sym, sizeof(type), 0, cudaMemcpyHostToDevice, localStream);
// 
// void loadConstants(int h_layers, int h_layerStep, float3* h_base,
//         float* h_hdata, float* h_cdata, float* h_lo, float* h_hi, float* h_loInd,
//         uint h_rows, uint h_cols, cudaTextureObject_t h_tex) {
// //    cudaMemcpyToSymbol(c_sliceToIm, h_CVToIm, 3*4*sizeof(float));
//     cudaMemcpyToSymbol(layers, &h_layers, sizeof(int));
//     cudaMemcpyToSymbol(layerStep, &h_layerStep, sizeof(int));
//     cudaMemcpyToSymbol(hdata, &h_hdata, sizeof(float*));
//     cudaMemcpyToSymbol(cdata, &h_cdata, sizeof(float*));
//     assert(h_lo);
//     cudaSafeCall(cudaMemcpyToSymbol(lo, &h_lo, sizeof(float*)));
//     cudaMemcpyToSymbol(hi, &h_hi, sizeof(float*));
//     cudaMemcpyToSymbol(loInd, &h_loInd, sizeof(float*));
//     SEND(uint,cols);
//     SEND(uint,rows);
//     SEND(float3*,base);
//     float* h_bf=(float*)h_base;
//     SEND(float*,bf);
//     SEND(cudaTextureObject_t,tex);
// }
// 
// //void loadS2I(float* h_sliceToIm){
// //    cudaMemcpyToSymbol(sliceToIm, h_sliceToIm, 3*3*sizeof(float));
// //}
// 
// 
// void updateCostColCaller(int cols,int rows, int y, m33 sliceToIm){
//     static cudaStream_t strs [32]={0,0,0,0, 0,0,0,0,  0,0,0,0, 0,0,0,0,  0,0,0,0, 0,0,0,0,  0,0,0,0, 0,0,0,0,};
//     static int num=0;
//     int here=(num++%31)+1;
//     if(!strs[here])
//         cudaStreamCreate(&(strs[here]));
//    dim3 dimBlock(64,1);
//    dim3 dimGrid((cols  + dimBlock.x - 1) / dimBlock.x,
//                 (rows + dimBlock.y - 1) / dimBlock.y);
//    updateCostCol<<<dimGrid, dimBlock,1,strs[here]>>>(sliceToIm,y*cols);
// }
// 
// __global__ void updateCostCol(m33 c_sliceToIm,unsigned int yoff)
// {
//     //per thread:
//         //find projection from column to image (3x2)
//         //for each pixel:
//             //finds L1 error
//             //blend in with old value
//             //if low
//                 //update low index
//                 //update high value
//             //if high
//                 //update high value
//         //save results
//     //    x  y  z  1 
//     // x  0     1  2
//     // y  3     4  5
//     // z  6     7  8
// 
//     unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
// //    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//     unsigned int offset=x+yoff;
//     
//     
//     // Find the current base color
//     float3 B = base[offset];
// 
//     float3 Z;
//     float3 W;
//     Z.x = c_sliceToIm.data[1];
//     Z.y = c_sliceToIm.data[4];
//     Z.z = c_sliceToIm.data[7];
// 
//     //add in the offset for the current column
//     W.x = c_sliceToIm.data[2] + c_sliceToIm.data[0] * x;
//     W.y = c_sliceToIm.data[5] + c_sliceToIm.data[3] * x;
//     W.z = c_sliceToIm.data[8] + c_sliceToIm.data[6] * x;
//     float *hp = hdata+offset;
//     float *cp = cdata+offset;
//     float minv = 1000;
//     float maxv = 0;
//     uint mini=0;
//     uint end=layers*layerStep;
//     for (uint z = 0; z < end; z+=layerStep, W.x += Z.x, W.y += Z.y, W.z += Z.z) {
//         //Texture Sample
//             float xi, yi, wi;
//             wi = W.z;
//             xi = W.x / wi;
//             yi = W.y / wi;
//             float4 c = tex2D<float4>(tex, xi, yi);
// 
//         {//Shader core
//             float v1 = fabsf(c.x - B.x);
//             float v2 = fabsf(c.y - B.y);
//             float v3 = fabsf(c.z - B.z);
//             float h = hp[z] + 1;
//             float ns = cp[z] * (1 - 1 / h) + (v1 + v2 + v3) / h;
//             if (ns < minv) {
//                 minv = ns;
//                 mini = z;
//             }
//             if (ns > maxv) {
//                 maxv = ns;
//             }
// 
//             hp[z] = h;
//             cp[z] = ns;
//         }
//     }
//     lo[offset]=minv;
//     hi[offset]=maxv;
//     loInd[offset]=mini;
// }
// __global__ void passThrough();
// void passThroughCaller(int cols,int rows){
//    dim3 dimBlock(64,4);
//    dim3 dimGrid((cols  + dimBlock.x - 1) / dimBlock.x,
//                 (rows + dimBlock.y - 1) / dimBlock.y);
//    passThrough<<<dimGrid, dimBlock>>>();
// }
// 
// __global__ void passThrough()
// {
// 
// 
//     unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
// 
// 
// 
// 
//     float4 c = tex2D<float4>(tex, x, y);
//     cdata[x+y*cols]=c.x;
// }
// 
// __global__ void persp(m34 p);
// void perspCaller(int cols,int rows,m34 p){
//    dim3 dimBlock(64,4);
//    dim3 dimGrid((cols  + dimBlock.x - 1) / dimBlock.x,
//                 (rows + dimBlock.y - 1) / dimBlock.y);
//    persp<<<dimGrid, dimBlock>>>(p);
// }
// __global__ void persp(m34 p)
// {
//     unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//     float xf=x;
//     float yf=y;
// 
//     float wi = p.data[8]*xf + p.data[9]*yf + p.data[10]*0 + p.data[11];
//     float xi = (p.data[0]*xf + p.data[1]*yf + p.data[2] *0 + p.data[3])/wi;
//     float yi = (p.data[4]*xf + p.data[5]*yf + p.data[6] *0 + p.data[7])/wi;
// 
// 
// 
//     float4 c = tex2D<float4>(tex, xi, yi);
//     cdata[x+y*cols]=c.x;
// }
// 
// __global__ void volumeProject(m34 p);
// void volumeProjectCaller(int cols,int rows,m34 p){
//    dim3 dimBlock(64,4);
//    dim3 dimGrid((cols  + dimBlock.x - 1) / dimBlock.x,
//                 (rows + dimBlock.y - 1) / dimBlock.y);
//    volumeProject<<<dimGrid, dimBlock>>>(p);
// }
// 
// 
// __global__ void volumeProject(m34 p)
// {
//     unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//     float xf=x;
//     float yf=y;
// 
//     float wi = p.data[8]*xf + p.data[9]*yf + p.data[10]*0 + p.data[11];
//     float xi = (p.data[0]*xf + p.data[1]*yf + p.data[2] *0 + p.data[3]);
//     float yi = (p.data[4]*xf + p.data[5]*yf + p.data[6] *0 + p.data[7]);
// 
//     for(unsigned int z=0;z<layers;z++){
//         float wiz = wi+p.data[10]*z;
//         float xiz = xi+p.data[2] *z;
//         float yiz = yi+p.data[6] *z;
//         float4 c = tex2D<float4>(tex, xiz/wiz, yiz/wiz);
//         cdata[x+y*cols+z*layerStep]=c.x;
//     }
// }
// 
// __global__ void simpleCost(m34 p);
// void simpleCostCaller(int cols,int rows,m34 p){
//    dim3 dimBlock(64,4);
//    dim3 dimGrid((cols  + dimBlock.x - 1) / dimBlock.x,
//                 (rows + dimBlock.y - 1) / dimBlock.y);
//    simpleCost<<<dimGrid, dimBlock>>>(p);
// }
// 
// 
// __global__ void simpleCost(m34 p)
// {
//     unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//     float xf=x;
//     float yf=y;
//     unsigned int offset=x+y*cols;
//     float3 B = base[x+y*cols];
//     float wi = p.data[8]*xf + p.data[9]*yf + p.data[11];
//     float xi = (p.data[0]*xf + p.data[1]*yf + p.data[3]);
//     float yi = (p.data[4]*xf + p.data[5]*yf + p.data[7]);
// 
//     for(unsigned int z=0;z<layers;z++){
//         float c0=cdata[offset+z*layerStep];
//         float wiz = wi+p.data[10]*z;
//         float xiz = xi+p.data[2] *z;
//         float yiz = yi+p.data[6] *z;
//         float4 c = tex2D<float4>(tex, xiz/wiz, yiz/wiz);
//         float v1 = fabsf(c.x - B.x);
//         float v2 = fabsf(c.y - B.y);
//         float v3 = fabsf(c.z - B.z);
//         cdata[offset+z*layerStep]=c0+v1+v2+v3;
//     }
// }
// __global__ void globalWeightedCost(m34 p,float weight);
// void globalWeightedCostCaller(int cols,int rows,m34 p,float weight){
//    dim3 dimBlock(64,4);
//    dim3 dimGrid((cols  + dimBlock.x - 1) / dimBlock.x,
//                 (rows + dimBlock.y - 1) / dimBlock.y);
//    globalWeightedCost<<<dimGrid, dimBlock>>>(p, weight);
// }
// 
// 
// __global__ void globalWeightedCost(m34 p,float weight)
// {
//     unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//     float xf=x;
//     float yf=y;
//     unsigned int offset=x+y*cols;
//     float3 B = base[x+y*cols];
//     float wi = p.data[8]*xf + p.data[9]*yf + p.data[11];
//     float xi = (p.data[0]*xf + p.data[1]*yf + p.data[3]);
//     float yi = (p.data[4]*xf + p.data[5]*yf + p.data[7]);
// 
//     for(unsigned int z=0;z<layers;z++){
//         float c0=cdata[offset+z*layerStep];
//         float wiz = wi+p.data[10]*z;
//         float xiz = xi+p.data[2] *z;
//         float yiz = yi+p.data[6] *z;
//         float4 c = tex2D<float4>(tex, xiz/wiz, yiz/wiz);
//         float v1 = fabsf(c.x - B.x);
//         float v2 = fabsf(c.y - B.y);
//         float v3 = fabsf(c.z - B.z);
//         cdata[offset+z*layerStep]=c0*weight+(v1+v2+v3)*(1-weight);
//     }
// }



}}}}




