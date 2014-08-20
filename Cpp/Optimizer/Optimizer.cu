#include <assert.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "Optimizer.cuh"


#ifndef __CUDACC__
#define __constant__
#define __global__
#define __host__
#define __device__
#endif






namespace cv { namespace gpu { namespace device {
    namespace dtam_optimizer{

static unsigned int arows,acols;

#define SEND(type,sym) cudaMemcpyToSymbol(sym, &h_ ## sym, sizeof(type));

void loadConstants(uint h_rows, uint h_cols, uint, uint,
        float*, float* , float* , float* , float* ,
        float*) {

        //special
        arows=h_rows;
        acols=h_cols;
        assert(arows>0);
}

cudaStream_t localStream=0;

const int BLOCKX2D=32;
const int BLOCKY2D=32;
#define GENERATE_CUDA_FUNC2D(funcName,arglist,notypes)                                     \
static __global__ void funcName arglist;                                                        \
void funcName##Caller arglist{                                                           \
   dim3 dimBlock(BLOCKX2D,BLOCKY2D);                                                                  \
   dim3 dimGrid((acols  + dimBlock.x - 1) / dimBlock.x,                                  \
                (arows + dimBlock.y - 1) / dimBlock.y);                                  \
   funcName<<<dimGrid, dimBlock,0,localStream>>>notypes;                                  \
   cudaSafeCall( cudaGetLastError() );\
};static __global__ void funcName arglist


#define GENERATE_CUDA_FUNC2DROWS(funcName,arglist,notypes)                                     \
static __global__ void funcName arglist;                                                        \
void funcName##Caller arglist{                                                           \
   dim3 dimBlock(BLOCKX2D,BLOCKY2D);                                                                  \
   dim3 dimGrid(1,                                  \
                (arows + dimBlock.y - 1) / dimBlock.y);                                  \
   funcName<<<dimGrid, dimBlock,0,localStream>>>notypes;                                  \
   cudaSafeCall( cudaGetLastError() );\
};static __global__ void funcName arglist

const int BLOCKX1D=256;

#define GENERATE_CUDA_FUNC1D(funcName,arglist,notypes)                                     \
static __global__ void funcName arglist;                                                        \
void funcName##Caller arglist{                                                           \
   dim3 dimBlock(BLOCKX1D);                                                              \
   dim3 dimGrid((acols*arows) / dimBlock.x);                                             \
   funcName<<<dimGrid, dimBlock,0,localStream>>>notypes;                                  \
   cudaSafeCall( cudaGetLastError() );\
};static __global__ void funcName arglist


__device__
static inline float afunc(float costval,float theta,float d,float ds,float a,float lambda){
//    return costval;
    return 1.0f/(2.0f*theta)*ds*ds*(d-a)*(d-a) + costval*lambda;//Literal implementation of Eq.14, note the datastep^2 factor to scale correctly
//     return 1.0/(2.0*theta)*(d-a)*(d-a) + data[a]*lambda;//forget the ds^2 factor for better numerical behavior(sometimes)
//     return std::abs(1.0/(2.0*theta)*ds*ds*(d-a)) + data[a]*lambda;//L1 Version
}

//template <int layers>
GENERATE_CUDA_FUNC1D(minimizeA,
                        (float*cdata,float*a, float* d, int layers,float theta,float lambda),
                        (cdata,a,d,layers,theta,lambda)) {
   // __shared__ float s0[32*BLOCKX1D];
   // float* s=s0+threadIdx.x*32;
    unsigned int pt = blockIdx.x * blockDim.x + threadIdx.x;
    float dv=d[pt];
    float *out=a+pt;
    float *cp=cdata+pt;
    const int layerStep=blockDim.x*gridDim.x;
    const int l=layerStep;
    const float depthStep=1.0f/layers;
    float vlast,vnext,v,A,B,C;

    unsigned int mini=0;

    float minv= v = afunc(cp[0], theta ,dv,depthStep,0,lambda);

    vnext = afunc(cp[l], theta ,dv,depthStep,1,lambda);
#pragma unroll 4
    for(unsigned int z=2;z<layers;z++){
        vlast=v;
        v=vnext;
        vnext = afunc(cp[z*l], theta ,dv,depthStep,z,lambda);
        if(v<minv){
            A=vlast;
            C=vnext;
            minv=v;
            mini=z-1;
        }
    }

//    a[pt]=mini;
//    return;//the no interpolation soln

    if (vnext<minv){//last was best
        *out=layers-1;
        return;
    }

    if (mini==0){//first was best
        *out=0;
        return;
    }

    B=minv;//avoid divide by zero, since B is already <= others, make < others

    float denom=(A-2*B+C);
    float delt=(A-C)/(denom*2);
    //value=A/2*(delt)*(delt-1)-B*(delt+1)*(delt-1)+C/2*(delt+1)*(delt);
//    minv=B-(A-C)*delt/4;
    if(denom!=0)
        *out=delt+float(mini);
    else
        *out=mini;//float(mini);
}

GENERATE_CUDA_FUNC1D(minimizeAshared,
                        (float*cdata,float*a, float* d,int rows,int cols, int layers,float theta,float lambda),
                        (cdata,a,d,rows,cols,layers,theta,lambda)) {
    __shared__ float s0[32*BLOCKX1D];
    float* s=s0+threadIdx.x*32;
    unsigned int pt = blockIdx.x * blockDim.x + threadIdx.x;
    float dv=d[pt];
    float *out=a+pt;
    float *cp=cdata+pt;
    int layerStep=rows*cols;
    const float depthStep=1.0f/layers;
    const int l=layerStep;


    float minv=__int_as_float(0x7F800000);//inf
    unsigned int mini=0;
#pragma unroll 8
    for(unsigned int z=0;z<layers;z++){
        float c=cp[z*l];

        float val = afunc(c, theta ,dv,depthStep,z,lambda);
        if(val<minv){
            minv=val;
            mini=z;
        }
        s[z]=val;
    }
//    a[pt]=mini;
//    return;//the no interpolation soln
    if (mini==0){
        *out=0;
        return;
    }
    if (mini==layers-1){
        *out=layers-1;
        return;
    }
    float A=s[mini-1];
    float B=minv;//avoid divide by zero, since B is already <= others, make < others
    float C=s[mini+1];
    float denom=(A-2*B+C);
    float delt=(A-C)/denom*.5;
    //value=A/2*(delt)*(delt-1)-B*(delt+1)*(delt-1)+C/2*(delt+1)*(delt);
//    minv=B-(A-C)*delt/4;
    if(denom!=0)
        *out=delt+float(mini);
    else
        *out=100;//float(mini);
}

//cool but cryptic min algo
GENERATE_CUDA_FUNC1D(minimizeAcool,
                        (float*cdata,float*a, float* d,int rows,int cols, int layers,float theta,float lambda),
                        (cdata,a,d,rows,cols,layers,theta,lambda)) {
    //__shared__ float s[32];
    unsigned int pt = blockIdx.x * blockDim.x + threadIdx.x;
    float dv=d[pt];
    float *cp=cdata+pt;
    int layerStep=rows*cols;
    const float depthStep=1.0f/layers;
    const int l=layerStep;


    float minv=__int_as_float(0x7F800000);//inf
    float mini=0;
#pragma unroll 32
    for(int z=0;z<layers;z++){
        float c=cp[z*l];
        //s[z]=cp[z*l];

        float val = afunc(c, theta ,dv,depthStep,z,lambda);

        val=__int_as_float(__float_as_int(val)&0xFFFFFF00|z);//load the depth into the least significant 8 bits of the float
        if(val<minv){
            minv=val;
            mini=z;
            mini=__float_as_int(minv)&0x000000FF;
        }
//        minv=fminf(minv,val);
    }
    int loc = __float_as_int(minv)&0x000000FF;

    a[pt]=mini;
}







static __global__ void computeG1  (float* pp, float* g1p, float* gxp, float* gyp, float lambda, int cols);
static __global__ void computeG2  (float* pp, float* g1p, float* gxp, float* gyp, float lambda, int cols);
void computeGCaller  (float* pp, float* g1p, float* gxp, float* gyp, float lambda, int cols){
//   dim3 dimBlock(BLOCKX2D,BLOCKY2D);
   dim3 dimBlock(BLOCKX2D,4);
   dim3 dimGrid(1,
                (arows + dimBlock.y - 1) / dimBlock.y);
   computeG1<<<dimGrid, dimBlock,0,localStream>>>(pp, g1p, gxp, gyp, lambda, cols);
   computeG2<<<dimGrid, dimBlock,0,localStream>>>(pp, g1p, gxp, gyp, lambda, cols);
   cudaSafeCall( cudaGetLastError() );
};

GENERATE_CUDA_FUNC2DROWS(computeG1,
                     (float* pp, float* g1p, float* gxp, float* gyp, float lambda, int cols),
                     (pp, g1p, gxp, gyp, lambda, cols)) {
//TODO: make compatible with cuda 2.0 and lower (remove shuffles). Probably through texture fetch

//Original pseudocode for this function:
    // //subscripts u,d,l,r mean up,down,left,right
    // void computeG(){
    //     // g0 is the strongest nearby gradient (excluding point defects)
    //     g0x=fabsf(pr-pl);//|dx|
    //     g0y=fabsf(pd-pu);//|dy|
    //     g0=max(g0x,g0y);
    //     // g1 is the scaled g0 through the g function exp(-alpha*x^beta)
    //     g1=sqrt(g0); //beta=0.5
    //     alpha=3.5;
    //     g1=exp(-alpha*g1);
    //     //hard to explain this without a picture, but breaks are where both neighboring pixels are near a change
    //     gx=max(g1r,g1);
    //     gy=max(g1d,g1);
    //     gu=gyu;  //upper spring is the lower spring of the pixel above
    //     gd=gy;   //lower spring
    //     gr=gx;   //right spring
    //     gl=gxl;  //left spring is the right spring of the pixel to the left
    // }
    const float alpha=3.5f;
    int x = threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int upoff=-(y!=0)*cols;
    int dnoff=(y<gridDim.y*blockDim.y-1)*cols;
    //itr0
    int pt=x+y*cols;
    float ph,pn,pu,pd,pl,pr;
    float g0x,g0y,g0,g1,gt,gsav;
    float tmp;
    ph=pp[pt];
    pn=pp[pt+blockDim.x];

    pr=__shfl_down(ph,2);
    tmp=__shfl_up(pn,30);
    if(threadIdx.x>=30){
        pr=tmp;
    }
    pl=ph;
    pu=pp[pt+upoff];
    pd=pp[pt+dnoff];


    // g0 is the strongest nearby gradient (excluding point defects)
        gt=fabsf(pr-pl);
        g0x=__shfl_up(gt,1);//?xxxxxx no prior val
        gsav=__shfl_down(gt,31);//x000000 for next time
        g0x=threadIdx.x>0?g0x:0.0f;//0xxxxxx
        g0y=fabsf(pd-pu);

        g0=fmaxf(g0x,g0y);
    // g1 is the scaled g0 through the g function
        g1=sqrt(g0);
        g1=exp(-alpha*g1);
    //save
        g1p[pt]=g1;

    x+=32;
    //itr 1:n-2
    for(;x<cols-32;x+=32){
        pt=x+y*cols;
        ph=pn;
        pn=pp[pt+blockDim.x];
        pr=__shfl_down(ph,2);
        tmp=__shfl_up(pn,30);
        pr=threadIdx.x>=30?tmp:pr;

        pl=ph;
        pu=pp[pt+upoff];
        pd=pp[pt+dnoff];

        // g0 is the strongest nearby gradient (excluding point defects)
            gt=fabsf(pr-pl);
            g0x=__shfl_up(gt,1);//?xxxxxx
            g0x=threadIdx.x>0?g0x:gsav;//xxxxxxx
            gsav=__shfl_down(gt,31);//x000000 for next time
            g0y=fabsf(pd-pu);

            g0=fmaxf(g0x,g0y);

        // g1 is the scaled g0 through the g function
            g1=sqrt(g0);
            g1=exp(-alpha*g1);
        //save
            g1p[pt]=g1;
    }

    //itr n-1
    pt=x+y*cols;
    ph=pn;
    pr=__shfl_down(ph,2);
    pl=ph;
    pu=pp[pt+upoff];
    pd=pp[pt+dnoff];

    // g0 is the strongest nearby gradient (excluding point defects)
        gt=fabsf(pr-pl);
        g0x=__shfl_up(gt,1);//?xxxxxx
        g0x=threadIdx.x>0?g0x:gsav;//xxxxxxx
        g0y=fabsf(pd-pu);

        g0=fmaxf(g0x,g0y);
    // g1 is the scaled g0 through the g function
        g1=sqrt(g0);
        g1=exp(-alpha*g1);
    //save
        g1p[pt]=g1;
}
GENERATE_CUDA_FUNC2DROWS(computeG2,
                     (float* pp, float* g1p, float* gxp, float* gyp, float lambda, int cols),
                     (pp, g1p, gxp, gyp, lambda, cols)) {
    int x = threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int dnoff=(y<gridDim.y*blockDim.y-1)*cols;
    //itr0
    int pt=x+y*cols;
    float g1h,g1n,g1u,g1d,g1r,g1l,gx,gy;
    float tmp;
//part2, find gx,gy
    x = threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    //itr0
    pt=x+y*cols;

    g1h=g1p[pt];
    g1n=g1p[pt+blockDim.x];
    g1r=__shfl_down(g1h,1);
    tmp=__shfl_up(g1n,31);
    if(threadIdx.x>=31){
        g1r=tmp;
    }
    g1l=g1h;
    g1u=g1h;
    g1d=g1p[pt+dnoff];

    gx=fmaxf(g1l,g1r);
    gy=fmaxf(g1u,g1d);

    //save
        gxp[pt]=gx;
        gyp[pt]=gy;
    x+=32;
    //itr 1:n-2
    for(;x<cols-32;x+=32){
        pt=x+y*cols;
        g1h=g1n;
        g1n=g1p[pt+blockDim.x];
        g1r=__shfl_down(g1h,1);
        tmp=__shfl_up(g1n,31);
        g1r=threadIdx.x>=31?tmp:g1r;

        g1l=g1h;
        g1u=g1h;
        g1d=g1p[pt+dnoff];

        gx=fmaxf(g1l,g1r);
        gy=fmaxf(g1u,g1d);
        //save
            gxp[pt]=gx;
            gyp[pt]=gy;
    }

    //itr n-1
    pt=x+y*cols;
    g1h=g1n;
    g1r=__shfl_down(g1h,1);
    g1l=g1h;
    g1u=g1h;
    g1d=g1p[pt+dnoff];

    gx=fmaxf(g1l,g1r);
    gy=fmaxf(g1u,g1d);


    //save
        gxp[pt]=gx;
        gyp[pt]=gy;

}


//This version is faster, but makes synchronization errors at the lines between parts 1 and 2.
//Could be fixed by a second pass for part 2 over the stitch lines, but I don't have time to figure that out
//right now.
GENERATE_CUDA_FUNC2DROWS(computeGunsafe,
                     (float* pp, float* g1p, float* gxp, float* gyp, float lambda, int cols),
                     (pp, g1p, gxp, gyp, lambda, cols)) {
//TODO: make compatible with cuda 2.0 and lower (remove shuffles). Probably through texture fetch
//TODO: rerun kernel on lines with y%32==31 or y%32==0 to fix stitch lines

//Original pseudocode for this function:
    // //subscripts u,d,l,r mean up,down,left,right
    // void computeG(){
    //     // g0 is the strongest nearby gradient (excluding point defects)
    //     g0x=fabsf(pr-pl);//|dx|
    //     g0y=fabsf(pd-pu);//|dy|
    //     g0=max(g0x,g0y);
    //     // g1 is the scaled g0 through the g function exp(-alpha*x^beta)
    //     g1=sqrt(g0); //beta=0.5
    //     alpha=3.5;
    //     g1=exp(-alpha*g1);
    //     //hard to explain this without a picture, but breaks are where both neighboring pixels are near a change
    //     gx=max(g1r,g1);
    //     gy=max(g1d,g1);
    //     gu=gyu;  //upper spring is the lower spring of the pixel above
    //     gd=gy;   //lower spring
    //     gr=gx;   //right spring
    //     gl=gxl;  //left spring is the right spring of the pixel to the left
    // }
    const float alpha=3.5f;
    int x = threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int upoff=-(y!=0)*cols;
    int dnoff=(y<gridDim.y*blockDim.y-1)*cols;
    //itr0
    int pt=x+y*cols;
    float ph,pn,pu,pd,pl,pr;
    float g0x,g0y,g0,g1,g1h,g1n,g1u,g1d,g1r,g1l,gx,gy,gt,gsav;
    float tmp;
    ph=pp[pt];
    pn=pp[pt+blockDim.x];

    pr=__shfl_down(ph,2);
    tmp=__shfl_up(pn,30);
    if(threadIdx.x>=30){
        pr=tmp;
    }
    pl=ph;
    pu=pp[pt+upoff];
    pd=pp[pt+dnoff];


    // g0 is the strongest nearby gradient (excluding point defects)
        gt=fabsf(pr-pl);
        g0x=__shfl_up(gt,1);//?xxxxxx no prior val
        gsav=__shfl_down(gt,31);//x000000 for next time
        g0x=threadIdx.x>0?g0x:0.0f;//0xxxxxx
        g0y=fabsf(pd-pu);

        g0=fmaxf(g0x,g0y);
    // g1 is the scaled g0 through the g function
        g1=sqrt(g0);
        g1=exp(-alpha*g1);
    //save
        g1p[pt]=g1;

    x+=32;
    //itr 1:n-2
    for(;x<cols-32;x+=32){
        pt=x+y*cols;
        ph=pn;
        pn=pp[pt+blockDim.x];
        pr=__shfl_down(ph,2);
        tmp=__shfl_up(pn,30);
        pr=threadIdx.x>=30?tmp:pr;

        pl=ph;
        pu=pp[pt+upoff];
        pd=pp[pt+dnoff];

        // g0 is the strongest nearby gradient (excluding point defects)
            gt=fabsf(pr-pl);
            g0x=__shfl_up(gt,1);//?xxxxxx
            g0x=threadIdx.x>0?g0x:gsav;//xxxxxxx
            gsav=__shfl_down(gt,31);//x000000 for next time
            g0y=fabsf(pd-pu);

            g0=fmaxf(g0x,g0y);

        // g1 is the scaled g0 through the g function
            g1=sqrt(g0);
            g1=exp(-alpha*g1);
        //save
            g1p[pt]=g1;
    }

    //itr n-1
    pt=x+y*cols;
    ph=pn;
    pr=__shfl_down(ph,2);
    pl=ph;
    pu=pp[pt+upoff];
    pd=pp[pt+dnoff];

    // g0 is the strongest nearby gradient (excluding point defects)
        gt=fabsf(pr-pl);
        g0x=__shfl_up(gt,1);//?xxxxxx
        g0x=threadIdx.x>0?g0x:gsav;//xxxxxxx
        g0y=fabsf(pd-pu);

        g0=fmaxf(g0x,g0y);
    // g1 is the scaled g0 through the g function
        g1=sqrt(g0);
        g1=exp(-alpha*g1);
    //save
        g1p[pt]=g1;

//part2, find gx,gy
    x = threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    //itr0
    pt=x+y*cols;

    g1h=g1p[pt];
    g1n=g1p[pt+blockDim.x];
    g1r=__shfl_down(g1h,1);
    tmp=__shfl_up(g1n,31);
    if(threadIdx.x>=31){
        g1r=tmp;
    }
    g1l=g1h;
    g1u=g1h;
    g1d=g1p[pt+dnoff];

    gx=fmaxf(g1l,g1r);
    gy=fmaxf(g1u,g1d);

    //save
        gxp[pt]=gx;
        gyp[pt]=gy;
    x+=32;
    //itr 1:n-2
    for(;x<cols-32;x+=32){
        pt=x+y*cols;
        g1h=g1n;
        g1n=g1p[pt+blockDim.x];
        g1r=__shfl_down(g1h,1);
        tmp=__shfl_up(g1n,31);
        g1r=threadIdx.x>=31?tmp:g1r;

        g1l=g1h;
        g1u=g1h;
        g1d=g1p[pt+dnoff];

        gx=fmaxf(g1l,g1r);
        gy=fmaxf(g1u,g1d);
        //save
            gxp[pt]=gx;
            gyp[pt]=gy;
    }

    //itr n-1
    pt=x+y*cols;
    g1h=g1n;
    g1r=__shfl_down(g1h,1);
    g1l=g1h;
    g1u=g1h;
    g1d=g1p[pt+dnoff];

    gx=fmaxf(g1l,g1r);
    gy=fmaxf(g1u,g1d);


    //save
        gxp[pt]=gx;
        gyp[pt]=gy;

}

__device__ inline float saturate(float x){
    //return x;
    return x/fmaxf(1.0f,fabsf(x));
}

__global__ void updateQD  (float* gqxpt, float* gqypt, float *dpt, float * apt,
                float *gxpt, float *gypt, int cols, float sigma_q, float sigma_d, float epsilon,
                float theta);//DANGER, no interblock synchronization = weird instability
static __global__ void updateQ  (float* gqxpt, float* gqypt, float *dpt, float * apt,
                float *gxpt, float *gypt, int cols, float sigma_q, float sigma_d, float epsilon,
                float theta);
static __global__ void updateD  (float* gqxpt, float* gqypt, float *dpt, float * apt,
                float *gxpt, float *gypt, int cols, float sigma_q, float sigma_d, float epsilon,
                float theta);

void updateQDCaller(float* gqxpt, float* gqypt, float *dpt, float * apt,
        float *gxpt, float *gypt, int cols, float sigma_q, float sigma_d, float epsilon,
        float theta) {

    dim3 dimBlock(BLOCKX2D, BLOCKY2D);
    dim3 dimGrid(1, (arows + dimBlock.y - 1) / dimBlock.y);
    assert(dimGrid.y>0);
    cudaSafeCall( cudaGetLastError() );
    updateQ<<<dimGrid, dimBlock,0,localStream>>>( gqxpt, gqypt, dpt, apt,
            gxpt, gypt, cols, sigma_q, sigma_d, epsilon, theta);
    cudaSafeCall( cudaGetLastError() );
    updateD<<<dimGrid, dimBlock,0,localStream>>>( gqxpt, gqypt, dpt, apt,
            gxpt, gypt, cols, sigma_q, sigma_d, epsilon, theta);
    cudaSafeCall( cudaGetLastError() );
};
__global__ void updateQD  (float* gqxpt, float* gqypt, float *dpt, float * apt,
                float *gxpt, float *gypt, int cols, float sigma_q, float sigma_d, float epsilon,
                float theta) {
    //TODO: make compatible with cuda 2.0 and lower (remove shuffles). Probably through texture fetch

    //Original pseudocode for this function:
//void updateQD(){
//    //shifts are shuffles!
//    for (all x in blocks of warpsize;;){
//        //qx update
//        float dh,dn,qxh,gx,gqx,qyh,gy,gqy;
//        //load(dh,dn,gxh,gqx);//load here, next(the block to the right), local constant, old x force(with cached multiply)
//        dr=dh<<1;
//        tmp=dn>>31;
//        if (rt)
//            dr=tmp;
//        qxh=gqx/gxh;
//        qxh = (qxh+sigma_q*gxh*(dr-dh))/(1+sigma_q*epsilon);//basic spring force equation f=k(x-x0)
//        gqx = saturate(gxh*qxh);//spring saturates (with cached multiply), saturation force proportional to prob. of not an edge.
//        gqxpt[pt]=gqx;
//
//        //qy update
//        s[bpt]=dn;
//        if(!btm){
//            dd=s[bpt+bdnoff];
//        }else{
//            dd=dpt[pt+dnoff];
//        }
//        qyh=gqy/gy;
//        qyh=(qyh+sigma_q*gyh*(dd-dh))/(1+sigma_q*epsilon);
//        gqy=saturate(gyh*qyh);
//        gqypt[pt]=gqy;
//
//        //dx update
//        gqr=gqx;
//        gql=gqx>>1;
//        if (lf)
//            gql=gqsave;
//        gqsave=gqx<<31;//save for next iter
//        dacc = gqr - gql;//dx part
//
//        //dy update and d store
//        gqd=gqy;
//        s[bpt]=gqy;
//        if(!top)
//            gqu=s[bpt+bupoff];
//        else
//            gqu=gqxpt[pt + upoff];
//        dacc += gqd-gqu; //dy part
//        d = (d + sigma_d*(dacc+1/theta*ah))/(1+sigma_d/theta);
//        dpt[pt]=d;
//    }
//}
    __shared__ float s[32*BLOCKY2D];
    int x = threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    bool rt=x==31;
    bool lf=x==0;
    bool top=y==0;
    bool btm=y==gridDim.y*blockDim.y-1;
    bool btop=threadIdx.y==0;
    bool bbtm=threadIdx.y==blockDim.y-1;
    int pt, bpt,bdnoff ,dnoff, bupoff, upoff;


    float tmp,gqsave;
    gqsave=0;
    bpt = threadIdx.x+threadIdx.y*blockDim.x;
    bdnoff=blockDim.x;
    dnoff=(!btm)*cols;
    bupoff=-blockDim.x;
    upoff=-(!top)*cols;

    pt=x+y*cols;

    float dh,dn;
    dn=dpt[pt];


    for(;x<cols;x+=32){
        float qx,gx,gqx,qy,gy,gqy;
        pt=x+y*cols;


        //qx update
        {
            float dr;
            //load(dh,dn,gxh,gqx);//load here, next(the block to the right), local constant, old x force(with cached multiply)

            //load
            {
                dh=dn;
                if(x<cols-32){
                    dn=dpt[pt+32];

                }
                gqx=gqxpt[pt];
                gx=gxpt[pt];
//                gx=1.0f;

            }

            dr=__shfl_down(dh,1);
            tmp=__shfl_up(dn,31);
            if (rt && x<cols-32)
                dr=tmp;
            qx = gqx/gx;
            qx = (qx+sigma_q*gx*(dr-dh))/(1+sigma_q*epsilon);//basic spring force equation f=k(x-x0)
            gqx = saturate(gx*qx);//spring saturates (with cached multiply), saturation force proportional to prob. of not an edge.
            //gqxpt[pt]=gqx;
        }
        dpt[pt] = dh;
        //qy update
        {
            float dd;
            //load
                    {
                        gqy=gqypt[pt];
                        gy=gypt[pt];
//                        gy=1.0f;
                    }
            s[bpt]=dh;
            __syncthreads();
            if(!bbtm){
                dd=s[bpt+bdnoff];
            }else{
                dd=dpt[pt+dnoff];
            }
            qy = gqy/gy;
            qy = (qy+sigma_q*gy*(dd-dh))/(1+sigma_q*epsilon);
            gqy = saturate(gy*qy);
            //gqypt[pt]=gqy;
        }
        float dacc;
        //dx update
        {
            float gqr,gql;
            gqr=gqx;
            gql=__shfl_up(gqx,1);
            if (lf)
                gql=gqsave;
            gqsave=__shfl_down(gqx,31);//save for next iter
            dacc = gqr - gql;//dx part
        }
        float d=dh;
        //dy update and d store
        {
            float a;
            //load
            {
                a=apt[pt];
            }
            float gqu,gqd;

            gqd=gqy;
            s[bpt]=gqy;
            __syncthreads();
            if(!btop)
                gqu=s[bpt+bupoff];
            else
                gqu=gqypt[pt + upoff];
            if(y==0)
                gqu=0.0f;
            dacc += gqd-gqu; //dy part
            d = ( d + sigma_d*(dacc + a/theta) ) / (1 + sigma_d/theta);
            //dpt[pt] = d;
        }
        __syncthreads();
        gqxpt[pt]=gqx;
        gqypt[pt]=gqy;
        dpt[pt] = d;
        __syncthreads();
    }
}


GENERATE_CUDA_FUNC2DROWS(updateQ,
                (float* gqxpt, float* gqypt, float *dpt, float * apt,
                float *gxpt, float *gypt, int cols, float sigma_q, float sigma_d, float epsilon,
                float theta),
                ( gqxpt, gqypt, dpt, apt,
                        gxpt, gypt, cols, sigma_q, sigma_d, epsilon, theta)) {
    //TODO: make compatible with cuda 2.0 and lower (remove shuffles). Probably through texture fetch

    //Original pseudocode for this function:
//void updateQD(){
//    //shifts are shuffles!
//    for (all x in blocks of warpsize;;){
//        //qx update
//        float dh,dn,qxh,gx,gqx,qyh,gy,gqy;
//        //load(dh,dn,gxh,gqx);//load here, next(the block to the right), local constant, old x force(with cached multiply)
//        dr=dh<<1;
//        tmp=dn>>31;
//        if (rt)
//            dr=tmp;
//        qxh=gqx/gxh;
//        qxh = (qxh+sigma_q*gxh*(dr-dh))/(1+sigma_q*epsilon);//basic spring force equation f=k(x-x0)
//        gqx = saturate(gxh*qxh);//spring saturates (with cached multiply), saturation force proportional to prob. of not an edge.
//        gqxpt[pt]=gqx;
//
//        //qy update
//        s[bpt]=dn;
//        if(!btm){
//            dd=s[bpt+bdnoff];
//        }else{
//            dd=dpt[pt+dnoff];
//        }
//        qyh=gqy/gy;
//        qyh=(qyh+sigma_q*gyh*(dd-dh))/(1+sigma_q*epsilon);
//        gqy=saturate(gyh*qyh);
//        gqypt[pt]=gqy;
//
//        //dx update
//        gqr=gqx;
//        gql=gqx>>1;
//        if (lf)
//            gql=gqsave;
//        gqsave=gqx<<31;//save for next iter
//        dacc = gqr - gql;//dx part
//
//        //dy update and d store
//        gqd=gqy;
//        s[bpt]=gqy;
//        if(!top)
//            gqu=s[bpt+bupoff];
//        else
//            gqu=gqxpt[pt + upoff];
//        dacc += gqd-gqu; //dy part
//        d = (d + sigma_d*(dacc+1/theta*ah))/(1+sigma_d/theta);
//        dpt[pt]=d;
//    }
//}
    __shared__ float s[32*BLOCKY2D];
    int x = threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    bool rt=x==31;

    bool bbtm=threadIdx.y==blockDim.y-1;
    int pt, bpt,bdnoff ,dnoff;


    float tmp;
    bpt = threadIdx.x+threadIdx.y*blockDim.x;
    bdnoff=blockDim.x;
    dnoff=(y<gridDim.y*blockDim.y-1)*cols;



    pt=x+y*cols;

    float dh,dn;
    dn=dpt[pt];


    for(;x<cols;x+=32){
        float qx,gx,gqx,qy,gy,gqy;
        pt=x+y*cols;


        //qx update
        {
            float dr;
            //load(dh,dn,gxh,gqx);//load here, next(the block to the right), local constant, old x force(with cached multiply)

            //load
            {
                dh=dn;
                if(x<cols-32){
                    dn=dpt[pt+32];

                }
                gqx=gqxpt[pt];
                gx=gxpt[pt]+.01f;
//                gx=1.0f;
            }

            dr=__shfl_down(dh,1);
            tmp=__shfl_up(dn,31);
            if (rt && x<cols-32)
                dr=tmp;
            qx = gqx/gx;
            //qx+=(gx*(dr-dh)-epsilon*qx)*.5f;//simplified step
            qx = (qx+sigma_q*gx*(dr-dh))/(1+sigma_q*epsilon);//basic spring force equation f=k(x-x0)
            gqx = saturate(gx*qx);//spring saturates (with cached multiply), saturation force proportional to prob. of not an edge.
            gqxpt[pt]=gqx;
        }

        //qy update
        {
            float dd;
            //load
                    {
                        gqy=gqypt[pt];
                        gy=gypt[pt]+.01f;
//                        gy=1.0f;
                    }
            s[bpt]=dh;
            __syncthreads();
            if(!bbtm)
                dd=s[bpt+bdnoff];
            else
                dd=dpt[pt+dnoff];
            __syncthreads();
            qy = gqy/gy;
            //qy+=(gy*(dd-dh)-epsilon*qy)*.5f;//simplified step
            qy = (qy+sigma_q*gy*(dd-dh))/(1+sigma_q*epsilon);
            gqy = saturate(gy*qy);

            gqypt[pt]=gqy;
        }
        //__syncthreads();
    }
}

GENERATE_CUDA_FUNC2DROWS(updateD,
                (float* gqxpt, float* gqypt, float *dpt, float * apt,
                float *gxpt, float *gypt,int cols, float sigma_q, float sigma_d, float epsilon,
                float theta),
                ( gqxpt, gqypt, dpt, apt,
                        gxpt, gypt, cols, sigma_q, sigma_d, epsilon, theta)) {
    //TODO: make compatible with cuda 2.0 and lower (remove shuffles). Probably through texture fetch

    //Original pseudocode for this function:
//void updateQD(){
//    //shifts are shuffles!
//    for (all x in blocks of warpsize){
//        //qx update
//        float dh,dn,qxh,gx,gqx,qyh,gy,gqy;
//        //load(dh,dn,gxh,gqx);//load here, next(the block to the right), local constant, old x force(with cached multiply)
//        dr=dh<<1;
//        tmp=dn>>31;
//        if (rt)
//            dr=tmp;
//        qxh=gqx/gxh;
//        qxh = (qxh+sigma_q*gxh*(dr-dh))/(1+sigma_q*epsilon);//basic spring force equation f=k(x-x0)
//        gqx = saturate(gxh*qxh);//spring saturates (with cached multiply), saturation force proportional to prob. of not an edge.
//        gqxpt[pt]=gqx;
//
//        //qy update
//        s[bpt]=dn;
//        if(!btm){
//            dd=s[bpt+bdnoff];
//        }else{
//            dd=dpt[pt+dnoff];
//        }
//        qyh=gqy/gy;
//        qyh=(qyh+sigma_q*gyh*(dd-dh))/(1+sigma_q*epsilon);
//        gqy=saturate(gyh*qyh);
//        gqypt[pt]=gqy;
//
//        //dx update
//        gqr=gqx;
//        gql=gqx>>1;
//        if (lf)
//            gql=gqsave;
//        gqsave=gqx<<31;//save for next iter
//        dacc = gqr - gql;//dx part
//
//        //dy update and d store
//        gqd=gqy;
//        s[bpt]=gqy;
//        if(!top)
//            gqu=s[bpt+bupoff];
//        else
//            gqu=gqxpt[pt + upoff];
//        dacc += gqd-gqu; //dy part
//        d = (d + sigma_d*(dacc+1/theta*ah))/(1+sigma_d/theta);
//        dpt[pt]=d;
//    }
//}
    __shared__ float s[32*BLOCKY2D];
    int x = threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    bool lf=x==0;
    bool top=y==0;
    bool btop=threadIdx.y==0;
    int pt, bpt, bupoff, upoff;


    float gqsave=0;
    bpt = threadIdx.x+threadIdx.y*blockDim.x;

    bupoff=-blockDim.x;
    upoff=-(!top)*cols;

    pt=x+y*cols;




    for(;x<cols;x+=32){
        float gqx,gqy;
        pt=x+y*cols;


        float dacc;
        //dx update
        {
            float gqr,gql;
            gqr=gqx=gqxpt[pt];
            gql=__shfl_up(gqx,1);
            if (lf)
                gql=gqsave;
            gqsave=__shfl_down(gqx,31);//save for next iter
            dacc = gqr - gql;//dx part
        }
        //dy update and d store
        {
            float a;
            //load
            {
                a=apt[pt];
            }
            float gqu,gqd;
            float d=dpt[pt];
            gqd=gqy=gqypt[pt];
            s[bpt]=gqy;
            __syncthreads();
            if(!btop)
                gqu=s[bpt+bupoff];
            else
                gqu=gqypt[pt + upoff];
            if(y==0)
                gqu=0.0f;
            dacc += gqd-gqu; //dy part
            //d += dacc*.5f;//simplified step
            d = ( d + sigma_d*(dacc + a/theta) ) / (1 + sigma_d/theta);

            dpt[pt] = d;
        }
        __syncthreads();//can't figure out why this is needed, but it is to avoid subtle errors in Qy at the ends of the warp
    }
}

}}}}




