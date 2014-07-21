namespace cv { namespace gpu { namespace device {
    namespace dtam_updateCost{

__constant__ float c_sliceToIm[3 * 3];
__constant__ int  layers;
__constant__ int layerStep;
__constant__ float* hdata;
__constant__ float* cdata;
__constant__ float* lo;
__constant__ float* hi;
__constant__ uint* loInd;

void loadConstants(float* h_sliceToIm, int h_layers, int h_layerStep, float* h_hdata, float* h_cdata,float* lo, float* hi, uint* loInd){
    cudaMemcpyToSymbol(c_sliceToIm, h_sliceToIm, 3*3*sizeof(float));
    cudaMemcpyToSymbol(&layers, h_layers, sizeof(int));
    cudaMemcpyToSymbol(&layerStep, h_layerStep, sizeof(int));
    cudaMemcpyToSymbol(&hdata, h_hdata, sizeof(float*));
    cudaMemcpyToSymbol(&cdata, h_cdata, sizeof(float*));
    cudaMemcpyToSymbol(&lo, h_lo, sizeof(float*));
    cudaMemcpyToSymbol(&hi, h_hi, sizeof(float*));
    cudaMemcpyToSymbol(&loInd, h_loInd, sizeof(uint*));
}



template<typename V>
__global__ void updateCostCol( V* baseRow,
         cudaTextureObject_t tex) 
{
    //    x  y  z  1 
    // x  0     1  2
    // y  3     4  5
    // z  6     7  8
    
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    
    // Find the current base color
    V B = baseRow[x];

    float3 Z;
    float3 W;
    Z.x = c_sliceToIm[1];
    Z.y = c_sliceToIm[4];
    Z.z = c_sliceToIm[7];

    //add in the offset for the current column
    W.x = c_sliceToIm[2] + c_sliceToIm[0] * x;
    W.y = c_sliceToIm[5] + c_sliceToIm[3] * x;
    W.z = c_sliceToIm[8] + c_sliceToIm[6] * x;
    hdata = hdata[x];
    cdata = cdata[x];
    float minv = 1000;
    float maxv = 0;
    for (uint z = 0; z < layers; z++, W.x += Z.x, W.y += Z.y, W.z += Z.z) {
        float xi, yi, wi;
        wi = W.z;
        xi = W.x / wi;
        yi = W.y / wi;
        float4 c = tex2D<float4>(tex, xi, yi);

        float v1 = fabsf(c.x - B.x);
        float v2 = fabsf(c.y - B.y);
        float v3 = fabsf(c.z - B.z);
        float h = hdata[z * layerStep] + 1;
        float ns = cdata[z * layerStep] * (1 - 1 / h) + (v1 + v2 + v3) / h;
        if (ns < minv) {
            minv = ns;
            mini = z;
        }
        if (ns > maxv) {
            maxv = ns;
        }

        hdata[z * layerStep] = h;
        cdata[z * layerStep] = ns;
    }
}

}}}}

