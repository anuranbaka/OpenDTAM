// Free for non-commercial, non-military, and non-critical 
// use unless incorporated in OpenCV. 
// Inherits OpenCV Licence if in OpenCV.


#ifndef ARRAYTEXTURE_HPP
#define ARRAYTEXTURE_HPP
#include <opencv2/gpu/gpu.hpp>
#include <cuda_runtime.h>
class ArrayTexture {
public:
    int* refcount;
    int ref_count;
    cudaArray* cuArray;
    cudaTextureObject_t texObj;
    
    ArrayTexture(const cv::gpu::CudaMem& image, const cv::gpu::Stream& cvStream =
    cv::gpu::Stream::Null());
    
    ArrayTexture& operator = (const ArrayTexture& tex) {
        if (this != &tex) {
            release();
            
            if (tex.refcount)
                CV_XADD(tex.refcount, 1);
            
            this->refcount=tex.refcount;
        }
        
        return *this;
    }
    
    void release() {
        if (refcount && CV_XADD(refcount, -1) == 1)
            deallocate();
    }
    
    void deallocate(){
        cudaDestroyTextureObject(texObj);
        cudaFreeArray(cuArray);
        
    }
    
    ~ArrayTexture() {
        release();
    }
    
};

#endif // ARRAYTEXTURE_HPP
