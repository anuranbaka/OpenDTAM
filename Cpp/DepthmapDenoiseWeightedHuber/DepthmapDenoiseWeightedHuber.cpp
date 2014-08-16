// Free for non-commercial, non-military, and non-critical 
// use unless incorporated in OpenCV. 
// Inherits OpenCV Licence if in OpenCV.


#include "DepthmapDenoiseWeightedHuber.hpp"
#include "DepthmapDenoiseWeightedHuber.cuh"
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core.hpp>


using namespace std;
using namespace cv;
using namespace cv::cuda;
namespace cv{
    namespace cuda{
        static void memZero(GpuMat& in,Stream& cvStream);

        static void memZero(GpuMat& in,Stream& cvStream){
            cudaSafeCall(cudaMemsetAsync(in.data,0,in.rows*in.cols*sizeof(float),cv::cuda::StreamAccessor::getStream(cvStream)));
        }

        DepthmapDenoiseWeightedHuberImpl::DepthmapDenoiseWeightedHuberImpl(const cv::cuda::GpuMat& visibleLightImage,
                                                                Stream cvStream) : 
                                                                rows(visibleLightImage.rows),
                                                                cols(visibleLightImage.cols),
                                                                visibleLightImage(visibleLightImage), 
                                                                cvStream(cvStream)
        {
            
            alloced=0;
            cachedG=0;
            dInited=0;
        }

        Ptr<DepthmapDenoiseWeightedHuber>
        CV_EXPORTS createDepthmapDenoiseWeightedHuber(InputArray visibleLightImage, Stream cvStream){
            return Ptr<DepthmapDenoiseWeightedHuber>(new DepthmapDenoiseWeightedHuberImpl(visibleLightImage.getGpuMat(),cvStream));
        };
    }
}





