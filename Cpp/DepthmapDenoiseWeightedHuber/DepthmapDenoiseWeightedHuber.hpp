// Free for non-commercial, non-military, and non-critical 
// use unless incorporated in OpenCV. 
// Inherits OpenCV Licence if in OpenCV.


#ifndef DepthmapDenoiseWeightedHuber_H
#define DepthmapDenoiseWeightedHuber_H
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/core.hpp>

namespace cv{
    namespace cuda{
        /////////////////////////////////////////
        // DepthMapDenoiseWeightedHuber

        //! OpenDTAM Variant of Chambolle & Pock denoising
        //!
        //! The complicated half of the DTAM algorithm's mapping core,
        //! but can be used independently to refine depthmaps.
        //!
        //! Contributed by Paul Foster for GSoC 2014
        //! Original algorithm described by Richard Newcombe, Steven J. Lovegrove, and Andrew J. Davison. 
        //! "DTAM: Dense tracking and mapping in real-time."
        class CV_EXPORTS DepthmapDenoiseWeightedHuber : public cv::Algorithm
        {
        public:
            //! This may be called repeatedly to iteratively refine the internal depthmap
            virtual cv::cuda::GpuMat operator()(InputArray input,
                                                float epsilon,
                                                float theta) = 0;
            
            //! In case you want to do these explicitly
            virtual void allocate(int rows, int cols, InputArray gx=GpuMat(),InputArray gy=GpuMat()) = 0;
            virtual void cacheGValues() = 0;
        };

        //! The visibleLightImage is a CV_32FC1 grayscale image of the scene, which can be used as a hint for edge placement.
        CV_EXPORTS Ptr<DepthmapDenoiseWeightedHuber>
            createDepthmapDenoiseWeightedHuber(InputArray visibleLightImage=GpuMat(), Stream cvStream=Stream::Null());

            
            
            
        class DepthmapDenoiseWeightedHuberImpl : public DepthmapDenoiseWeightedHuber
        {
        public:
            //CostVolume cv;//The cost volume we are attached to
            
            DepthmapDenoiseWeightedHuberImpl(const cv::cuda::GpuMat& visibleLightImage=cv::cuda::GpuMat(),cv::cuda::Stream cvStream=cv::cuda::Stream::Null());
            cv::cuda::GpuMat operator()(InputArray ain,
                                        float epsilon,
                                        float theta);




            


            //public parameters
            float epsilon,theta;

            
            cv::cuda::GpuMat visibleLightImage;
            //buffers
            cv::cuda::GpuMat _qx,_qy,_d,_a,_g,_g1,_gx,_gy;
            cv::cuda::GpuMat stableDepth;
            float getTheta(){return theta;}
            
            //in case you want to do these explicitly
            void allocate(int rows,int cols, InputArray gxin=cv::cuda::GpuMat(), InputArray gyin=cv::cuda::GpuMat());
            void cacheGValues();
            
        private:
            int rows;
            int cols;
            
            void computeSigmas(float epsilon,float theta);
            

            //internal parameter values
            float sigma_d,sigma_q;

            //flags
            bool cachedG;
            int alloced;
            int dInited;
            

        public:
            cv::cuda::Stream cvStream;

            
        };
        

        
    }
}
#endif // DepthmapDenoiseWeightedHuber_H
