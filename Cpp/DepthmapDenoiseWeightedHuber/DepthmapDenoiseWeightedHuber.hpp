// Free for non-commercial, non-military, and non-critical 
// use unless incorporated in OpenCV. 
// Inherits OpenCV Licence if in OpenCV.


#ifndef DepthmapDenoiseWeightedHuber_H
#define DepthmapDenoiseWeightedHuber_H
#include <opencv2/core/core.hpp>//for CV_EXPORTS
#include <opencv2/gpu/gpu.hpp>
namespace cv{
    namespace gpu{
        /////////////////////////////////////////
        // DepthMapDenoiseWeightedHuber

        //! OpenDTAM Variant of Chambolle & Pock denoising
        //!
        //! The complicated half of the DTAM algorithm's mapping core,
        //! but can be used independently to refine depthmaps.
        //!
        //! Contributed by Paul Foster for GSoC 2014 OpenDTAM project.
        //! High level algorithm described by Richard Newcombe, Steven J. Lovegrove, and Andrew J. Davison. 
        //! "DTAM: Dense tracking and mapping in real-time."
        //! Which was in turn based on Chambolle & Pock's
        //! "A first-order primal-dual algorithm for convex problems with applications to imaging."

        class CV_EXPORTS DepthmapDenoiseWeightedHuber : public cv::Algorithm
        {
        public:
            //! This may be called repeatedly to iteratively refine the internal depthmap
            virtual GpuMat operator()(InputArray input,
                                      float epsilon,
                                      float theta) = 0;

            //! In case you want to do these explicitly
            virtual void allocate(int rows, int cols, InputArray gx = GpuMat(),InputArray gy = GpuMat()) = 0;
            virtual void cacheGValues(InputArray visibleLightImage = GpuMat()) = 0;
            
            virtual void setStream(Stream s) = 0;
            virtual Stream getStream() = 0;
            
            virtual void setAlpha(float alpha) = 0;
            virtual float getAlpha() = 0;
            
            virtual void setBeta(float beta) = 0;
            virtual float getBeta() = 0;
        };

        //! The visibleLightImage is a CV_32FC1 grayscale image of the scene, which can be used as a hint for edge placement.
        CV_EXPORTS Ptr<DepthmapDenoiseWeightedHuber>
            createDepthmapDenoiseWeightedHuber(InputArray visibleLightImage=GpuMat(), Stream cvStream=Stream::Null());
    }
}

    
#include <opencv2/gpu/gpu.hpp>

namespace cv{
    namespace gpu{
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

        class DepthmapDenoiseWeightedHuberImpl : public DepthmapDenoiseWeightedHuber
        {
        public:
            //CostVolume cv;//The cost volume we are attached to
            
            DepthmapDenoiseWeightedHuberImpl(const GpuMat& visibleLightImage=GpuMat(),Stream cvStream=Stream::Null());
            GpuMat operator()(InputArray ain, float epsilon, float theta);

            GpuMat visibleLightImage;
            //buffers
            GpuMat _qx,_qy,_d,_a,_g,_g1,_gx,_gy;
            GpuMat stableDepth;


            //in case you want to do these explicitly
            void allocate(int rows,int cols, InputArray gxin = GpuMat(), InputArray gyin = GpuMat());
            void cacheGValues(InputArray visibleLightImage=GpuMat());

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
            void setStream(Stream s){CV_Assert(!"Not Implemented");};
            Stream getStream(){CV_Assert(!"Not Implemented");};
            
            void setAlpha(float alpha){CV_Assert(!"Not Implemented");};
            float getAlpha(){CV_Assert(!"Not Implemented");};
            
            void setBeta(float beta){CV_Assert(!"Not Implemented");};
            float getBeta(){CV_Assert(!"Not Implemented");};

            Stream cvStream;
        }; 
    }
}
#endif // DepthmapDenoiseWeightedHuber_H
