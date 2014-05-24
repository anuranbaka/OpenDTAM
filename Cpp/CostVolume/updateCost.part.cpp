#include "Cost.h"
//in Cost.cpp
//TODO: This general form doesn't actually compile,which is actually desired behavior because there is no general form for photoCost
template <class T, class PixelType>
void Cost<T,PixelType>::updateCost(const cv::Mat_<PixelType>& image, 
                const cv::Matx44d& currentCameraPose)
{
    
    for(int n=0; n < depth.size(); ++n){
        cv::Mat_<PixelType> plane;
        cv::Mat_<uchar> mask;
        reproject(image, cameraMatrix, pose, currentCameraPose, depth[n], plane, mask);
        
        //
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                if(mask(i,j)){//valid point
                    data[n+j*layers+i*layers*cols]=photoCost(plane(i,j),baseImage(i,j));
                    hits(i,j)++;
                }
            }
        }
    }
    
    
}
/*
template <class T>
void Cost<T,cv::Vec3f>::updateCost(const cv::Mat_<cv::Vec3f>& image, 
                                   const cv::Matx44d& currentCameraPose)
{
    
    for(int n=0; n < depth.size(); ++n){
        cv::Mat_<cv::Vec3f> plane;
        cv::Mat_<uchar> mask;
        reproject(image, cameraMatrix, pose, currentCameraPose, depth[n], plane, mask);
        
        //
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                if(mask(i,j)){//valid point
                    data[n+j*layers+i*layers*cols]=photoCost(plane(i,j),baseImage(i,j));
                    hits(i,j)++;
                }
            }
        }
    }
    
    
}*/