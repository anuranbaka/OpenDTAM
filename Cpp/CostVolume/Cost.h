#ifndef COST_H
#define COST_H
#include <opencv2/core/core.hpp>
#include <vector>
// The cost volume. Conceptually arranged as an image plane, corresponding 
// to the keyframe, lying on top of the actual cost volume, a 3D two channel matrix storing 
// the total cost of all rays that have passed through a voxel, and the number of rays that 
// have hit that voxel.
//
// There is also the depth array, which stores the inverse depths of each plane of the cost.
//
// For efficiency, the accumulated cost and hit count are seperate arrays even though 
// conceptually they are just different channels of the same image.
//
// The cost volume doesn't support updating by a different camera than the one that took the 
// keyframe, because that would violate a bunch of assumptions for DTAM
#define COST_H_DEFAULT_NEAR .03

template <class T, class PixelType>// T is the datatype of the cost volume, PixelType is the image 
class Cost{
public:
    cv::Mat_<float> hits;// number of times a cell has been hit
    const cv::Mat_<PixelType> baseImage;
    cv::Mat_<T> lo;
    cv::Mat_<T> hi;
    int rows;
    int cols;
    int layers;
    const cv::Matx33d cameraMatrix;
    const cv::Matx44d pose;//the affine transform representing the world -> camera frame transformation
    T* data;// stores the [rows][cols][layers] array of sum of costs so far
    T* hit;//stores the number of times each cell has been hit by a ray
    const std::vector<T> depth;
    
    Cost<T,PixelType>& operator=( const Cost<T,PixelType>& other );
    Cost();
    Cost(const cv::Mat_<PixelType>& baseImage, int layers,                  const cv::Mat& cameraMatrix, const cv::Mat& R, const cv::Mat& Tr);// autogenerate default depths
    Cost(const cv::Mat_<PixelType>& baseImage, int layers,                  const cv::Mat& cameraMatrix, const cv::Matx44d& cameraPose);// autogenerate default depths
    Cost(const cv::Mat_<PixelType>& baseImage, const std::vector<T>& depth, const cv::Mat& cameraMatrix, const cv::Mat& R, const cv::Mat& Tr);//use given depths
    Cost(const cv::Mat_<PixelType>& baseImage, const std::vector<T>& depth, const cv::Mat& cameraMatrix, const cv::Matx44d& cameraPose);//use given depths
    
    void updateCost(const cv::Mat_<PixelType>& image, const cv::Matx44d& currentCameraPose);
    
    
    
    
    
private:
    cv::Mat_<T> dataContainer; //stores the actual data for data*, used for auto allocation behavior
    cv::Mat_<T> hitContainer; //stores the actual data for hit*, used for auto allocation behavior
    
    T photoCost(const PixelType& a,const PixelType& b);

    std::vector<T> generateDepths(int layers){
        std::vector<T> depths;
        for(T n=0; n<layers; n++){
            depths.push_back(n/(n-1)*COST_H_DEFAULT_NEAR);
        }
        return depths;// generate the depth list, the signature will probably change
    }
    const cv::Matx44d convertPose(const cv::Mat& R, const cv::Mat& Tr){
        cv::Mat pose=cv::Mat::eye(4,4, CV_64F);
        R.copyTo(pose(cv::Range(0,3),cv::Range(0,3)));
        Tr.copyTo(pose(cv::Range(0,3),cv::Range(3,4)));
        return cv::Matx44d(pose);
    }
};
#include "Cost.cpp" //HACK: this should be renamed, but things had to be pulled in to make it compile
#endif