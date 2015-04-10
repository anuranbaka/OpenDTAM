// Free for non-commercial, non-military, and non-critical 
// use unless incorporated in OpenCV. 
// Inherits OpenCV Licence if in OpenCV.


#ifndef COSTVOLUME_HPP
#define COSTVOLUME_HPP


#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

typedef  int FrameID;

class CostVolume
{
public:
    FrameID fid;
    int rows;
    int cols;
    int layers;
    float near; //inverse depth of center of voxels in layer layers-1
    float far;  //inverse depth of center of voxels in layer 0
    float depthStep;
    float initialWeight;
    cv::Mat cameraMatrix;//Note! should be in OpenCV format
    cv::Mat R;
    cv::Mat T;
    cv::Mat projection;//projects world coordinates (x,y,z) into (rows,cols,layers)

    cv::cuda::GpuMat baseImage;
    cv::cuda::GpuMat baseImageGray;
    cv::cuda::GpuMat lo;
    cv::cuda::GpuMat hi;
    cv::cuda::GpuMat loInd;

    float * data;
    float * hits;

    cv::cuda::GpuMat dataContainer;
    cv::cuda::GpuMat hitContainer;

    int count;
    cv::cuda::Stream cvStream;

    void updateCost(const cv::Mat& image, const cv::Mat& R, const cv::Mat& T);//Accepts pinned RGBA8888 or BGRA8888 for high speed
    
    CostVolume(){}
    ~CostVolume();
    CostVolume(cv::Mat image, FrameID _fid, int _layers, float _near, float _far,
            cv::Mat R, cv::Mat T, cv::Mat _cameraMatrix, float initialCost=3.0, float initialWeight=.001);

    //HACK: remove this function in release
    cv::Mat downloadOldStyle( int layer){
        cv::Mat cost;
        cv::cuda::GpuMat tmp=dataContainer.rowRange(layer,layer+1);
        tmp.download(cost);
        cost=cost.reshape(0,rows);
        return cost;
    }

private:
    void solveProjection(const cv::Mat& R, const cv::Mat& T);
    void checkInputs(const cv::Mat& R, const cv::Mat& T,
            const cv::Mat& _cameraMatrix);
    void simpleTex(const cv::Mat& image,cv::cuda::Stream cvStream=cv::cuda::Stream::Null());

private:
    //temp variables ("static" containers)
    cv::Ptr<char> _cuArray;//Ptr<cudaArray*> really
    cv::Ptr<char> _texObj;//Ptr<cudaTextureObject_t> really
    cv::Mat cBuffer;//Must be pagable
    cv::Ptr<char> ref;
};

#endif // COSTVOLUME_HPP
