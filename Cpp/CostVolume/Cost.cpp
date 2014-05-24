#include "Cost.h"
#include "updateCost.part.cpp"

#define COST_CPP_DATA_MIN 0.1

template <class T, class PixelType>
Cost<T,PixelType>::Cost(const cv::Mat_<PixelType>& baseImage, int layers, const cv::Mat& cameraMatrix, const cv::Mat& R, const cv::Mat& Tr):
baseImage(baseImage),
rows(baseImage.rows),
cols(baseImage.cols),
depth(generateDepths(layers)),
layers(depth.size()),
cameraMatrix(cameraMatrix),
pose(convertPose(R,Tr)),
dataContainer(cv::Mat(baseImage.rows*baseImage.cols*layers,1,cv::DataType<T>::type, cv::Scalar(COST_CPP_DATA_MIN))),//allocate enough data to hold all of the cost volume
hitContainer(cv::Mat(baseImage.rows*baseImage.cols*layers,1,cv::DataType<T>::type, cv::Scalar(1))),//allocate enough data to hold all of the hits info in cost volume
lo(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<T>::type, cv::Scalar(COST_CPP_DATA_MIN))),
hi(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<T>::type, cv::Scalar(COST_CPP_DATA_MIN)))
{
    data=(float*)dataContainer.data;
    hit=(float*)hitContainer.data;
}

template <class T, class PixelType>
Cost<T,PixelType>::Cost(const cv::Mat_<PixelType>& baseImage, int layers, const cv::Mat& cameraMatrix, const cv::Matx44d& cameraPose):
baseImage(baseImage),
rows(baseImage.rows),
cols(baseImage.cols),
depth(generateDepths(layers)),
layers(depth.size()),
cameraMatrix(cameraMatrix),
pose(cameraPose),
dataContainer(cv::Mat(baseImage.rows*baseImage.cols*layers,1,cv::DataType<T>::type, cv::Scalar(COST_CPP_DATA_MIN))),//allocate enough data to hold all of the cost volume
hitContainer(cv::Mat(baseImage.rows*baseImage.cols*layers,1,cv::DataType<T>::type, cv::Scalar(1))),//allocate enough data to hold all of the hits info in cost volume
lo(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<T>::type, cv::Scalar(COST_CPP_DATA_MIN))),
hi(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<T>::type, cv::Scalar(COST_CPP_DATA_MIN)))
{
    data=(float*)dataContainer.data;
    hit=(float*)hitContainer.data;
}

template <class T, class PixelType>
Cost<T,PixelType>::Cost(const cv::Mat_<PixelType>& baseImage, const std::vector<T>& depth, const cv::Mat& cameraMatrix, const cv::Mat& R, const cv::Mat& Tr):
baseImage(baseImage),
rows(baseImage.rows),
cols(baseImage.cols),
depth(depth),
layers(depth.size()),
cameraMatrix(cameraMatrix),
pose(convertPose(R,Tr)),
dataContainer(cv::Mat(baseImage.rows*baseImage.cols*layers,1,cv::DataType<T>::type, cv::Scalar(COST_CPP_DATA_MIN))),//allocate enough data to hold all of the cost volume
hitContainer(cv::Mat(baseImage.rows*baseImage.cols*layers,1,cv::DataType<T>::type, cv::Scalar(1))),//allocate enough data to hold all of the hits info in cost volume
lo(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<T>::type, cv::Scalar(COST_CPP_DATA_MIN))),
hi(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<T>::type, cv::Scalar(COST_CPP_DATA_MIN)))
{
    data=(float*)dataContainer.data;
    hit=(float*)hitContainer.data;
}

template <class T, class PixelType>
Cost<T,PixelType>::Cost(const cv::Mat_<PixelType>& baseImage, const std::vector<T>& depth, const cv::Mat& cameraMatrix, const cv::Matx44d& cameraPose):
    baseImage(baseImage),
    rows(baseImage.rows),
    cols(baseImage.cols),
    layers(depth.size()),
    depth(depth),
    cameraMatrix(cameraMatrix),
    pose(cameraPose),
    dataContainer(cv::Mat(baseImage.rows*baseImage.cols*layers,1,cv::DataType<T>::type, cv::Scalar(COST_CPP_DATA_MIN))),//allocate enough data to hold all of the cost volume
    hitContainer(cv::Mat(baseImage.rows*baseImage.cols*layers,1,cv::DataType<T>::type, cv::Scalar(1))),//allocate enough data to hold all of the hits info in cost volume
    lo(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<T>::type, cv::Scalar(COST_CPP_DATA_MIN))),
    hi(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<T>::type, cv::Scalar(COST_CPP_DATA_MIN)))
{
    data=(float*)dataContainer.data;
    hit=(float*)hitContainer.data;
}

template <class T, class PixelType>
Cost<T,PixelType>::Cost(){};

template <class T, class PixelType>
Cost<T,PixelType>& Cost<T,PixelType>::operator=( const Cost<T,PixelType>& other ){
    Cost<T,PixelType> *mycost=new Cost<T,PixelType>(other.baseImage,other.depth,cv::Mat(other.cameraMatrix),other.pose);
    return *mycost;//HACK:sort of a memory leak I think. Don't know of a better mechanism of defeating the constness problem
}