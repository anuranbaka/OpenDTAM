#include "Cost.h"

#define COST_CPP_SUBPARTS
#include "updateCost.part.cpp"
#include "min.part.cpp"
#undef COST_CPP_SUBPARTS
#include "updateCost.part.hpp"
#define COST_CPP_DATA_MIN 3
#define COST_CPP_INITIAL_WEIGHT .001


Cost::Cost(const cv::Mat& baseImage, int layers, const cv::Mat& cameraMatrix, const cv::Mat& R, const cv::Mat& Tr):
baseImage(baseImage),
rows(baseImage.rows),
cols(baseImage.cols),
layers(layers),
depth(generateDepths(layers)),
cameraMatrix(cameraMatrix),
pose(convertPose(R,Tr)),
lo(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN))),
hi(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN))),
dataContainer(cv::Mat(baseImage.rows*baseImage.cols*layers,1,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN))),//allocate enough data to hold all of the cost volume
hitContainer(cv::Mat(baseImage.rows*baseImage.cols*layers,1,cv::DataType<float>::type, cv::Scalar(COST_CPP_INITIAL_WEIGHT)))//allocate enough data to hold all of the hits info in cost volume
{
    init();
}


Cost::Cost(const cv::Mat& baseImage, int layers, const cv::Mat& cameraMatrix, const cv::Matx44d& cameraPose):
baseImage(baseImage),
rows(baseImage.rows),
cols(baseImage.cols),
layers(layers),
depth(generateDepths(layers)),
cameraMatrix(cameraMatrix),
pose(cameraPose),
dataContainer(cv::Mat(baseImage.rows*baseImage.cols*layers,1,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN))),//allocate enough data to hold all of the cost volume
hitContainer(cv::Mat(baseImage.rows*baseImage.cols*layers,1,cv::DataType<float>::type, cv::Scalar(COST_CPP_INITIAL_WEIGHT))),//allocate enough data to hold all of the hits info in cost volume
lo(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN))),
hi(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN)))
{
    init();
}


Cost::Cost(const cv::Mat& baseImage, const std::vector<float>& depth, const cv::Mat& cameraMatrix, const cv::Mat& R, const cv::Mat& Tr):
baseImage(baseImage),
rows(baseImage.rows),
cols(baseImage.cols),
depth(depth),
layers(depth.size()),
cameraMatrix(cameraMatrix),
pose(convertPose(R,Tr)),
dataContainer(cv::Mat(baseImage.rows*baseImage.cols*depth.size(),1,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN))),//allocate enough data to hold all of the cost volume
hitContainer(cv::Mat(baseImage.rows*baseImage.cols*depth.size(),1,cv::DataType<float>::type, cv::Scalar(COST_CPP_INITIAL_WEIGHT))),//allocate enough data to hold all of the hits info in cost volume
lo(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN))),
hi(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN)))
{
    init();
}


Cost::Cost(const cv::Mat& baseImage, const std::vector<float>& depth, const cv::Mat& cameraMatrix, const cv::Matx44d& cameraPose):
baseImage(baseImage),
rows(baseImage.rows),
cols(baseImage.cols),
layers(depth.size()),
depth(depth),
cameraMatrix(cameraMatrix),
pose(cameraPose),
dataContainer(cv::Mat(baseImage.rows*baseImage.cols*depth.size(),1,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN))),//allocate enough data to hold all of the cost volume
hitContainer(cv::Mat(baseImage.rows*baseImage.cols*depth.size(),1,cv::DataType<float>::type, cv::Scalar(COST_CPP_INITIAL_WEIGHT))),//allocate enough data to hold all of the hits info in cost volume
lo(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN))),
hi(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN)))
{
    init();
}

const cv::Mat Cost::depthMap(){
    //Returns the best available depth map
    // Code should not rely on the particular mapping of true 
    // internal data to true inverse depth, as this may change.
    // Currently depth is just a constant multiple of the index, so
    // infinite depth is always represented. This is likely to change.
    if(stableDepth.data){
        return stableDepth*depthStep;
    }
    return _a*depthStep;
}


