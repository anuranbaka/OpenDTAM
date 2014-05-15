#include "updateCost.cpp"
#include "reproject.cpp"

Cost(Mat_<PixelType> baseImage, int rows,int cols,int layers, Mat& cameraMatrix, Mat& cameraPose):
    baseImage(baseImage);
    rows(rows),
    cols(cols),
    layers(layers),
    cameraMatrix(cameraMatrix),
    pose(cameraPose),
    container(rows*cols*layers,1,DataType<T>::type, Scalar(0)),//allocate enough data to hold all of the cost volume
    container(rows*cols*layers,1,DataType<T>::type, Scalar(0)),//allocate enough data to hold all of the hits info in cost volume
{
    data=dataContainer.data;
    hit=hitContainer.data;
}
