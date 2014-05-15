#include "updateCost.cpp"
#include "reproject.cpp"

Cost(Mat_<PixelType> baseImage, int rows,int cols,int layers, Mat& cameraMatrix, Mat& cameraPose):
    baseImage(baseImage);
    rows(rows),
    cols(cols),
    layers(layers),
    cameraMatrix(cameraMatrix),
    pose(cameraPose),
    container(rows*cols*layers,1,DataType<T>::type, Scalar(0)),
    hits(rows, cols, DataType<float>::type, Scalar(1))
{
    data=contatainer.data;
}