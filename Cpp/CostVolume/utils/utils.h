#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
Mat make4x4(const Mat& mat){
    
    if (mat.rows!=4||mat.cols!=4){
        Mat tmp=Mat::eye(4,4,mat.type());
        tmp(Range(0,mat.rows),Range(0,mat.cols))=mat*1.0;

        return tmp;
    }else{
        return mat;
    }
}