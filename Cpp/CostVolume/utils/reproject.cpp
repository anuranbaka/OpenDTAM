#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "reproject.hpp"
#include "tictoc.h"

using namespace cv;
using namespace std;

void reproject( const Mat& src,
                const Mat& cameraMatrix,
                const Mat& cameraAffinePoseBaseIn,
                const Mat& cameraAffinePoseAlternateIn,
                float invDepth,
                Mat& dst,
                Mat& mask){
    Mat cameraAffinePoseBase=cameraAffinePoseBaseIn.clone();
    Mat cameraAffinePoseAlternate=cameraAffinePoseAlternateIn.clone();
    assert(cameraMatrix.rows==3 && cameraMatrix.cols==3);
    if (cameraAffinePoseBase.rows==3)
        vconcat(cameraAffinePoseBase,Mat()=(Mat_<double>(1,4) << 0,0,0,1),cameraAffinePoseBase);
    if (cameraAffinePoseAlternate.rows==3)
        vconcat(cameraAffinePoseAlternate,Mat()=(Mat_<double>(1,4) << 0,0,0,1),cameraAffinePoseAlternate);


    Mat_<double> cA_Apln;
    Mat_<double> bottom=(Mat_<double>(1,3) << 0,0,invDepth);
    vconcat(Mat(cameraMatrix.inv()),bottom,cA_Apln);

    //cout<<"cA_Apln"<<cA_Apln<<endl;
    const Mat_<double> cW_A(cameraAffinePoseAlternate.inv());//optimizable
    const Mat_<double>  cB_W(cameraAffinePoseBase);
    Mat_<double>  cBpln_B;
    hconcat(cameraMatrix,Mat()=(Mat_<double>(3,1) << 0,0,0), cBpln_B);
    Mat_<double> cBpln_Apln=cBpln_B*cB_W*cW_A*cA_Apln;

//    {//debug
//        cout<<"depth"<<invDepth<<endl;
//        cout<<"warpmat: "<<cBpln_Apln<<endl;
//    }

    warpPerspective(src, dst, cBpln_Apln, src.size() ,INTER_NEAREST,BORDER_CONSTANT,Scalar(-1.0e100,-1.0e100,-1.0e100));//.013 s for nearest, .021 for linear


    mask = (dst >0);//0.00247 s 
}

/*
template <class T>
void reproject( const Mat_<T>& src,
                const Matx33d& cameraMatrix,
                const Matx44d& cameraAffinePoseBase,
                const Matx44d& cameraAffinePoseAlternate,
                float invDepth,
                Mat_<T>& dst,
                Mat_<uchar>& mask){

    Mat_<double> cA_Apln(Mat_<double>(3,4) << Mat_<double>(cameraMatrix.inv()), 0,0,invDepth);
    cA_Apln=cA_Apln.t();
    //cout<<"cA_Apln"<<cA_Apln<<endl;
    Mat_<double> cW_A=cameraAffinePoseAlternate.inv();//optimizable
    Mat_<double> & cB_W=cameraAffinePoseBase;
    Mat_<double> & cBpln_B;
    hconcat(cameraMatrix,Mat(Mat_<double>(3,1) << 0,0,0), cBpln_B);
    Mat_<double> cBpln_Apln=cBpln_B*cB_W*cW_A*cA_Apln;

    warpPerspective(src, dst, cBpln_Apln, src.size(),INTER_LINEAR,BORDER_CONSTANT,Scalar(-1.0) );
    mask = (dst >= 0);
}*/
