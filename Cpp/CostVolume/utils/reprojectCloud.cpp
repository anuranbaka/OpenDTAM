#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "utils/utils.h"
#include "reprojectCloud.hpp"

//debug
#include "tictoc.h"
#include "graphics.hpp"

//This reprojects a depthmap and image to another view. Pixels not predicted are set to the color of 0,0

using namespace cv;
using namespace std;

Mat reprojectCloud(const Mat _im, const Mat _depth, const Mat _oldPose, const Mat _newPose, const Mat _cameraMatrix){
    tic();
    Mat im=_im;
    Mat_<float> depth=_depth;
    Mat oldPose=make4x4(_oldPose);
    Mat newPose=make4x4(_newPose);
    Mat cameraMatrix=make4x4(_cameraMatrix);
    Mat  proj(4,4,CV_64FC1);
    Mat_<Vec3f> xyin(im.rows,im.cols);
    Mat_<Vec2f> xyout(im.rows,im.cols);

//     cout<<cameraMatrix<<endl;
//     cout<<newPose<<endl;
//     cout<<oldPose<<endl;

    proj=cameraMatrix*newPose*oldPose.inv()*cameraMatrix.inv();
    
    
//     cout<<proj*(Mat_<double>(4,1)<<5,3,1,4)<<endl;
//     cout<<(_cameraMatrix.inv()*(Mat_<double>(3,1)<<5,3,1))/.25<<endl;
//     cout<<newPose*oldPose.inv()*(Mat_<double>(4,1)<<-2.614297589359934,  -1.970833333333333,4,1)<<endl;
//     cout<<_cameraMatrix*(Mat_<double>(3,1)<<-4.386563809008884,    -4.192947124058795,4.065032570174338)/4.065032570174338<<endl;
//     cout<<proj*(Mat_<double>(4,1)<<5,3,.25,1)/  1.016258142543584    <<endl;//should match on image 2
    
    
    
    
    
    
//     cout<<"This should be affine:"<<endl;
//     cout<<proj<<endl;
    Mat tmp=proj.colRange(2,4).clone();
    tmp.col(1).copyTo(proj.col(2));
    tmp.col(0).copyTo(proj.col(3));
    proj=proj.rowRange(0,3).clone();
//      cout<<proj<<endl;
//      //Check if conversions are rounded or truncated
//     tmp=(Mat_<double>(4,1)<<5,3,.7,1);
//     tmp.convertTo(tmp,CV_32SC1);
//     cout<<tmp<<endl;

    float* pt=(float*) (xyin.data);
    float* d=(float*) (depth.data);
    for(int i=0;i<im.rows;i++){
        for(int j=0;j<im.cols;j++,pt+=3,d++){
            pt[0]=j;
            pt[1]=i;
            pt[2]=*d;
        }
    }

    perspectiveTransform(xyin,xyout,proj);
    tic();
    Mat xy;
    xyout.convertTo(xy,CV_32SC2);//rounds apparently, was unable to trace
    int* xyd=(int *)(xy.data);
    Mat_<float> xmap(im.rows,im.cols,0.0);
    Mat_<float> ymap(im.rows,im.cols,0.0);
    float* xm=(float*)(xmap.data);
    float* ym=(float*)(ymap.data);
    for(int i=0;i<im.rows;i++){
        for(int j=0;j<im.cols;j++,xyd+=2){
            if(xyd[1]<im.rows && xyd[1]>=0 && xyd[0]>0&&xyd[0]<im.cols){
                xmap(xyd[1],xyd[0])=j;
                ymap(xyd[1],xyd[0])=i;
            }
        }
    }
    toc();
    tic();
    Mat im2;
    remap( im, im2, xmap, ymap, CV_INTER_NN, BORDER_CONSTANT, Scalar(0,0, 0) );
    toc();
    //namedWindow("backtrans",CV_WINDOW_OPENGL);
    pfShow("backtrans",im2);
    tic();
    
    toc();
    //cvUpdateWindow("backtrans");
    //exit(0);
    return xyout;
    
}
