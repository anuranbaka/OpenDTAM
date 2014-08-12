#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#ifndef DTAM_UTILS_HPP
#define DTAM_UTILS_HPP
using namespace cv;
static Mat make4x4(const Mat& mat){
    
    if (mat.rows!=4||mat.cols!=4){
        Mat tmp=Mat::eye(4,4,mat.type());
        tmp(Range(0,mat.rows),Range(0,mat.cols))=mat*1.0;

        return tmp;
    }else{
        return mat;
    }
}

static Mat rodrigues(const Mat& p){
    
    Mat tmp;
    Rodrigues(p,tmp);
    return tmp;
}

static void LieToRT(InputArray Lie, OutputArray _R, OutputArray _T){
    Mat p = Lie.getMat();
    _R.create(3,3,CV_64FC1);
    Mat R = _R.getMat();
    _T.create(3,1,CV_64FC1);
    Mat T = _T.getMat();
    if(p.cols==1){
        p = p.t();
    }
        
    rodrigues(p.colRange(Range(0,3))).copyTo(R);
    Mat(p.colRange(Range(3,6)).t()).copyTo(T);


}


static void RTToLie(InputArray _R, InputArray _T, OutputArray Lie ){

    Mat R = _R.getMat();
    Mat T = _T.getMat();
    Lie.create(1,6,T.type());
    
    Mat p = Lie.getMat(); 
    assert(p.size()==Size(6,1));
    p=p.reshape(1,6);
    if(T.rows==1){
        T = T.t();
    }
    
    rodrigues(R).copyTo(p.rowRange(Range(0,3)));
    T.copyTo(p.rowRange(Range(3,6)));
    assert(Lie.size()==Size(6,1));
}
static Mat RTToLie(InputArray _R, InputArray _T){

    Mat P;
    RTToLie(_R,_T,P);
    return P;
}
static void PToLie(InputArray _P, OutputArray Lie){

    Mat P = _P.getMat();
    assert(P.cols == P.rows && P.rows == 4);
    Mat R = P(Range(0,3),Range(0,3));
    Mat T = P(Range(0,3),Range(3,4));
    RTToLie(R,T,Lie);
    assert(Lie.size()==Size(6,1));
}
static void RTToP(InputArray _R, InputArray _T, OutputArray _P ){
    
    Mat R = _R.getMat();
    Mat T = _T.getMat();
    Mat P = _P.getMat();
    hconcat(R,T,P);
    make4x4(P).copyTo(_P);
}
static Mat RTToP(InputArray _R, InputArray _T){
    
    Mat R = _R.getMat();
    Mat T = _T.getMat();
    Mat P;
    hconcat(R,T,P);
    make4x4(P);
    return P;
}
static void LieToP(InputArray Lie, OutputArray _P){
    Mat p = Lie.getMat();
    _P.create(4,4,p.type());
    Mat P = _P.getMat();
    if(p.cols==1){
        p = p.t();
    } 
    
    Mat R=rodrigues(p.colRange(Range(0,3)));
    Mat T=p.colRange(Range(3,6)).t();
    hconcat(R,T,P);
    make4x4(P).copyTo(_P);
}
static Mat LieToP(InputArray Lie){
    Mat P;
    LieToP(Lie,P);
    return P;
}

static Mat LieSub(Mat A, Mat B){
    Mat Pa;
    Mat Pb;
    LieToP(A,Pa);
    LieToP(B,Pb);
    Mat out;
    assert(A.size()==Size(6,1) && B.size()==Size(6,1));
    PToLie(Pa*Pb.inv(),out);
    return out;
}

static Mat LieAdd(Mat A, Mat B){
    Mat Pa;
    Mat Pb;
    
    LieToP(A,Pa);
    LieToP(B,Pb);
    Mat out;
    PToLie(Pa*Pb,out);
    return out;
}

#endif 