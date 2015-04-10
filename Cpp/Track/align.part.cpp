/*
 *  Track_align.cpp
 *  
 *
 *  Created by Paul Foster on 6/4/14.
 *  
 *
 */


#include "Track.hpp"
#include "Align_part.cpp"
#include "tictoc.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
//needs: 
//Mat base
//Mat cameraMatrix
//Mat depth
//Cost cv
//cols

//Models Used:
//
// Warp is function of p, the parameters
// Model 1: Template(:)=Image(Warp(:)+dWarp(:))
// Model 2: Template(dWarp_inv(:))=Image(Warp(:))
//
// nb: the "+" means composition, i.e. Warp(:)+dWarp(:)=dWarp(Warp)
//
// 
// J1=dI/dWarp*dWarp/dp=grad(I)(Warp)*dWarp/dp
// J*dp=T-I
// (J'J)*dp=J1'I
// dp=(J'J)^-1*J'I     //A O(n*N) operation if J cacheable, else O(n^2*N) operation
//
// The first model is more correct, since Image is smooth 
// but Template is not smooth.
// However, the second allows caching the Jacobians and should 
// work if the depth is mostly smooth.
// Both models are averaged for ESM, giving a second order method
// but at the cost of being neither correct nor cachable.
// However, ESM is good for initial alignment because no depth is 
// used, so the presumed map is smooth both ways. Might go faster 
// to just cache anyway though and do more non-ESM steps in same
// amount of time.
//
// The paper is clear that it uses ESM for the initial levels of 
// the pyramid, and implies that it uses Model 1 for the full
// estimation. TODO:I would like to allow either choice to be made.
//
using namespace cv;
using namespace std;
#define LEVELS_2D 2

void createPyramid(const Mat& image,vector<Mat>& pyramid,int& levels){
    
    Mat in=image;
    if(levels==0){//auto size to end at >=15px tall (use height because shortest dim usually)
        for (float scale=1.0; scale>=15.0/image.rows; scale/=2, levels++);
    }
    assert(levels>0);
    int l2=levels-1;
    pyramid.resize(levels);
    pyramid[l2--]=in;
    
    for (float scale=0.5; l2>=0; scale/=2, l2--) {
        Mat out;

        
        resize(in,out,Size(),.5,.5,CV_INTER_AREA);
        pyramid[l2]=out;
        in=out;
    }
    
}

static void createPyramids(const Mat& base,
                           const Mat& depth,
                           const Mat& input,
                           const Mat& cameraMatrixIn,
                           vector<Mat>& basePyr,
                           vector<Mat>& depthPyr,
                           vector<Mat>& inPyr,
                           vector<Mat>& cameraMatrixPyr,
                           int& levels
){
    createPyramid(base,basePyr,levels);
    createPyramid(depth,depthPyr,levels);
    createPyramid(input,inPyr,levels);
    int l2=0;
    cameraMatrixPyr.resize(levels);
    // Figure out camera matrices for each level
    for (double scale=1.0,l2=levels-1; l2>=0; scale/=2, l2--) {
        Mat cameraMatrix=make4x4(cameraMatrixIn.clone());
        cameraMatrix(Range(0,2),Range(2,3))+=.5;
        cameraMatrix(Range(0,2),Range(0,3))*= scale;
        cameraMatrix(Range(0,2),Range(2,3))-=.5;
        cameraMatrixPyr[l2]=cameraMatrix;
    }
    
}
void Track::align(){
    align_gray(baseImage, depth, thisFrame);
};

void Track::align_gray(Mat& _base, Mat& depth, Mat& _input){
    Mat input,base,lastFrameGray;
    input=makeGray(_input);
    base=makeGray(_base);
    lastFrameGray=makeGray(lastFrame)  ;
    
    tic();
    int levels=6; // 6 levels on a 640x480 image is 20x15
    int startlevel=0;
    int endlevel=6;

    Mat p=LieSub(pose,basePose);// the Lie parameters 
    cout<<"pose: "<<p<<endl;

    vector<Mat> basePyr,depthPyr,inPyr,cameraMatrixPyr;
    createPyramids(base,depth,input,cameraMatrix,basePyr,depthPyr,inPyr,cameraMatrixPyr,levels);
    
    vector<Mat> lfPyr;
    createPyramid(lastFrameGray,lfPyr,levels);
    
    


    
    int level=startlevel;
    Mat p2d=Mat::zeros(1,6,CV_64FC1);
    for (; level<LEVELS_2D; level++){
        int iters=1;
        for(int i=0;i<iters;i++){
            //HACK: use 3d alignment with depth disabled for 2D. ESM would be much better, but I'm lazy right now.
            align_level_largedef_gray_forward(  lfPyr[level],//Total Mem cost ~185 load/stores of image
                                                depthPyr[level]*0.0,
                                                inPyr[level],
                                                cameraMatrixPyr[level],//Mat_<double>
                                                p2d,                //Mat_<double>
                                                CV_DTAM_FWD,
                                                1,
                                                3);
//             if(tocq()>.01)
//                 break;
        }
    }
    p=LieAdd(p2d,p);
//     cout<<"3D iteration:"<<endl;
    for (level=startlevel; level<levels && level<endlevel; level++){
        int iters=1;
        for(int i=0;i<iters;i++){
            float thr = (levels-level)>=2 ? .05 : .2; //more stringent matching on last two levels 
            bool improved;
            improved = align_level_largedef_gray_forward(   basePyr[level],//Total Mem cost ~185 load/stores of image
                                                            depthPyr[level],
                                                            inPyr[level],
                                                            cameraMatrixPyr[level],//Mat_<double>
                                                            p,                //Mat_<double>
                                                            CV_DTAM_FWD,
                                                            thr,
                                                            6);
            
//             if(tocq()>.5){
//                 cout<<"completed up to level: "<<level-startlevel+1<<"   iter: "<<i+1<<endl;
//                 goto loopend;//olny sactioned use of goto, the double break
//             }
//             if(!improved){
//                 break;
//             }
        }
    }
    loopend:
    
    pose=LieAdd(p,basePose);
    static int runs=0;
    //assert(runs++<2);
    toc();
    
}

// See reprojectCloud.cpp for explanation of the form 
// of the camera matrix for inverse depth reprojection.
//
// From that result, the form of the camera matrix for a 
// scaled camera is trivial:
//
// Camera Matrix scaling:
//
// [ f*s    0    (cx-.5)*s+.5    ] [  xc ] [ xp   ]
// [ 0      f*s  (cy-.5)*s+.5    ]*[  yc ]=[ yp   ]
// [ 0      0         1         0] [  wc ] [ 1   ]
// [ 0      0         0         1] [  zc ] [ 1/zc ]
// 
// 
// The equations: 
// All solvers fundamentally solve:
// J*dp=T-I
// by doing:
// (J'*J)^-1*J'*(T-I)
// The problem is we don't want to use pixels
// corresponding to occluded regions

// Track::cacheBaseDerivatives(){
//     Scharr( src_gray, g_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
//     Scharr( src_gray, g_y, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
// }




void Track::ESM(){

//     
//     //Build map differentials:
//     {
//         //build identity map
//         Mat_<Vec3f> xyin(im.rows,im.cols);
//         float* pt=(float*) (xyin.data);
//         float* d=(float*) (depth.data);
//         for(int i=0;i<im.rows;i++){
//             for(int j=0;j<im.cols;j++,pt+=3,d++){
//                 pt[0]=j;
//                 pt[1]=i;
//                 pt[2]=*d;
//             }
//         }
//         for (i=0; i<numDF; i++) {
//             //build slightly perturbed matrix
//             double small=1e-6;
//             Mat_<double> ptemp=p.clone();
//             ptemp(0,i)+=small;
//             Mat dR=rodrigues(ptemp(0,Range(0,3)));
//             Mat dT=ptemp(0,Range(3,6));
//             Mat dA;
//             hconcat(dR,dT,dA);
//             dA=make4x4(dA);
//             
//             float* pt=(float*) (xyin.data);
//             float* d=(float*) (depth.data);
//             for(int i=0;i<im.rows;i++){
//                 for(int j=0;j<im.cols;j++,pt+=3,d++){
//                     pt[0]=j;
//                     pt[1]=i;
//                     pt[2]=*d;
//                 }
//             } 
//         }
//     }
//     
//     for (int i=0;i<iters;i++) {
//         
//         p=ESMStep();
//     }
// 
    
    
    
}



// vector<double> Track::PJCRStep(const Mat& base,
//                                const Mat& depth,
//                                const Mat& input,
//                                const Mat& cameraMatrix,
//                                const vector<double> p,
//                                const vector<double>& maxStep=vector<double>()
//                                ){
//     
//     
// }
// 
// vector<double> Track::PJCFStep(const Mat& base,
//                               const Mat& depth,
//                               const Mat& input,
//                               const Mat& cameraMatrix,
//                               const vector<double>& p,
//                               const vector<double>& maxStep=vector<double>()
//                               ){
//     
//     
// }


void Track::cacheDerivatives(){
//     int r=rows;
//     int c=cols;
//     //Build the in map 
//     Mat_<Vec3f> idMap3;
//     {
//         idMap3.create(r,c);//[rows][cols][3]
//         float* id3=(float*) (idMap3.data);
//         float* dp=(float*) (depth.data);
//         int offset=0;
//         for(int i=0;i<r;i++){
//             for(int j=0;j<c;j++,offset++){
//                 id3[offset*3+0]=j;
//                 id3[offset*3+1]=i;
//                 id3[offset*3+2]=dp[offset];
//             }
//         }
//     }
//     
//     //Build the unincremented transform: (Mem cost 2 layer store,3 load :5)
//     Mat baseMap(rows,cols,CV_32FC2);
//     {
//         Mat_<double> p = Mat::zeros(1,6,CV_64FC1);
//         Mat baseProj=paramsToProjection(p,cameraMatrix);
//         Mat baseMap(r,c,CV_32FC2);
//         perspectiveTransform(idMap3,baseMap,baseProj);
//     }
//     
//     int numParams = 3;
//     for (int paramNum=0; paramNum<numParams; paramNum++) {
// 
//         //Build the incremented transform
//         Mat_<double> p = Mat::zeros(1,6,CV_64FC1);
//         p(0,paramNum)+=small;
//         Mat proj=paramsToProjection(p,cameraMatrix);
//         Mat tmp; hconcat(proj.colRange(Range(0,2)) ,proj.colRange(Range(3,4)) , tmp);//convert to 2D since only doing that for ESM
//         proj=tmp;
//         
//         //get a row of dmap/dp
//         Mat outmap=dMdp.row(paramNum);
//         
//         perspectiveTransform(idMap3,outmap,proj);//outmap=baseMap+dMap/dp*small (Mem cost 5)
//         
//         
//         //subtract off the base to make a differential  (Mem cost 6)
//         //this cannot be done below in the J loop because it would need 5 pointers
//         // which is bad for cache (4-way set associative)
//         outmap-=baseMap;//outmap=dMap/dp*small
//     }
//     //Cache the Template gradient
//     getGradientInterleave(baseImage,gradBase);
//     
//     
}

#define BSZ 16 //This should be adjusted to fit the cache size

static inline void JacobianCore(Mat& dMdp,
                                Mat& G,
                                Mat& J,
                                Mat& H,
                                int numParams)
{
    //tmp=M*G;
    //J = sumchannels(tmp);
    {
        float* tmp=(float *)malloc(sizeof(float)*BSZ*2*numParams);
        float* tp=tmp;
        const float* gp=G.ptr<float>(0);
        for(int pn=0;pn<numParams;pn++){
            const float* Mp=dMdp.ptr<float>(pn);//get a pointer to the row of the dMdp matrix
            float* Jp=J.ptr<float>(pn);
            for(int c=0;c<BSZ*2;c++){//multiply 
                tp[c]=Mp[2*c]*gp[2*c]+Mp[2*c+1]*gp[2*c+1];
            }
            for(int c=0;c<BSZ;c++){//multiply 
                Jp[c]=tp[2*c+0]+tp[2*c+1];
            }
            tp+=sizeof(float)*BSZ*2;
        }
        free(tmp);
    }
    
    H+=J*J.t();//add to the hessian accumulator TODO: this might need to be increased to hold things as doubles
}

static inline void solveJacobian(Mat& dMdp,
                          Mat& G,
                          Mat& J,
                          Mat& err,
                          Mat& p,
                          int numParams)
{
    int c=dMdp.cols;
    int nb=c/BSZ;
    Mat H(numParams,numParams,CV_64FC1);
    H=0.0;
    for (int i=0;i<nb;i++){
        int offset = i*BSZ;
        Mat _dMdp=dMdp.colRange(Range(offset,offset+BSZ));
        Mat _G=G.colRange(Range(offset,offset+BSZ));
        Mat _J=J.colRange(Range(offset,offset+BSZ));
        JacobianCore(_dMdp, _G, _J, H, numParams);
    }
    cout<<"Summed Hessian: "<<H<<endl;
    cout<<"Recalculated Hessian: "<<J*J.t()<<endl;
    // Now J has been filled out and H is complete
    p+=H.inv()*(J*err);
}







    

void ESMStep(const Mat& gradMTI,//2ch * N cols
             const Mat& dMdp,//2ch * 3 rows * N cols
             double* p,
             double* maxStep//maximum rotation
             )
{
    
    
}

// void getJacobianPiece(Mat idMap, Mat xyin, Mat p, Mat Jpiece){
//     Mat dR=rodrigues(p(0,Range(0,3)));
//     Mat dT=p(0,Range(3,6));
//     Mat dA;
//     hconcat(dR,dT,dA);
//     dA=make4x4(dA);
//     
//     Mat proj=cameraMatrix*dA*cameraMatrix.inv();
//     //The column swap
//     Mat tmp=proj.colRange(2,4).clone();
//     tmp.col(1).copyTo(proj.col(2));
//     tmp.col(0).copyTo(proj.col(3));
//     //The row drop
//     proj=proj.rowRange(0,3).clone();
//     
//     
//     Mat_<Vec2f> xyout(rows,cols);
//     perspectiveTransform(xyin,xyout,proj);//xyout is a two channel x,y map
//     
//     // convert to single channel stacked array type map
//     
//     
// }
// //Returns (J'*J)^-1
// void getHessian(Mat J,Mat H){
//     H=(J.t()*J).inv();
// }


// void calculateJacobianAtScale(){
//     rows=rows0*s;
//     cols=cols0*s;
//     I=scale(I0,s);
//     T=scale(T0,s);
//     d=scale(d0,s);
//     cameraMatrix=scaleCameraMatrix(cameraMatrix0,s);
//     
//     
//     // build the identitiy map
//     Mat_<Vec3f> idMap3(rows,cols);
//     Mat_<Vec2f> idMap2(rows,cols);
//     float* id3=(float*) (idMap3.data);
//     float* id2=(float*) (idMap2.data);
//     float* dp=(float*) (d.data);
//     int offset=0;
//     for(int i=0;i<rows;i++){
//         for(int j=0;j<cols;j++,offset++){
//             id3[offset*3+0]=j;
//             id3[offset*3+1]=i;
//             id3[offset*3+2]=dp[offset];
//             id2[offset*2+0]=j;
//             id2[offset*2+1]=i;
//         }
//     }
//     
//     Mat_<Vec2f> outmap(rows,cols);
//     vector<Mat> dMapdp[numParams]
//     for (int i=0; i<numParams; i++) {
//         //Build the incremented transform
//         Mat_<double> p_plus_dp=p.clone();
//         p_plus_dp(0,i)+=small;
//         Mat dR=rodrigues(p(0,Range(0,3)));
//         Mat dT=p(0,Range(3,6));
//         Mat dA;
//         hconcat(dR,dT,dA);
//         dA=make4x4(dA);
//         proj=cameraMatrix*dA*cameraMatrix.inv();
//         
//         perspectiveTransform(id3,outmap,proj);
//         dMapdp[i]=outmap.clone();
//         
//         
//     }
// 
//     
//     Mat dR=rodrigues(p(0,Range(0,3)));
//     Mat dT=p(0,Range(3,6));
//     Mat dA;
//     hconcat(dR,dT,dA);
//     dA=make4x4(dA);
//     
//     
//     
// }































                              
                                
