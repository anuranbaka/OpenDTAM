#include <opencv2/core.hpp>
#include <iostream>
#include <stdio.h>



//Mine
#include "convertAhandaPovRayToStandard.h"
#include "CostVolume/utils/reproject.hpp"
#include "CostVolume/utils/reprojectCloud.hpp"
#include "CostVolume/Cost.h"
#include "CostVolume/CostVolume.hpp"
#include "DepthmapDenoiseWeightedHuber/DepthmapDenoiseWeightedHuber.hpp"
#include "Optimizer/Optimizer.hpp"
#include "graphics.hpp"
#include "set_affinity.h"

#include "utils/utils.hpp"


//debug
#include "tictoc.h"







//A test program to make the mapper run
using namespace cv;
using namespace cv::cuda;
using namespace std;

int App_main( int argc, char** argv );

void myExit(){
    ImplThread::stopAllThreads();
}
int main( int argc, char** argv ){

    initGui();

    int ret=App_main(argc, argv);
    myExit();
    return ret;
}


int App_main( int argc, char** argv )
{
    int numImg=50;

#if !defined WIN32 && !defined _WIN32 && !defined WINCE && defined __linux__ && !defined ANDROID
    pthread_setname_np(pthread_self(),"App_main");
#endif

    char filename[500];
    Mat image, cameraMatrix, R, T;
    vector<Mat> images,Rs,Ts;
    
    Mat ret;//a place to return downloaded images to

    
    double reconstructionScale=5/5.;

    for(int i=0;i<numImg;i++){
        Mat tmp;
        sprintf(filename,"../../Trajectory_30_seconds/scene_%03d.png",i);
        convertAhandaPovRayToStandard("../../Trajectory_30_seconds",
                                      i,
                                      cameraMatrix,
                                      R,
                                      T);
        Mat image;
        cout<<"Opening: "<< filename << endl;
        
        imread(filename, -1).convertTo(image,CV_32FC3,1.0/65535.0);
        resize(image,image,Size(),reconstructionScale,reconstructionScale);
        
        images.push_back(image.clone());
        Rs.push_back(R.clone());
        Ts.push_back(T.clone());

    }
    cv::cuda::CudaMem cret(images[0].rows,images[0].cols,CV_32FC1);
    ret=cret.createMatHeader();
    //Setup camera matrix
    double sx=reconstructionScale;
    double sy=reconstructionScale;
    cameraMatrix+=(Mat)(Mat_<double>(3,3) <<    0.0,0.0,0.5,
                                                0.0,0.0,0.5,
                                                0.0,0.0,0.0);
    cameraMatrix=cameraMatrix.mul((Mat)(Mat_<double>(3,3) <<    sx,0.0,sx,
                                                                0.0,sy ,sy,
                                                                0.0,0.0,1.0));
    cameraMatrix-=(Mat)(Mat_<double>(3,3) <<    0.0,0.0,0.5,
                                                0.0,0.0,0.5,
                                                0.0,0.0,0);
    int layers=32;
    int imagesPerCV=2;
    CostVolume cv(images[0],(FrameID)0,layers,0.015,0.0,Rs[0],Ts[0],cameraMatrix);;
    
    

    int imageNum=0;
    cv::cuda::Stream s;
    for (int imageNum=0;imageNum<numImg;imageNum++){//cycle through images forever
        T=Ts[imageNum];
        R=Rs[imageNum];
        image=images[imageNum];
          
        if(cv.count<imagesPerCV){
            cv.updateCost(image, R, T);
        }
        else{
            //Attach optimizer
            DepthmapDenoiseWeightedHuber denoiser(cv.rows,cv.cols,cv.baseImageGray,cv.cvStream);
            Optimizer optimizer(cv);
            optimizer.initOptimization();
            GpuMat a=cv.loInd.clone();
            GpuMat d;
            denoiser.cacheGValues();
            
            denoiser._gx.download(ret,optimizer.cvStream);
            pfShow("G function:x direction", ret, 0, cv::Vec2d(0, 1));
            denoiser._gy.download(ret,optimizer.cvStream);
            pfShow("G function:y direction", ret, 0, cv::Vec2d(0, 1));
            optimizer._a.download(ret);
                pfShow("A", ret, 0, cv::Vec2d(0, layers));
                waitKey(0);
                gpause();
            
            
            
            bool doneOptimizing; int Acount=0; int QDcount=0;
            do{
//                 cout<<"Theta: "<< optimizer.getTheta()<<endl;
// 
                optimizer._a.download(ret);
                pfShow("A", ret, 0, cv::Vec2d(0, layers));
                
//                 optimizer.epsilon*=optimizer.thetaStep;

                for (int i = 0; i < 10; i++) {
                    optimizer.stableDepth.download(ret);
                    d=denoiser(a,optimizer.epsilon,optimizer.getTheta());
                    optimizer.stableDepth.download(ret);
                    QDcount++;
                    
                    denoiser._qx.download(ret);
                    pfShow("Q function:x direction", ret, 0, cv::Vec2d(-1, 1));
                    denoiser._qy.download(ret);
                    pfShow("Q function:y direction", ret, 0, cv::Vec2d(-1, 1));
                    denoiser._d.download(ret);
                    pfShow("D function", ret, 0, cv::Vec2d(0, layers));
                }
                doneOptimizing=optimizer.optimizeA(d,a);
                Acount++;
            }while(!doneOptimizing);
//             cout<<"A iterations: "<< Acount<< "  QD iterations: "<<QDcount<<endl;
//             pfShow("Depth Solution", optimizer.depthMap(), 0, cv::Vec2d(cv.far, cv.near));
//             gpause();
            cv=CostVolume(images[imageNum],(FrameID)0,layers,0.010,0.0,Rs[imageNum],Ts[imageNum],cameraMatrix);
            s=optimizer.cvStream;
        }
        
    }
    s.waitForCompletion();
   
    return 0;
}


