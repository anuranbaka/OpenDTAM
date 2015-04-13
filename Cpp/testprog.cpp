#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>



//Mine
#include "convertAhandaPovRayToStandard.h"
#include "CostVolume/utils/reproject.hpp"
#include "CostVolume/utils/reprojectCloud.hpp"
#include "CostVolume/Cost.h"
#include "CostVolume/CostVolume.hpp"
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
    int numImg=500;

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
    CostVolume cv(images[0],(FrameID)0,layers,0.010,0.0,Rs[0],Ts[0],cameraMatrix);;
    
    
    int imageNum=0;
    cv::cuda::Stream s;
    for (int imageNum=0;imageNum<numImg;imageNum++){
        T=Ts[imageNum];
        R=Rs[imageNum];
        image=images[imageNum];
          
        if(cv.count<imagesPerCV){
            cv.updateCost(image, R, T);
        }
        else{
            //Attach optimizer
            Ptr<DepthmapDenoiseWeightedHuber> dp = createDepthmapDenoiseWeightedHuber(cv.baseImageGray,cv.cvStream);
            DepthmapDenoiseWeightedHuber& denoiser=*dp;
            Optimizer optimizer(cv);
            optimizer.initOptimization();
            GpuMat a(cv.loInd.size(),cv.loInd.type());
            cv.loInd.copyTo(a,cv.cvStream);
            GpuMat d;
            denoiser.cacheGValues();
            ret=image*0;
            pfShow("A function", ret, 0, cv::Vec2d(0, layers));
            pfShow("D function", ret, 0, cv::Vec2d(0, layers));
            pfShow("A function loose", ret, 0, cv::Vec2d(0, layers));
            pfShow("Predicted Image",ret,0,Vec2d(0,1));
            pfShow("Actual Image",ret);
//                pfShow("A", ret, 0, cv::Vec2d(0, layers));
//                waitKey(0);
//                gpause();
            
             gpause();
            bool doneOptimizing; int Acount=0; int QDcount=0;
            do{
//                 cout<<"Theta: "<< optimizer.getTheta()<<endl;
//
                if(Acount==0)
                    gpause();
               a.download(ret);
               pfShow("A function", ret, 0, cv::Vec2d(0, layers));

//                 optimizer.epsilon*=optimizer.thetaStep;

                for (int i = 0; i < 10; i++) {
                    d=denoiser(a,optimizer.epsilon,optimizer.getTheta());
                    QDcount++;

                   d.download(ret);
                   pfShow("D function", ret, 0, cv::Vec2d(0, layers));
                }
                doneOptimizing=optimizer.optimizeA(d,a);
                Acount++;
            }while(!doneOptimizing);
            optimizer.lambda=.01;
            optimizer.optimizeA(d,a);
            optimizer.cvStream.waitForCompletion();
            a.download(ret);
               pfShow("A function loose", ret, 0, cv::Vec2d(0, layers));
               gpause();
//             cout<<"A iterations: "<< Acount<< "  QD iterations: "<<QDcount<<endl;
//             pfShow("Depth Solution", optimizer.depthMap(), 0, cv::Vec2d(cv.far, cv.near));
            imageNum=0;
            cv=CostVolume(images[imageNum],(FrameID)0,layers,0.010,0.0,Rs[imageNum],Ts[imageNum],cameraMatrix);
            s=optimizer.cvStream;
            for (int imageNum=0;imageNum<numImg;imageNum=(imageNum+1)%numImg){
                reprojectCloud(images[imageNum],images[0],optimizer.depthMap(),RTToP(Rs[0],Ts[0]),RTToP(Rs[imageNum],Ts[imageNum]),cameraMatrix);
            }
        }
        
    }
    s.waitForCompletion();
    Stream::Null().waitForCompletion();
    return 0;
}


