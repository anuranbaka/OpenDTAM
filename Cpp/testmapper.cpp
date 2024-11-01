#include <opencv2/core/core.hpp>
#include <iostream>
#include <stdio.h>

#include "convertAhandaPovRayToStandard.h"
#include "CostVolume/utils/reproject.hpp"
#include "CostVolume/utils/reprojectCloud.hpp"
#include "CostVolume/Cost.h"
#include "CostVolume/CostVolume.hpp"
#include "Optimizer/Optimizer.hpp"
#include "DepthmapDenoiseWeightedHuber/DepthmapDenoiseWeightedHuber.hpp"
#include "graphics.hpp"
#include "set_affinity.h"
#include "utils/utils.hpp"
#include "tictoc.h"

//A test program to make the mapper run
using namespace cv;
using namespace cv::gpu;
using namespace std;

int App_main( int argc, char** argv )
{
    const int    numImg              = 50;
    const int    imagesPerCV         = 5;
    const int    layers              = 32;
    const float  near                = 0.010f;
    const float  far                 = 0.0f;
    const double reconstructionScale = 1.0;
    const double sx                  = reconstructionScale;
    const double sy                  = reconstructionScale;

#if !defined WIN32 && !defined _WIN32 && !defined WINCE && defined __linux__ && !defined ANDROID
    pthread_setname_np(pthread_self(), "App_main");
#endif

    char filename[500];
    Mat cameraMatrix;
    Mat         image,  R , T ;
    vector<Mat> images, Rs, Ts;
    for(int i=0; i<numImg; i++){
        sprintf(filename, "../../Trajectory_30_seconds/scene_%03d.png", i);
        cout<<"Opening: "<< filename << endl;
        imread(filename, -1).convertTo(image, CV_32FC3, 1.0/65535.0);
        resize(image, image, Size(), reconstructionScale, reconstructionScale);
        images.push_back(image.clone());

        convertAhandaPovRayToStandard("../../Trajectory_30_seconds",
                                      i, cameraMatrix,
                                      R, T);
        Rs.push_back(R.clone());
        Ts.push_back(T.clone());
    }

    Mat ret; //a place to return downloaded images to
    CudaMem cret(images[0].rows, images[0].cols, CV_32FC1);
    ret=cret.createMatHeader();

    //Setup camera matrix
    cameraMatrix=cameraMatrix.mul((Mat)(Mat_<double>(3,3) <<    sx,0.0,sx,
                                                                0.0,sy ,sy,
                                                                0.0,0.0,1.0));

    CostVolume costvolume(images[0], (FrameID)0, layers, near, far, Rs[0], Ts[0], cameraMatrix);

    // Set up display windows
    ret=image*0; // hack to get a zeroed image
    pfShow("A function",       ret, 0, cv::Vec2d(0, layers));
    pfShow("D function",       ret, 0, cv::Vec2d(0, layers));
    pfShow("A function loose", ret, 0, cv::Vec2d(0, layers));
    pfShow("Predicted Image",  ret, 0, cv::Vec2d(0,1));
    pfShow("Actual Image",     ret);

    cv::gpu::Stream s;
    for(int imageNum=0; imageNum < numImg; imageNum++){
        T=Ts[imageNum];
        R=Rs[imageNum];
        image=images[imageNum];

        if(costvolume.count < imagesPerCV){
            costvolume.updateCost(image, R, T); // increments costvolume.count
        }
        else{
            //Attach optimizer
            Optimizer optimizer(costvolume);
            optimizer.initOptimization();
            s=optimizer.cvStream;
            
            Ptr<DepthmapDenoiseWeightedHuber> dp = createDepthmapDenoiseWeightedHuber(costvolume.baseImageGray,
                                                                                      costvolume.cvStream);
            DepthmapDenoiseWeightedHuber& denoiser=*dp;
            denoiser.cacheGValues();

            GpuMat a(costvolume.loInd.size(), costvolume.loInd.type());
            costvolume.cvStream.enqueueCopy(costvolume.loInd, a);
            GpuMat d;

            bool doneOptimizing; int Acount=0; int QDcount=0;
            do{
              for(int i = 0; i < 10; i++) {
                d=denoiser(a, optimizer.epsilon, optimizer.getTheta());
                QDcount++;
              }
              d.download(ret);
              pfShow("D function", ret, 0, cv::Vec2d(0, layers));
              
              doneOptimizing=optimizer.optimizeA(d,a);
              Acount++;
              a.download(ret);
              pfShow("A function", ret, 0, cv::Vec2d(0, layers));
            }while(!doneOptimizing);

            optimizer.lambda=0.01f;
            optimizer.optimizeA(d,a);
            optimizer.cvStream.waitForCompletion();
            a.download(ret);
            pfShow("A function loose", ret, 0, cv::Vec2d(0, layers));

            for(int i=0; i<numImg; i++){
                reprojectCloud(images[i], images[0],
                               optimizer.depthMap(),
                               RTToP(Rs[0], Ts[0]),
                               RTToP(Rs[i], Ts[i]),
                               cameraMatrix);
            }

            // reset costvolume
            costvolume=CostVolume(images[0], (FrameID)0, layers, near, far, Rs[0], Ts[0], cameraMatrix);
        }
        s.waitForCompletion();// so we don't lock the whole system up forever
    }
    s.waitForCompletion();
    Stream::Null().waitForCompletion();
    return 0;
}

void myExit(){
    ImplThread::stopAllThreads();
}

int main( int argc, char** argv ){
    initGui();
    int ret = App_main(argc, argv);
    myExit();
    return ret;
}
