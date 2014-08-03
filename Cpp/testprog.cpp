#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <unistd.h>


//Mine
#include "convertAhandaPovRayToStandard.h"
#include "CostVolume/utils/reproject.hpp"
#include "CostVolume/utils/reprojectCloud.hpp"
#include "CostVolume/Cost.h"
#include "CostVolume/CostVolume.hpp"
#include "Optimizer/Optimizer.hpp"
#include "graphics.hpp"
#include "set_affinity.h"
#include "Track/Track.hpp"
#include "utils/utils.hpp"


//debug
#include "tictoc.h"
const static bool valgrind=0;
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

//A test program to make the mapper run
using namespace cv;
using namespace std;

int App_main( int argc, char** argv );


int main( int argc, char** argv ){

//    set_affinity(1);
//    //cvStartLoop(&App_main,argc, argv);//will crash if used with opengl!

    initGui();
    return App_main(argc, argv);
}
void myExit(){
    cudaDeviceSynchronize();

    allDie=1;
    gcheck();
    usleep(100000);
    exit(0);
}

int App_main( int argc, char** argv )
{
    cv::gpu::CudaMem imageContainer;
    pthread_setname_np(pthread_self(),"App_main");
        
    FileStorage fs;
    void* junk;
    cudaMalloc(&junk,5500);
    cudaFree(junk);

    Mat cameraAffinePoseBase;
    int imageNum=0;
    char filename[500];
    Mat cameraMatrix,R,T;
    convertAhandaPovRayToStandard("/local_store/Dropbox/Research/DTAM GSoC/OpenDTAM/Trajectory_30_seconds",
                                  imageNum,
                                  cameraMatrix,
                                  R,
                                  T);


//     cout<<"cameraMatrix: "<<cameraMatrix<<"\n";
//     cout<< "R : "<<R<<"\n";
//     cout<< "T : "<<T<<"\n";
    sprintf(filename,"/local_store/Dropbox/Research/DTAM GSoC/OpenDTAM/Trajectory_30_seconds/scene_%03d.png",imageNum);
    Mat image;
    
    double sc=5/5.;
    if (!valgrind){
        imread(filename,-1).convertTo(image,CV_32FC3,1.0/65535.0);   // Read the file

        cameraMatrix=(Mat_<double>(3,3) << 481.20*sc,0.0,320*sc-.5,
                          0.0,480.0*sc,240*sc-.5,
                          0.0,0.0,1.0);
        resize(image,image,Size(),sc,sc);
//            cout<<"cameraMatrix: "<<cameraMatrix<<"\n";
    }else{
        image.create(480,640,CV_32FC3);
        image=0.5;
    }
    
    hconcat(R,T,cameraAffinePoseBase);

    Cost cost(image.clone(),32, cameraMatrix, R,T);

    Mat tmp;
    //image.convertTo(tmp,CV_8UC4, 255.0);
    CostVolume cv(image,(FrameID)1,32,0.015,0.0,R,T,cameraMatrix);
    Optimizer optimizer(cv);
    optimizer.initOptimization();


    //Track tracker(cost);
    Track tracker(cv,optimizer);
    //assert(cost.rows==480);

    vector<Mat> images,Rs,Ts;
    for(int i=0;i<=50;i++){
        Mat tmp;
        sprintf(filename,"/local_store/Dropbox/Research/DTAM GSoC/OpenDTAM/Trajectory_30_seconds/scene_%03d.png",i);
        convertAhandaPovRayToStandard("/local_store/Dropbox/Research/DTAM GSoC/OpenDTAM/Trajectory_30_seconds",
                                      i,
                                      tmp,
                                      R,
                                      T);
        Mat image;
        cout<<filename<<endl;

        if (!valgrind){
            imread(filename, -1).convertTo(image,CV_32FC3,1.0/65535.0);
//            cameraMatrix.rowRange(0,2)/=3;
            resize(image,image,Size(),sc,sc);
        }else{
            image.create(480,640,CV_32FC3);
            image=0.5;
        }
        images.push_back(image.clone());
        Rs.push_back(R.clone());
        Ts.push_back(T.clone());

    }

    cudaProfilerStart();
    while(1){
        for (int imageNum=1;imageNum<=50;imageNum++){
            if(imageNum==2){
                cudaDeviceSynchronize();
            //    cudaProfilerStart();
            }
            char filename[500];
            Mat R,T;
//             convertAhandaPovRayToStandard("/local_store/Dropbox/Research/DTAM GSoC/OpenDTAM/Trajectory_30_seconds",
//                                     imageNum,
//                                     cameraMatrix,
//                                     R,
//                                     T);
// 
//     //         cout<<"cameraMatrix: "<<cameraMatrix<<"\n";
//     //         cout<< "R : "<<R<<"\n";
//     //         cout<< "T : "<<T<<"\n";
//             sprintf(filename,"/local_store/Dropbox/Research/DTAM GSoC/OpenDTAM/Trajectory_30_seconds/scene_%03d.png",imageNum);
//             Mat image;
//             imread(filename, -1).convertTo(image,CV_32FC3,1.0/65535.0);   // Read the file
            T=Ts[imageNum];
            R=Rs[imageNum];
            image=images[imageNum];

            //cost.updateCostL1(image,R,T);//dbg
            Mat cameraAffinePoseAlternate,mask;
            hconcat(R,T,cameraAffinePoseAlternate);
            imageContainer.create(image.rows,image.cols,CV_8UC4);
            Mat tmp,ret;
            cvtColor(image,tmp,CV_RGB2RGBA);
            Mat imageContainerRef=imageContainer;//Required by ambiguous conversion rules
            tmp.convertTo(imageContainerRef,CV_8UC4,255.0);
            if (imageNum<30){
                cv.updateCost(imageContainer, R, T);
//                cv.loInd.download(ret);
//                assert(cv.loInd.isContinuous());
//                pfShow("loInd Soln", cv.downloadOldStyle(0));
//                gpause();
            }
            cudaDeviceSynchronize();

            if (imageNum==30){
                optimizer.initOptimization();
                bool doneOptimizing;
                do{
                    cout<<"Theta: "<< optimizer.theta<<endl;
                   optimizer._a.download(ret);
                   pfShow("A", ret, 0, cv::Vec2d(0, 32));

    //                optimizer.cacheGValues();
    //                optimizer._gy.download(ret);
    //                pfShow("G function", ret, 0, cv::Vec2d(0, 1));
    //                gpause();
                    for (int i = 0; i < 10; i++) {
                        optimizer.optimizeQD();
//                        cudaDeviceSynchronize();
                       optimizer._qx.download(ret);
                       pfShow("Qx function", ret, 0, cv::Vec2d(-1, 1));
                       optimizer._gy.download(ret);
                       pfShow("Gy function", ret, 0, cv::Vec2d(0, 1));
                       optimizer._d.download(ret);
                       pfShow("D function", ret, 0, cv::Vec2d(0, 32));
//                        gpause();
                        
                    }
//                    cudaDeviceSynchronize();
                    doneOptimizing=optimizer.optimizeA();
                }while(!doneOptimizing);
                cudaDeviceSynchronize();
                optimizer._d.download(ret);
                
                pfShow("Depth Solution", ret, 0, cv::Vec2d(0, 32));
                gpause();
               // myExit();
            }

            cv.loInd.download(ret);
           // pfShow("Initial Min Soln",ret,0,cv::Vec2d(0,32));

            //gpause();
//
//            if (cost.imageNum<3){
//                tic();
//                cost.updateCostL1(image,R,T);
//                toc();
//            }
//            if (cost.imageNum==2){
//                cost.initOptimization();
////                 cost.optimize();//Launches the optimizer threads
////                 while(cost.running){usleep(1000);};
//            }
//
//
//            const Mat thisPose(cost.convertPose(R,T));
//
//             reprojectCloud(image,cost.baseImage, cost._d*cost.depthStep, Mat(cost.pose), thisPose, Mat(cost.cameraMatrix));


            if(imageNum==1){
                tracker.pose=tracker.basePose.clone();
            }
            //Test out the Tracker
            {
                Mat tp;
                RTToLie(R,T,tp);
                //tracker.pose=tp.clone();//Give the answer
                tracker.depth=optimizer.depthMap();

                tracker.addFrame(image);

                tracker.align();
                Mat p=tracker.pose;
                cout << "True Pose: "<< tp << endl;
                cout << "True Delta: "<< LieSub(tp,tracker.basePose) << endl;
                cout << "Recovered Pose: "<< p << endl;
                cout << "Recovered Delta: "<< LieSub(p,tracker.basePose) << endl;
                cout << "Pose Error: "<< p-tp << endl;

                reprojectCloud(image,cost.baseImage, tracker.depth, Mat(cost.pose) , LieToP(tracker.pose), Mat(cost.cameraMatrix));

//                 Mat R2,T2;
//                 LieToRT(p, R2, T2);
//
//                 if (imageNum==2&&cost2.imageNum==0){
//                     cost2=Cost(image.clone(),32, cameraMatrix, R2,T2);
//                     gpause();
//
//                 }
//                 if (cost2.imageNum<2){
//                     cost2.updateCostL1(image,R2,T2);
//                 }
//                 if (cost2.imageNum==1){
//                     cost2.initOptimization();
//                     cost2.optimize();//Launches the optimizer threads
//                 }

            }
//
////             if (cost.imageNum==1){
////                 gpause();
////             }
//
////             if (cost.imageNum==2){
////                 cost.initOptimization();//jumpstart the optimization with the approximate answer at a few images
////                 usleep(1000000);
////                 tracker.pose=tracker.basePose;
////             }
//
//
//
//            if(! image.data )                              // Check for invalid input
//            {
//                cout <<  "Could not open or find the image" << std::endl ;
//                return -1;
//            }
//
//    //         namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
//    //         imshow( "Display window", cost.baseImage );                   // Show our image inside it.
//    //     waitKey(0);                                          // Wait for a keystroke in the window
//    usleep(10);

            //cout<<cv.downloadOldStyle(5);
        }
//        allDie=1;
//        sleep(10);
       // cudaProfilerStop();
        myExit();
    }
    while(1){
        usleep(1000);
    }
end:

    return 0;
}


