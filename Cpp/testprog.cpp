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
#include "Scheduler/Frame.hpp"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "OpenDTAM.hpp"

const static bool valgrind=0;

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
    ImplThread::stopAllThreads();
//     allDie=1;
//     gcheck();
//     usleep(100000);
//     exit(0);
}

int App_main( int argc, char** argv )
{
//     VideoCapture cap(0); // open the default camera
// 
//     if(!cap.isOpened()) // check if we succeeded
//         return -1;
//     while(1){
//         Mat frame;
//         cap >> frame; // get a new frame from camera
//         pfShow("frame",frame);
//     }


    int numImg=50;
    cv::gpu::CudaMem imageContainer;
    pthread_setname_np(pthread_self(),"App_main");
//         cout<<sizeof(Frame)<<endl;
//         cout<<"hi"<<endl;

    FileStorage fs;
    void* junk;
    cudaMalloc(&junk,5500);
    cudaFree(junk);

    Mat cameraAffinePoseBase;
    int imageNum=0;
    char filename[500];
    Mat cameraMatrix,R,T;
    vector<Mat> images,Rs,Ts;
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

        Size2d s0=image.size();
        resize(image,image,Size(),sc,sc);
        Size2d sn=image.size();
        double sx=(sn.width/s0.width);
        double sy=(sn.height/s0.height);
        cameraMatrix+=(Mat)(Mat_<double>(3,3) << 0,0.0,0.5,
                                            0.0,0.0,0.5,
                                            0.0,0.0,0);
        cameraMatrix=cameraMatrix.mul((Mat)(Mat_<double>(3,3) <<    sx,0.0,sx,
                                                0.0,sy ,sy,
                                                0.0,0.0,1.0));
        cameraMatrix-=(Mat)(Mat_<double>(3,3) << 0,0.0,0.5,
                                            0.0,0.0,0.5,
                                            0.0,0.0,0);


        
//            cout<<"cameraMatrix: "<<cameraMatrix<<"\n";
    }else{
        image.create(480,640,CV_32FC3);
        image=0.5;
    }

    OpenDTAM odm(cameraMatrix);
    odm.addFrameWithPose(image,R,T);
    cout<<cameraMatrix<<endl;
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


    for(int i=0;i<=numImg;i++){
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
    odm.addFrameWithPose(images[10],Rs[10],Ts[10]);
//     while(1){
//         usleep(100000);
//     }
    cudaProfilerStart();
    while(1){
        for (int imageNum=2;imageNum<=numImg;imageNum++){
            odm.addFrame(images[imageNum]);
            
            
            
            
            if(imageNum==1){
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


            imageContainer.create(image.rows,image.cols,CV_8UC4);
            Mat tmp,ret;
            cvtColor(image,tmp,CV_RGB2RGBA);
            Mat imageContainerRef=imageContainer;//Required by ambiguous conversion rules
            tmp.convertTo(imageContainerRef,CV_8UC4,255.0);
//             if (imageNum<2){
//                 cv.updateCost(imageContainer, R, T);
// //                cv.loInd.download(ret);
// //                assert(cv.loInd.isContinuous());
// //                pfShow("loInd Soln", cv.downloadOldStyle(0));
// //                gpause();
//             }


            if (imageNum==10    ){//ucv test
                //init ucv
                tic();
                imageContainer.create(image.rows,image.cols,CV_8UC4);
                Mat tmp,ret;
                cvtColor(image,tmp,CV_RGB2RGBA);
                Mat imageContainerRef=imageContainer;//Required by ambiguous conversion rules
                tmp.convertTo(imageContainerRef,CV_8UC4,255.0);
                CostVolume cv2(images[0],(FrameID)0,32,0.015,0.0,Rs[0],Ts[0],cameraMatrix);
                cv2.updateCost(imageContainer, R, T);
                
                Optimizer optimizer2(cv2);
//                pfShow("ADD", cv2.downloadOldStyle(0));
//                gpause();

                optimizer2.initOptimization();
                
                //cudaStreamSynchronize();
                bool doneOptimizing;
                do{
//                    cout<<"Theta: "<< optimizer2.theta<<endl;
//                   optimizer2._a.download(ret);
//                   pfShow("A", ret, 0, cv::Vec2d(0, 32));

    //                optimizer.cacheGValues();
    //                optimizer._gy.download(ret);
    //                pfShow("G function", ret, 0, cv::Vec2d(0, 1));
    //                gpause();
                    for (int i = 0; i < 10; i++) {
                        optimizer2.optimizeQD();
//                        cudaDeviceSynchronize();
//                       optimizer2._qx.download(ret);
//                       pfShow("Qx function", ret, 0, cv::Vec2d(-1, 1));
//                       optimizer2._gy.download(ret);
//                       pfShow("Gy function", ret, 0, cv::Vec2d(0, 1));
//                       optimizer2._d.download(ret);
//                       pfShow("D function", ret, 0, cv::Vec2d(0, 32));
//                       usleep(100000);
                        //gpause();
                        
                    }
//                    cudaDeviceSynchronize();
                    doneOptimizing=optimizer2.optimizeA();
                }while(!doneOptimizing);
                while(!optimizer2.cvStream.queryIfComplete()){
                    pfShow("Ongoing Solution", optimizer2.depthMap());
                }
                toc();
                
                
//                 pfShow("Depth Solution", optimizer2.depthMap());
//                 optimizer2._a.download(ret);
//                 pfShow("Depth Solution", ret, 0, cv::Vec2d(0, 32));
//                 gpause();
                cv=cv2;
                optimizer=optimizer2;

//                 myExit();
            }




//             cv.loInd.download(ret);
//             pfShow("Initial Min Soln",ret,0,cv::Vec2d(0,32));

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
                pfShow("Tdepth",tracker.depth);
                tracker.addFrame(image);

                tracker.align();
                Mat p=tracker.pose;
//                 {//debug
//                     cout << "True Pose: "<< tp << endl;
//                     cout << "True Delta: "<< LieSub(tp,tracker.basePose) << endl;
//                     cout << "Recovered Pose: "<< p << endl;
//                     cout << "Recovered Delta: "<< LieSub(p,tracker.basePose) << endl;
//                     cout << "Pose Error: "<< p-tp << endl;
//                 }
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
    usleep(100000);

            //cout<<cv.downloadOldStyle(5);

        }
//        allDie=1;
//        sleep(10);
       // cudaProfilerStop();
       while(1)usleep(100000);;
        break;  
    }

end:

    return 0;
}


