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
    initGui();
    return App_main(argc, argv);
}
void myExit(){
    ImplThread::stopAllThreads();
}

int App_main( int argc, char** argv )
{
    int numImg=50;
    cv::gpu::CudaMem imageContainer;
    pthread_setname_np(pthread_self(),"App_main");

    char filename[500];
    Mat image, cameraMatrix, R, T;
    vector<Mat> images,Rs,Ts;

    
    double reconstructionScale=5/5.;

    for(int i=0;i<=numImg;i++){
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

    Cost cost(images[0].clone(),32, cameraMatrix, Rs[0],Ts[0]);

    CostVolume cv(images[0],(FrameID)0,32,0.015,0.0,Rs[0],Ts[0],cameraMatrix);

    Optimizer optimizer(cv);
    optimizer.initOptimization();

    Track tracker(cv,optimizer);
    
    OpenDTAM odm(cameraMatrix);
    odm.addFrameWithPose(images[0],Rs[0],Ts[0]);
    odm.addFrameWithPose(images[10],Rs[10],Ts[10]);
    for (int imageNum=2;imageNum<=numImg;imageNum++){
        odm.addFrame(images[imageNum]);
        usleep(100000);
    }
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
                  optimizer2._a.download(ret);
                  pfShow("A", ret, 0, cv::Vec2d(0, 32));

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
                      optimizer2._d.download(ret);
                      pfShow("D function", ret, 0, cv::Vec2d(0, 32));
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




            if(imageNum==10){
                tracker.pose=RTToLie(R,T);
            }
            //Test out the Tracker
            {
                Mat tp;
//                 tracker.pose=tp.clone();//Give the answer
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

            }

    usleep(100000);

        }
}

end:

    return 0;
}


