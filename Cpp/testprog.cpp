#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
// #include <unistd.h>


//Mine
#include "convertAhandaPovRayToStandard.h"
// #include "CostVolume/utils/reproject.hpp"
// #include "CostVolume/utils/reprojectCloud.hpp"
// #include "CostVolume/Cost.h"
// #include "CostVolume/CostVolume.hpp"
// #include "Optimizer/Optimizer.hpp"
#include "graphics.hpp"
// #include "set_affinity.h"
// #include "Track/Track.hpp"
// #include "utils/utils.hpp"


//debug
// #include "tictoc.h"
// #include "Scheduler/Frame.hpp"



#include "OpenDTAM.hpp"

const static bool valgrind=0;

//A test program to make the mapper run
using namespace cv;
using namespace std;

int App_main( int argc, char** argv );


int main( int argc, char** argv ){

    initGui();
    sleep(1);
    return App_main(argc, argv);
}
void myExit(){
//     cudaDeviceSynchronize();
    ImplThread::stopAllThreads();
}

int App_main( int argc, char** argv )
{
    int numImg=50;
#if !defined WIN32 && !defined _WIN32 && !defined WINCE && defined __linux__ && !defined ANDROID
    pthread_setname_np(pthread_self(),"App_main");
#endif
    FileStorage fs;

    int imageNum=0;
    
    Mat cameraMatrix,R,T;
    vector<Mat> images,Rs,Ts;

    Mat image; 
    double reconstructionScale=1/5.;



    for(int i=0;i<=numImg;i++){
        char filename[500];
        sprintf(filename,"../../Trajectory_30_seconds/scene_%03d.png",i);
        convertAhandaPovRayToStandard("../../Trajectory_30_seconds",
                                      i,
                                      cameraMatrix,
                                      R,
                                      T);
        Mat image;
        cout<<"Opening:"<<filename<<endl;
        
        imread(filename, -1).convertTo(image,CV_32FC3,1.0/65535.0);
        
        resize(image,image,Size(),reconstructionScale,reconstructionScale);
        images.push_back(image.clone());
        Rs.push_back(R.clone());
        Ts.push_back(T.clone());
    }
    double sx=reconstructionScale;
    double sy=reconstructionScale;
    cameraMatrix+=(Mat)(Mat_<double>(3,3) << 0,0.0,0.5,
                                            0.0,0.0,0.5,
                                            0.0,0.0,0);
    cameraMatrix=cameraMatrix.mul((Mat)(Mat_<double>(3,3) <<    sx,0.0,sx,
                                            0.0,sy ,sy,
                                            0.0,0.0,1.0));
    cameraMatrix-=(Mat)(Mat_<double>(3,3) << 0,0.0,0.5,
                                        0.0,0.0,0.5,
                                        0.0,0.0,0);
    
    OpenDTAM odm(cameraMatrix);
    odm.addFrameWithPose(images[0],Rs[0],Ts[0]);
    odm.addFrameWithPose(images[10],Rs[10],Ts[10]);
    for (int imageNum=2;imageNum<=numImg;imageNum++){
        odm.addFrame(images[imageNum]);
    }
    usleep(1000000);
    gcheck();
    myExit();
    return 0;
}


