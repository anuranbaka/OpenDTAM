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
#include "graphics.hpp"
#include "set_affinity.h"
#include "Track/Track.hpp"
#include "utils/utils.h"

//debug
#include "tictoc.h"

//A test program to make the mapper run
using namespace cv;
using namespace std;

int App_main( int argc, char** argv );


static void myshow(const string name,const Mat& _mat){
    Mat mat=_mat.clone();
    double min;
    double max;
    cv::minMaxIdx(mat, &min, &max);
    float scale = 1/ (max-min);
    mat.convertTo(mat,CV_32FC1, scale, -min*scale);
    mat.convertTo(mat,CV_8UC3, 255.0);//use 8 bit so we can have the nice mouse over
    cout<<name<<": view scale: "<<max-min<<endl;
    namedWindow(name, 1 );
    imshow( name, mat);
    
}

int main( int argc, char** argv ){
//namedWindow("backtrans",CV_WINDOW_OPENGL*0);
set_affinity(1);
//cvStartLoop(&App_main,argc, argv);//will crash if used with opengl!
initGui();
return App_main(argc, argv);
}

int App_main( int argc, char** argv )
{
    pthread_setname_np(pthread_self(),"App_main");
        
    FileStorage fs;


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
    imread(filename,-1).convertTo(image,CV_32FC3,1.0/65535.0);   // Read the file

    hconcat(R,T,cameraAffinePoseBase);

    Cost cost(image.clone(),32, cameraMatrix, R,T);
    Track tracker(cost);
    assert(cost.rows==480);
    
    vector<Mat> images,Rs,Ts;
    for(int i=0;i<=50;i++){
        sprintf(filename,"/local_store/Dropbox/Research/DTAM GSoC/OpenDTAM/Trajectory_30_seconds/scene_%03d.png",i);
        convertAhandaPovRayToStandard("/local_store/Dropbox/Research/DTAM GSoC/OpenDTAM/Trajectory_30_seconds",
                                      i,
                                      cameraMatrix,
                                      R,
                                      T);
        Mat image;
        cout<<filename<<endl;
        imread(filename, -1).convertTo(image,CV_32FC3,1.0/65535.0);
        images.push_back(image.clone());
        Rs.push_back(R.clone());
        Ts.push_back(T.clone());
    }
    while(1){
        for (int imageNum=1;imageNum<=50;imageNum++){
            char filename[500];
            Mat cameraMatrix,R,T;
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


            Mat cameraAffinePoseAlternate,mask;
            hconcat(R,T,cameraAffinePoseAlternate);

            if (cost.imageNum<2){
            cost.updateCostL1(image,R,T);
            }
            if (cost.imageNum==1){ 
                cost.optimize();//Launches the optimizer threads
            }
            const Mat thisPose(cost.convertPose(R,T));
            cost.initOptimization();//jumpstart the optimization with the approximate answer at 4 images
            
//             reprojectCloud(image,cost.baseImage, cost._d*cost.depthStep, Mat(cost.pose), thisPose, Mat(cost.cameraMatrix));
            //Test out the Tracker
            {
                Mat tp;
                RTToLie(R,T,tp); 
                tracker.depth=abs(cost._a*cost.depthStep);
                tracker.addFrame(image);
                //tracker.pose=tracker.basePose;
                tracker.align();
                Mat p=tracker.pose;
                cout << "True Pose: "<< tp << endl;
                cout << "Recovered Pose: "<< p << endl;
                cout << "Pose Error: "<< p-tp << endl;
            }
            
            
            if (imageNum==4){
                //cost.initOptimization();//jumpstart the optimization with the approximate answer at 4 images
                usleep(1000000);
                tracker.pose=tracker.basePose;
            }

            

            if(! image.data )                              // Check for invalid input
            {
                cout <<  "Could not open or find the image" << std::endl ;
                return -1;
            }

    //         namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    //         imshow( "Display window", cost.baseImage );                   // Show our image inside it.
    //     waitKey(0);                                          // Wait for a keystroke in the window
    usleep(10);
        }
    }
    while(1){
        usleep(1000);
    }

    return 0;
}


