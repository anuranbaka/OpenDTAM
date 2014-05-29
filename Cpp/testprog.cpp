#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>

//Mine
#include "convertAhandaPovRayToStandard.h"
#include "CostVolume/reproject.hpp"
#include "CostVolume/Cost.h"
#include "set_affinity.h"

//debug
#include "tictoc.h"

//A test program to make the mapper run
using namespace cv;
using namespace std;

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



int main( int argc, char** argv )
{
    tic();
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
    assert(cost.rows==480);

    for (int imageNum=1;imageNum<=5;imageNum++){
        char filename[500];
        Mat cameraMatrix,R,T;
        convertAhandaPovRayToStandard("/local_store/Dropbox/Research/DTAM GSoC/OpenDTAM/Trajectory_30_seconds",
                                   imageNum,
                                   cameraMatrix,
                                   R,
                                   T);

//         cout<<"cameraMatrix: "<<cameraMatrix<<"\n";
//         cout<< "R : "<<R<<"\n";
//         cout<< "T : "<<T<<"\n";
        sprintf(filename,"/local_store/Dropbox/Research/DTAM GSoC/OpenDTAM/Trajectory_30_seconds/scene_%03d.png",imageNum);
        Mat image;
        imread(filename, -1).convertTo(image,CV_32FC3,1.0/65535.0);   // Read the file



        Mat cameraAffinePoseAlternate,mask;
        hconcat(R,T,cameraAffinePoseAlternate);

        cost.updateCostL1(image,R,T);
        if (imageNum==1){
            set_affinity(1);//Move us to core 1
            cost.optimize();//Launches the optimizer threads
        }
        if (imageNum==4){
            cost.initOptimization();//jumpstart the optimization with the approximate answer at 4 images
        }
//         myshow("qx",abs(cost._qx));
//         myshow("d",cost._d);
//         myshow("a",cost._a);
//         cvWaitKey(1);
        

        if(! image.data )                              // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }

//         namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
//         imshow( "Display window", cost.baseImage );                   // Show our image inside it.
//     waitKey(0);                                          // Wait for a keystroke in the window
    }
    toc();
    while(1){
        usleep(1000);
    }
    return 0;
}


