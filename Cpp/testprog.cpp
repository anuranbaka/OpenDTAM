#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include "convertAhandaPovRayToStandard.h"
#include "reproject.hpp"
#include "Cost.h"
//A test program to make the mapper run
using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    FileStorage fs;

    Mat R = Mat_<uchar >::eye  (3, 3);
    Mat T = Mat_<double>::zeros(3, 1);

    fs << "R" << R;                                      // Write cv::Mat
    fs << "T" << T;

    fs["R"] >> R;                                      // Read cv::Mat
    fs["T"] >> T;
    Mat cameraAffinePoseBase;
    Cost<float,Vec3f> mycost;
    
    
    for (int imageNum=0;imageNum<=10;imageNum++){
        char filename[500];
        Mat cameraMatrix,R,T;
        convertAhandaPovRayToStandard("/local_store/Dropbox/Research/DTAM GSoC/OpenDTAM/Trajectory_30_seconds",
                                   imageNum,
                                   cameraMatrix,
                                   R,
                                   T);
        
        cout<<"cameraMatrix: "<<cameraMatrix<<"\n";
        cout<< "R : "<<R<<"\n";
        cout<< "T : "<<T<<"\n";
        sprintf(filename,"/local_store/Dropbox/Research/DTAM GSoC/OpenDTAM/Trajectory_30_seconds/scene_%03d.png",imageNum);
        Mat image;
        imread(filename, CV_LOAD_IMAGE_COLOR).convertTo(image,CV_32FC3,1/255.0);   // Read the file
        
        if(imageNum==0){
            hconcat(R,T,cameraAffinePoseBase);
            mycost=Cost<float,Vec3f>(image,32, cameraMatrix, R,T);
        }
        
        Mat cameraAffinePoseAlternate,mask;
        hconcat(R,T,cameraAffinePoseAlternate);
        reproject(image,
                cameraMatrix,
                cameraAffinePoseBase,
                cameraAffinePoseAlternate,
                .0019,
                image,
                mask);
        

        if(! image.data )                              // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }

        namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
        imshow( "Display window", image );                   // Show our image inside it.
    waitKey(0);                                          // Wait for a keystroke in the window
    }
    return 0;
}

