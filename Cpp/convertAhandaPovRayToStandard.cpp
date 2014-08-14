// This file converts the file format used at http://www.doc.ic.ac.uk/~ahanda/HighFrameRateTracking/downloads.html
// into the standard [R|T] world -> camera format used by OpenCV
// It is based on a file they provided there, but makes the world coordinate system right handed, with z up,
// x right, and y forward.

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>

using namespace cv;
using namespace std;
Vec3f direction;
Vec3f upvector;
void convertAhandaPovRayToStandard(const char * filepath,
                                   int imageNumber,
                                   Mat& cameraMatrix,
                                   Mat& R,
                                   Mat& T)
{
    char text_file_name[600];
    sprintf(text_file_name,"%s/scene_%03d.txt",filepath,imageNumber);

    cout << "text_file_name = " << text_file_name << endl;

    ifstream cam_pars_file(text_file_name);
    if(!cam_pars_file.is_open())
    {
        cerr<<"Failed to open param file, check location of sample trajectory!"<<endl;
        exit(1);
    }

    char readlinedata[300];

    Point3d direction;
    Point3d upvector;
    Point3d posvector;


    while(1){
        cam_pars_file.getline(readlinedata,300);
//         cout<<readlinedata<<endl;
        if ( cam_pars_file.eof())
            break;


        istringstream iss;


        if ( strstr(readlinedata,"cam_dir")!= NULL){


            string cam_dir_str(readlinedata);

            cam_dir_str = cam_dir_str.substr(cam_dir_str.find("= [")+3);
            cam_dir_str = cam_dir_str.substr(0,cam_dir_str.find("]"));

            iss.str(cam_dir_str);
            iss >> direction.x ;
            iss.ignore(1,',');
            iss >> direction.z ;
            iss.ignore(1,',') ;
            iss >> direction.y;
            iss.ignore(1,',');
//             cout << "direction: "<< direction.x<< ", "<< direction.y << ", "<< direction.z << endl;

        }

        if ( strstr(readlinedata,"cam_up")!= NULL){

            string cam_up_str(readlinedata);

            cam_up_str = cam_up_str.substr(cam_up_str.find("= [")+3);
            cam_up_str = cam_up_str.substr(0,cam_up_str.find("]"));


            iss.str(cam_up_str);
            iss >> upvector.x ;
            iss.ignore(1,',');
            iss >> upvector.z ;
            iss.ignore(1,',');
            iss >> upvector.y ;
            iss.ignore(1,',');



        }

        if ( strstr(readlinedata,"cam_pos")!= NULL){
//            cout<< "cam_pos is present!"<<endl;

            string cam_pos_str(readlinedata);

            cam_pos_str = cam_pos_str.substr(cam_pos_str.find("= [")+3);
            cam_pos_str = cam_pos_str.substr(0,cam_pos_str.find("]"));

//            cout << "cam pose str = " << endl;
//            cout << cam_pos_str << endl;

            iss.str(cam_pos_str);
            iss >> posvector.x ;
            iss.ignore(1,',');
            iss >> posvector.z ;
            iss.ignore(1,',');
            iss >> posvector.y ;
            iss.ignore(1,',');
//             cout << "position: "<<posvector.x<< ", "<< posvector.y << ", "<< posvector.z << endl;

        }

    }

    R=Mat(3,3,CV_64F);
    R.row(0)=Mat(direction.cross(upvector)).t();
    R.row(1)=Mat(-upvector).t();
    R.row(2)=Mat(direction).t();

    T=-R*Mat(posvector);
//     cout<<"T: "<<T<<endl<<"pos: "<<Mat(posvector)<<endl;
   /* cameraMatrix=(Mat_<double>(3,3) << 480,0.0,320.5,
										    0.0,480.0,240.5,
										    0.0,0.0,1.0);*/
    cameraMatrix=(Mat_<double>(3,3) << 481.20,0.0,319.5,
                  0.0,480.0,239.5,
                  0.0,0.0,1.0);

}



