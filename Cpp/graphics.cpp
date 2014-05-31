#include <opencv2/opencv.hpp> 
#include <boost/thread/mutex.hpp>
#include <vector>
#include <string>
#include "set_affinity.h"
using namespace std;
using namespace cv;
static vector<Mat> toShow;
static vector<string> nameShow;
static vector<int> props;
static vector<string> nameWin;
static boost::mutex Gmux; 

void pfShow(const string name,const Mat& _mat){
    Gmux.lock();
    nameShow.push_back(name);
    toShow.push_back(_mat);

    Gmux.unlock();
    while(nameShow.size()>5){
        usleep(100);
    }

}
void pfWindow(const string name,int prop){
    Gmux.lock();
    nameWin.push_back(name);
    props.push_back(prop);

    Gmux.unlock();
    while(nameWin.size()>5){
        usleep(100);
    }
}
void* guiLoop(void*){
    set_affinity(0);
    pthread_setname_np(pthread_self(),"Graphics");
    while(1){
        if (props.size()>0){
            Gmux.lock();
            string name=nameWin.back();
            nameWin.pop_back();
            int prop=props.back();
            props.pop_back();
            Gmux.unlock();
            namedWindow(name,prop);
        }
        if (toShow.size()>0){
            Gmux.lock();
            Mat mat=toShow.back().clone();
            toShow.pop_back();
            string name=string(nameShow.back());
            nameShow.pop_back();
            Gmux.unlock();
            double min;
            double max;
            cv::minMaxIdx(mat, &min, &max);
            float scale = 1/ (max-min);
            mat.convertTo(mat,CV_32FC1, scale, -min*scale);
            mat.convertTo(mat,CV_8UC3, 255.0);//use 8 bit so we can have the nice mouse over
            cout<<name<<": view scale: "<<max-min<<endl;
            
            imshow( name, mat);
            
        }
        waitKey(1);
    }
    return NULL;
}
void initGui(){
    pthread_t threadGui;
    pthread_create( &threadGui, NULL, &guiLoop, NULL);
}

