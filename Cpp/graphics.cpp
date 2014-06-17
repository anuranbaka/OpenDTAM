#include <opencv2/opencv.hpp> 
#include <boost/thread/mutex.hpp>
#include <queue>
#include <string>
#include "set_affinity.h"
#include "graphics.hpp"
using namespace std;
using namespace cv;
static queue<Mat> toShow;
static queue<string> nameShow;
static queue<int> props;
static queue<string> nameWin;
static boost::mutex Gmux; 
static volatile int ready=0;
void pfShow(const string name,const Mat& _mat){
    //cull frames
    if(rand()%30!=0)
        return;
    Gmux.lock();
    nameShow.push(name);
    toShow.push(_mat.clone());
    ready++;
    assert(nameShow.size()==ready);
    Gmux.unlock();
    while(nameShow.size()>5){
        usleep(100);
    }

}
void pfWindow(const string name,int prop){
    Gmux.lock();
    nameWin.push(name);
    props.push(prop);

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
            string name=nameWin.front();
            nameWin.pop();
            int prop=props.front();
            props.pop();
            Gmux.unlock();
            namedWindow(name,prop);
        }
        while (ready){
            Gmux.lock();
            Mat mat=toShow.front();
            toShow.pop();
            string name=string(nameShow.front());
            nameShow.pop();
            ready--;
            Gmux.unlock();
            double min; 
            double max;
            cv::minMaxIdx(mat, &min, &max);
//             float scale = 1/ (max-min);
//             mat.convertTo(mat,CV_MAKETYPE(CV_32F,mat.channels()), scale, -min*scale);
            float scale = 1/ (max-0);
            mat.convertTo(mat,CV_MAKETYPE(CV_32F,mat.channels()), scale, 0*scale);
            
            mat.convertTo(mat,CV_MAKETYPE(CV_8U,mat.channels()), 255.0);//use 8 bit so we can have the nice mouse over
//             cout<<name<<": view scale: "<<max-min<<endl;
//             cout<<name<<": min: "<<min<<"  max: "<< max<<endl;
            if(mat.rows<50){
            namedWindow(name,CV_WINDOW_KEEPRATIO);
            }
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

