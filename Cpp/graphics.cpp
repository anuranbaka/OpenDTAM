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
static queue<Vec2d> autoScale;

static queue<int> props;
static queue<string> nameWin;
static boost::mutex Gmux; 
static volatile int ready=0;
static volatile int pausing=0;

void gpause(){
    CV_XADD(&pausing,1);
}


void pfShow(const string name,const Mat& _mat,int defaultscale, Vec2d autoscale){
    if (defaultscale==1){
        autoscale=Vec2d(-1,-1);
    }
    //cull frames
    Gmux.lock();
    nameShow.push(name);
    toShow.push(_mat.clone());
    autoScale.push(autoscale);
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
template <class T>
static inline T take(queue<T>& q){
    T ref=q.front();
    q.pop();
    return ref;
}




void* guiLoop(void*){
    set_affinity(0);
    pthread_setname_np(pthread_self(),"Graphics");
    Mat mat;
    while(1){
        if(CV_XADD(&pausing,-1)>=1){//deal with pauses
            namedWindow("d",CV_WINDOW_KEEPRATIO);
            cout<<"Paused"<<endl;
            waitKey();
        }
        if (props.size()>0){//deal with new windows
            Gmux.lock();
            string name=take(nameWin);
            int prop=take(props);
            
            Gmux.unlock();
            namedWindow(name,prop);
        }
        if (ready){//deal with imshows
            Gmux.lock();
            assert(nameShow.size()>0);
            mat=take(toShow);
            string name=take(nameShow);
            Vec2d autoscale=take(autoScale);
            ready--;
            Gmux.unlock();
            if ((autoscale[0]==autoscale[1] && autoscale[0]==0)){
                double min; 
                double max;
                cv::minMaxIdx(mat, &min, &max);
                float scale = 1.0/ (max-min);
                mat.convertTo(mat,CV_MAKETYPE(CV_32F,mat.channels()), scale, -min*scale);

                
                mat.convertTo(mat,CV_MAKETYPE(CV_8U,mat.channels()), 255.0);//use 8 bit so we can have the nice mouse over
                cout<<name<<": view scale: "<<max-min<<endl;
                cout<<name<<": min: "<<min<<"  max: "<< max<<endl;
            }else if (autoscale[0]!=autoscale[1]){
                double scale= 1.0/(autoscale[1]-autoscale[0]);
                mat.convertTo(mat,CV_MAKETYPE(mat.type(),mat.channels()),scale,-autoscale[0]*scale);
            }
            if(mat.rows<50){
                namedWindow(name,CV_WINDOW_KEEPRATIO);
            }
            imshow( name, mat);
           
        }
        waitKey(1);
//         usleep(100);
    }
    return NULL;
}
void initGui(){
    pthread_t threadGui;
    pthread_create( &threadGui, NULL, &guiLoop, NULL);
}

