#include "min.part.hpp"
#include <opencv2/core/core.hpp>

void Cost::minv(uchar* _data,cv::Mat& _minIndex,cv::Mat& _minValue){
    minv((float*) _data, _minIndex, _minValue);
}



void Cost::minv(float* _data,cv::Mat& _minIndex,cv::Mat& _minValue){
    assert(_minIndex.type()==CV_32SC1);
    int r=rows;
    int c=cols;
    int l=layers;
    _minIndex.create(rows,cols,CV_32SC1);
    _minValue.create(rows,cols,CV_32FC1);
    float* data=(float*)( _data);
    int* minIndex=(int*)(_minIndex.data);
    float* minValue=(float*)(_minValue.data);
    
    for(int i=0,id=0;i<r*c;i++){//i is offset in 2d, id is offset in 3d
        //first element is min so far
        int mi=0;
        float mv=data[id];
        id++;
        for (int il=1;il<l;il++,id++){//il is layer index
            float v=data[id];
            if(mv>v){
                mi=il;
                mv=v;
            }
        }
        minIndex[i]=mi; 
        minValue[i]=mv; 
    }
}



void Cost::maxv(float* _data,cv::Mat& _maxIndex,cv::Mat& _maxValue){
    assert(_maxIndex.type()==CV_32SC1);
    
    int r=rows;
    int c=cols;
    int l=layers;
    _maxIndex.create(rows,cols,CV_32SC1);
    _maxValue.create(rows,cols,CV_32FC1);
    float* data=(float*)( _data);
    int* maxIndex=(int*)(_maxIndex.data);
    float* maxValue=(float*)(_maxValue.data);
    
    for(int i=0,id=0;i<r*c;i++){//i is offset in 2d, id is offset in 3d
        //first element is max so far
        int mi=0;
        float mv=data[id];
        id++;
        for (int il=1;il<l;il++,id++){//il is layer index
            float v=data[id];
            if(mv<v){
                mi=il;
                mv=v;
            }
        }
        maxIndex[i]=mi; 
        maxValue[i]=mv; 
    }
}

void Cost::minmax(){
    int r=rows;
    int c=cols;
    int l=layers;
    lo.create(rows,cols,CV_32FC1);
    hi.create(rows,cols,CV_32FC1);
    float* maxValue=(float*)(hi.data);
    float* minValue=(float*)(lo.data);
    
    for(int i=0,id=0;i<r*c;i++){//i is offset in 2d, id is offset in 3d
        //first element is max so far
        float mhiv=data[id];
        float mlov=data[id];
        id++;
        for (int il=1;il<l;il++,id++){//il is layer index
            float v=data[id];
            if(mhiv<v){
                mhiv=v;
            }
            if(mlov>v){
                mlov=v;
            }
        }
        minValue[i]=mlov; 
        maxValue[i]=mhiv; 
    }
}

