#ifndef COST_H
#define COST_H
#include <opencv2/core/core.hpp>
#include <vector>
#include "tictoc.h"
// The cost volume. Conceptually arranged as an image plane, corresponding
// to the keyframe, lying on top of the actual cost volume, a 3D two channel matrix storing
// the total cost of all rays that have passed through a voxel, and the number of rays that
// have hit that voxel.
//
// There is also the depth array, which stores the inverse depths of each plane of the cost.
//
// For efficiency, the accumulated cost and hit count are seperate arrays even though
// conceptually they are just different channels of the same image.
//
// The cost volume doesn't support updating by a different camera than the one that took the
// keyframe, because that would violate a bunch of assumptions for DTAM
#define COST_H_DEFAULT_NEAR .015


class Cost{
public:
    cv::Mat rayHits;// number of times a ray has been hit(not implemented)
    cv::Mat_<cv::Vec3f> baseImage;
    cv::Mat lo;
    cv::Mat hi;
    int rows;
    int cols;
    int layers;
    std::vector<float> depth;
    float near;
    float far;
    float depthStep;
    cv::Matx33d cameraMatrix;
    cv::Matx44d pose;//the affine transform representing the world -> camera frame transformation
    float* data;// stores the [rows][cols][layers] array of sum of costs so far
    float* hit;//stores the number of times each cell has been hit by a ray
    int imageNum;


    Cost();//DANGER: only use for copying to later
    Cost(const cv::Mat& baseImage, int layers,                      const cv::Mat& cameraMatrix, const cv::Mat& R, const cv::Mat& Tr);// autogenerate default depths
    Cost(const cv::Mat& baseImage, int layers,                      const cv::Mat& cameraMatrix, const cv::Matx44d& cameraPose);// autogenerate default depths
    Cost(const cv::Mat& baseImage, const std::vector<float>& depth, const cv::Mat& cameraMatrix, const cv::Mat& R, const cv::Mat& Tr);//use given depths
    Cost(const cv::Mat& baseImage, const std::vector<float>& depth, const cv::Mat& cameraMatrix, const cv::Matx44d& cameraPose);//use given depths
    


    void updateCostL1(const cv::Mat& image, const cv::Matx44d& currentCameraPose);
    void updateCostL1(const cv::Mat& image, const cv::Mat& R, const cv::Mat& Tr);
    void updateCostL2(const cv::Mat& image, const cv::Matx44d& currentCameraPose);
    void updateCostL2(const cv::Mat& image, const cv::Mat& R, const cv::Mat& Tr);
    void optimize();
    void initOptimization();
    
    const cv::Mat depthMap(); //return the best available depth map

    const cv::Matx44d convertPose(const cv::Mat& R, const cv::Mat& Tr){
        cv::Mat pose=cv::Mat::eye(4,4, CV_64F);
        R.copyTo(pose(cv::Range(0,3),cv::Range(0,3)));
        Tr.copyTo(pose(cv::Range(0,3),cv::Range(3,4)));
            
        return cv::Matx44d(pose);
    }


private:
    cv::Mat_<float> dataContainer; //stores the actual data for data*, used for auto allocation behavior
    cv::Mat_<float> hitContainer; //stores the actual data for hit*, used for auto allocation behavior


    //Initializer functions
    void init(){
        assert(baseImage.data);//make sure not trying to init an imageless object
        depthStep=((depth.back()-depth[0])/layers);
        near = depth.back();
        far  = depth.front();
        data=(float*)dataContainer.data;
        hit=(float*)hitContainer.data;
        _a.create(rows,cols,CV_32FC1);
        aptr=_a.data;
        _d.create(rows,cols,CV_32FC1);
        dptr=_d.data;
        imageNum=0;
        QDruncount=0;
        Aruncount=0;
        thetaStart=500.0;
        thetaMin=0.01;
        running_a=running_qd=false;
        initOptimization();

        epsilon=.1;
        lambda=.000001;
        thetaStep=.99;
    }
    std::vector<float> generateDepths(int layers){
        std::vector<float> depths;
        for(float n=0; n<layers; n++){
                depths.push_back(n/(layers-1)*COST_H_DEFAULT_NEAR);
            }
        return depths;// generate the depth list, the signature will probably change
    }

    
    //Utility Functions
    void minv(uchar*/*(float*)*/,cv::Mat& minIndex,cv::Mat& minValue);
    void minv(float*            ,cv::Mat& minIndex,cv::Mat& minValue);
    void maxv(float*/*(float*)*/,cv::Mat& maxIndex,cv::Mat& maxValue);   
    void minmax();    

    //DepthmapDenoiseWeightedHuber functions and data
    public:cv::Mat _qx,_qy,_d,_a,_g,_gu,_gd,_gl,_gr,_gbig;private:
    uchar* aptr;
    uchar* dptr;
    cv::Mat stableDepth;
    float theta,thetaStart,thetaStep,thetaMin,epsilon,lambda,sigma_d,sigma_q;
    
    void computeSigmas();
    void cacheGValues();
    
        //Q update
    public: void optimizeQD();private://NOT PUBLIC!!! JUST NEED TO ACCESS FROM A STATIC CALL. DO NOT USE!
        //A update
    float aBasic(float* data,float l,float ds,float d,float& value);
    public: void optimizeA();private://NOT PUBLIC!!! JUST NEED TO ACCESS FROM A STATIC CALL. DO NOT USE!
    
    
    //Instrumentation
    int QDruncount;
    int Aruncount;
    
    //Thread management
    public:
        volatile bool running_a, running_qd;
};


#endif
