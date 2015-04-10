#ifndef OPENDTAM_HPP
#define OPENDTAM_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <queue>
#include <list>
#include <iostream>
#include <stdio.h>
#include <unistd.h>


//Mine
#include "convertAhandaPovRayToStandard.h"
#include "CostVolume/utils/reproject.hpp"
#include "CostVolume/utils/reprojectCloud.hpp"
#include "CostVolume/Cost.h"
#include "CostVolume/CostVolume.hpp"
#include "DepthmapDenoiseWeightedHuber/DepthmapDenoiseWeightedHuber.hpp"
#include "graphics.hpp"
#include "set_affinity.h"
#include "Track/Track.hpp"
#include "utils/utils.hpp"
#include "Scheduler/Frame.hpp"
#include "Scheduler/SynchronizedBuffer.hpp"



//debug
#include "tictoc.h"

//move to cpp
#include "Track/Alignment.cpp"
#include "utils/ImplThreadLaunch.hpp"



using namespace cv;
using namespace std;
using namespace cv::gpu;
class OpenDTAM{
    int rows,cols;
    Mat cameraMatrix;
    int utrkLevel2DStart,utrkLevel2DEnd;
    int utrkLevel3DStart,utrkLevel3DEnd;
    int mtrkLevel3DStart,mtrkLevel3DEnd;
    int pyrLevels;
    Size ucvSize;
    vector<Mat> cameraMatrixPyr;

    StallableSynchronizedStack<Ptr<Frame> > fs;//the frame stream
    StallableSynchronizedQueue<Ptr<Frame> > utrkq;//frames needing fast tracking
    StallableSynchronizedStack<Ptr<Frame> > utrkd;//frames that have been tracked at rough level
    StallableSynchronizedQueue<Ptr<Frame> > mtrkq;//frames needing fine tracking
    StallableSynchronizedStack<Ptr<Frame> > mtrkd;//frames that have been tracked at fine level
    StallableSynchronizedStack<Ptr<Frame> > trkd; //frames that have been tracked at all levels

    StallableSynchronizedQueue<std::pair<Ptr<Frame>,std::vector<Ptr<Frame> > > > ucvq; //frames to convert to low quality cost volumes ("micro cv's")
    StallableSynchronizedQueue<std::pair<Ptr<Frame>,std::vector<Ptr<Frame> > > > mcvq; //frames to convert to high quality cost volumes ("macro cv's")
    StallableSynchronizedStack<Ptr<Frame> > ucvd;// frames with micro cv's (stack)
    StallableSynchronizedStack<Ptr<Frame> > mcvd;//frames with macro cv's (stack)
    StallableSynchronizedStack<Ptr<Frame> > cvd; //frames with all cv's  (stack)
    int Ttrkid;
    int Tutrkid;
   
    
    
    FrameID fn;
    bool initd;
    public:
        
    OpenDTAM(Mat cameraMatrix):
        rows(0),
        cols(0),
        cameraMatrix(cameraMatrix),
        pyrLevels(0),
        cameraMatrixPyr(),
        utrkLevel2DStart(0),utrkLevel2DEnd(0),
        utrkLevel3DStart(0),utrkLevel3DEnd(0),
        mtrkLevel3DStart(0),mtrkLevel3DEnd(0),
        fs(),//the frame stream
        utrkq(),//frames needing fast tracking
        utrkd(),//frames that have been tracked at rough level
        mtrkq(),//frames needing fine tracking
        mtrkd(),//frames that have been tracked at fine level
        trkd(), //frames that have been tracked at all levels
        ucvq(), //frames to convert to low quality cost volumes ("micro cv's")
        mcvq(), //frames to convert to high quality cost volumes ("macro cv's")
        ucvd(),// frames with micro cv's (stack)
        mcvd(),//frames with macro cv's (stack)
        cvd(), //frames with all cv's  (stack)
        fn(0),
        initd(0)
        {}
    ~OpenDTAM(){
        ImplThreadLauncher<OpenDTAM>::stopThread(Tutrkid);
    }
    
    //Non-default parameters should be set before calling init!
    void init(const Mat& image){
        
        
        //Determine size of frame
        rows=image.rows;
        cols=image.cols;
        
        //Auto determine number of pyramid levels and tracking depth
        if(pyrLevels==0){//auto size to end at >=15px tall (use height because shortest dim usually)
            for (double scale=1.0; scale>=15.0/rows; scale/=2, pyrLevels++);
        }
        if(!utrkLevel2DStart)   utrkLevel2DStart = 0;
        if(!utrkLevel2DEnd)     utrkLevel2DEnd   = min(utrkLevel2DStart+2, pyrLevels);
        if(!utrkLevel3DStart)   utrkLevel3DStart = min(utrkLevel2DEnd,     pyrLevels-1);//assure at least 1 level of 3d
        if(!utrkLevel3DEnd)     utrkLevel3DEnd   = min(utrkLevel3DStart+2, pyrLevels);
        if(!mtrkLevel3DStart)   mtrkLevel3DStart = min(utrkLevel3DEnd,     pyrLevels);
        if(!mtrkLevel3DEnd)     mtrkLevel3DEnd   = min(mtrkLevel3DStart+2, pyrLevels);
        
        //Construct cameraMatricies for pyramids
        {
            cameraMatrixPyr.reserve(pyrLevels);
            for(int i=0;i<pyrLevels;i++){
                cameraMatrixPyr.push_back(Mat());
            }
            //Figure out camera matrices for each level
            for (double scale=1.0,l2=pyrLevels-1; l2>=0; scale/=2, l2--) {
                Mat cameraMatrix2=cameraMatrix.clone();
                cameraMatrix2(Range(0,2),Range(2,3))+=.5;
                cameraMatrix2(Range(0,2),Range(0,3))*= scale;
                cameraMatrix2(Range(0,2),Range(2,3))-=.5;
                cameraMatrixPyr[l2]=cameraMatrix2;
            }
        }
        ucvSize=Size(128,96);
        
        Tutrkid=ImplThreadLauncher<OpenDTAM>::startThread(*this,&OpenDTAM::Tutrk,"uTrack",3);
    }
    

    
    typedef Ptr<Frame> Fp;
    FrameID addFrameWithPose(const Mat& image, const Mat& R, const Mat& T){
        if(!initd){
            init(image);
            initd=1;
        }
        assert(pyrLevels>0);
        //Increment frame counter
        FrameID fid = fn++;
        
        //Construct new frame
        Fp newFp(new Frame);
        {
            Frame tmp;
            //Fill known fields
            {
                tmp.fid=fid; 
                tmp.im=new Mat(image);
                tmp.gray=new Mat();
                cvtColor(image, *(tmp.gray),CV_BGR2GRAY); 
                tmp.R=R.clone(); 
                tmp.T=T.clone(); 
                tmp.reg2d=1; 
                tmp.reg3d=1; 
                tmp.gt=1; //rough+fine tracked and ground truth
            }
            *newFp = tmp;
        }
        
        
        
        //Construct pyramid
        newFp->pyramid = new vector<Mat>;
        createPyramid(*newFp->gray,*newFp->pyramid,pyrLevels);
        
        //Add frame to appropriate queues (last because exposes frame to other code)
        {
            //pushes have side effects, order them properly!
            fs   .push( newFp);//is a frame
            utrkd.push( newFp);//is rough tracked
            mtrkd.push( newFp);//is fine tracked
            trkd. push( newFp);//is tracked
        }
    }

    FrameID addFrame(Mat image){
        if(!initd||fn<2){
            CV_Error(CV_StsAssert, "OpenDTAM not inited properly (Did you add two posed frames yet?) before calling addFrame.");
        }
        //Increment frame counter
        FrameID fid = fn++;
        
        //Construct new frame
        Fp newFp(new Frame);
        {
            Frame tmp;
            //Fill known fields
            {
                tmp.fid=fid; 
                tmp.im=new Mat(image);
                tmp.gray=new Mat();
                cvtColor(image, *tmp.gray,CV_BGR2GRAY); 
            }
            *newFp = tmp;
        }
        
        //Construct pyramid
        newFp->pyramid = new vector<Mat>;
        createPyramid(*newFp->gray,*newFp->pyramid,pyrLevels);
        
        //Add frame to appropriate queues (last because exposes frame to other code)
        {
            //pushes have side effects, order them properly!
            fs   .push( newFp);//is a frame
            utrkq.push( newFp);//needs rough tracking
        }
    }

    void createPyramid(const Mat& image,vector<Mat>& pyramid,int& levels){
        Mat in=image;
        int l2=levels-1;
        pyramid.resize(levels);
        pyramid[l2--]=in;
        
        for (float scale=0.5; l2>=0; scale/=2, l2--) {
            Mat out;
            resize(in,out,Size(),.5,.5,CV_INTER_AREA);
            pyramid[l2]=out;
            in=out;
        }
        
    }
    

bool utrk(Ptr<Frame> _frame){
    Frame& frame=*_frame;
    
    //Find the last frame before this one
    Ptr<Frame> lfp;
    {//try for previous frame, if that doesn't work, use last utrkd frame 
        {
        ScopeLock s(fs.mutex);
        assert(frame.fid-1>=0);
        lfp=fs.q[frame.fid-1];
        }
        assert(lfp->reg3d);//this must be true for relative poses to make sense, remove assert if don't care.
        if(!lfp->reg3d){
        ScopeLock s(utrkd.mutex);
        lfp=utrkd.q.back();
        }
    }
    Frame & lf = *lfp;
    
    // Find a depth map to track from
    Ptr<Frame> basep;
    {
        ScopeLock s(ucvd.mutex);
        if(ucvd.q.size()==0){//no ucv yet! might be at init
            cout<<"OpenDTAM Tracking Starting."<<endl;
            return 0;
        }
        basep = ucvd.q.back();
    }
    Frame & base = *basep;
    
    Mat p2d=Mat::zeros(1,6,CV_64FC1);
    
    //Do 2D tracking
    static Mat Z(lf.im->rows,lf.im->cols,CV_32FC1);
    for (int level=utrkLevel2DStart; level<utrkLevel2DEnd; level++){
        int iters=3;
        for(int i=0;i<iters;i++){
            //HACK: use 3d alignment with depth disabled for 2D. ESM would be much better, but I'm lazy right now.
            bool success = align_level_largedef_gray_forward(   (*lf.pyramid)[level],//Total Mem cost ~185 load/stores of image
                                                                Mat(),
                                                                (*frame.pyramid)[level],
                                                                cameraMatrixPyr[level],//Mat_<double>
                                                                p2d,                //Mat_<double>
                                                                CV_DTAM_FWD,
                                                                .7,
                                                                3);
            
            if (!success){
                cout<<"Restarting ODM Tracking"<<endl;
                return 0;
            }
        }
    }
    frame.relPose2d=p2d;
    frame.reg2d=1;
    
    Mat pbase,plf,p;
    RTToLie(lf.R, lf.T, plf);//plf has global pose of lf
    RTToLie(base.R, base.T,pbase);//pbase has global pose of base
    p=LieAdd(p2d,plf);//p has global pose of frame
    p=LieSub(p,pbase);//p has relative pose to base

    //Do 3D tracking
    for (int level=utrkLevel3DStart; level<utrkLevel3DEnd; level++){
        float thr = (utrkLevel3DEnd-level)>=2 ? .05 : .2; //more stringent matching on last two levels 
        int iters=3;
        for(int i=0;i<iters;i++){
            bool success = align_level_largedef_gray_forward(   (*base.pyramid)[level],//Total Mem cost ~185 load/stores of image
                                                                base.optimizer->getBestDepthSoFar(),
                                                                (*frame.pyramid)[level],
                                                                cameraMatrixPyr[level],//Mat_<double>
                                                                p,                //Mat_<double>
                                                                CV_DTAM_FWD,
                                                                thr,
                                                                6);
            if (!success){
                cout<<"Restarting ODM Tracking"<<endl;
                return 0;
            }
        }
    }
    
    frame.relPose3d=p;
    p=LieAdd(p,pbase);//p has global pose
    LieToRT(p,frame.R,frame.T);
    frame.reg3d=1;
    
    return 1;
}

cv::cuda::CudaMem imageContainerUcv;
bool ucv(Ptr<Frame> _base,Ptr<Frame> _alt){
    Frame& base=*_base;
    Frame& alt=*_alt;
    Mat ucameraMatrix=this->cameraMatrix.clone();
    Mat b,a;
    
    Size2d s0=base.im->size();
    resize(*base.im,b,ucvSize);
    resize(*alt.im,a,ucvSize);
    Size2d sn=b.size();
    cout<<sum(a)<<sum(b)<<endl;
    double sx=(sn.width/s0.width);
    double sy=(sn.height/s0.height);
    ucameraMatrix+=(Mat)(Mat_<double>(3,3) << 0,0.0,0.5,
                                        0.0,0.0,0.5,
                                        0.0,0.0,0);
    ucameraMatrix=ucameraMatrix.mul((Mat)(Mat_<double>(3,3) <<    sx,0.0,sx,
                                            0.0,sy ,sy,
                                            0.0,0.0,1.0));
    ucameraMatrix-=(Mat)(Mat_<double>(3,3) << 0,0.0,0.5,
                                        0.0,0.0,0.5,
                                        0.0,0.0,0);

    Ptr<CostVolume> cvp(new CostVolume(b,base.fid,32,0.015,0.0,base.R,base.T,ucameraMatrix));
    CostVolume& cv=*cvp;
    
    imageContainerUcv.create(a.rows,a.cols,CV_8UC4);
    Mat tmp,ret;
    cvtColor(a,tmp,CV_RGB2RGBA);
    Mat imageContainerRef=imageContainerUcv;//Required by ambiguous conversion rules
    tmp.convertTo(imageContainerRef,CV_8UC4,255.0);
    cv.updateCost(imageContainerUcv, alt.R, alt.T);


    
    Ptr<DepthmapDenoiseWeightedHuber> optimizerp(new DepthmapDenoiseWeightedHuber(cv));
    DepthmapDenoiseWeightedHuber& optimizer=*optimizerp;
//     optimizer.thetaStart =    20.0;
//     optimizer.thetaMin=0.01;
//     optimizer.thetaStep=.99;
    optimizer.initOptimization();
//     gpause();
    bool doneOptimizing;
    do{ 
        cout<<"Theta: "<< optimizer.getTheta()<<endl;
        optimizer._a.download(ret);
        pfShow("uA", ret, 0, cv::Vec2d(0, 32));

//         optimizer.cacheGValues();
//         optimizer._gy.download(ret);
//         pfShow("G function", ret, 0, cv::Vec2d(0, 1));
//         gpause();
        for (int i = 0; i < 10; i++) {
            optimizer.optimizeQD();
//             cudaDeviceSynchronize();
//             optimizer._qx.download(ret);
//             pfShow("uQx function", ret, 0, cv::Vec2d(-1, 1));
//             optimizer._gy.download(ret);
//             pfShow("uGy function", ret, 0, cv::Vec2d(0, 1));
            optimizer._d.download(ret);
            pfShow("uD function", ret, 0, cv::Vec2d(0, 32));
        }
//         cudaDeviceSynchronize();
        doneOptimizing=optimizer.optimizeA();
    }while(!doneOptimizing);   
    optimizer.cvStream.waitForCompletion();
    base.cv=cvp;
    base.optimizer=optimizerp;
    ucvd.push(_base);
    cvd.push(_base);
//     gpause();
    return true;
}


void Tutrk(int* stop){
    cout<<"Thread Launched "<< stop << endl;
    while(!*stop){
        Ptr<Frame> myFrame=utrkq.pop();
        if(!utrk(myFrame)){
            utrkq.readStall();//prevent others from wasting time trying to track until got a new cv
            std::vector<Ptr<Frame> > frames=utrkd.peekn(2);
            ucv(frames[0],frames[1]);
            if(!utrk(myFrame)){//try again
                CV_Error(CV_StsAssert, "Tracking Lost! Could not recover. Quitting.");
            }
            utrkq.readUnstall();
        }
        utrkd.push(myFrame);
        trkd.push(myFrame);
        mtrkq.push(myFrame);
    }
}
};


#endif










































