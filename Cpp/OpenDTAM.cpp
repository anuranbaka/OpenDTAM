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
#include "Optimizer/Optimizer.hpp"
#include "graphics.hpp"
#include "set_affinity.h"
#include "Track/Track.hpp"
#include "utils/utils.hpp"
#include "Scheduler/Frame.hpp"
#include "Scheduler/SynchronizedBuffer.hpp"


//debug
#include "tictoc.h"

using namespace cv;
using namespace std;
using namespace cv::gpu;
class OpenDTAM{

StallableSynchronizedStack<Ptr<Frame> > fs;//the frame stream
StallableSynchronizedQueue<Ptr<Frame> > utrkq;//frames needing fast tracking
StallableSynchronizedStack<Ptr<Frame> > utrkd;//frames that have been tracked at rough level
StallableSynchronizedQueue<Ptr<Frame> > trkq;//frames needing fine tracking
StallableSynchronizedStack<Ptr<Frame> > mtrkd;//frames that have been tracked at fine level
StallableSynchronizedStack<Ptr<Frame> > trkd; //frames that have been tracked at all levels

StallableSynchronizedQueue<std::pair<Ptr<Frame>,std::vector<Ptr<Frame> > > > ucvq; //frames to convert to low quality cost volumes ("micro cv's")
StallableSynchronizedStack<Ptr<Frame> > ucvs;//micro cv's (stack)
StallableSynchronizedStack<Ptr<Frame> > mcvs;//macro cv's (stack)
StallableSynchronizedStack<Ptr<Frame> > cvs; //all cv's  (stack)

FrameID fn;
public:
    
OpenDTAM():fn(0){}

typedef Ptr<Frame> Fp;
FrameID addFrameWithPose(Mat image, Mat R, Mat T){
    FrameID fid = fn++;
    Frame tmp={};
    tmp.fid=fid; tmp.im=new Mat(image); tmp.R=R; tmp.T=T; tmp.reg2d=1; tmp.reg3d=1; tmp.gt=1; //rough+fine tracked and ground truth
    Fp newFp(new Frame);
    *newFp = tmp;
    fs.push(    newFp);//is a frame
    utrkd.push( newFp);//is rough tracked
    mtrkd.push( newFp);//is fine tracked
    trkd.push(  newFp);//is tracked
    
}

FrameID addFrame(Mat image){
    FrameID fid = fn++;
    Frame tmp={};
    tmp.fid=fid;tmp.im=new Mat(image); 
    Fp newFp(new Frame);
    *newFp = tmp;
    fs.push(    newFp);//is a frame
    utrkq.push( newFp);//is rough tracked 
}

};













































