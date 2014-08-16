#ifndef FRAME_HPP
#define FRAME_HPP

struct Frame{
    //All info associated with a frame
    //Data that can be discarded eventually 
    FrameID fid;
    Ptr<Mat> im;
    Ptr<Mat> gray;
    Mat R;
    Mat T;
    
    Mat relPose2d;
    Mat relPose3d;
    
//     //Ptr<Mat> CameraMatrix;
    int reg2d;//2d registered?
    int reg3d;//3d registered
    int gt;//ground truth
//     Mat pose;//the global pose
    Ptr<Frame> parent;//the frame whose cv was used for registration
//     vector<Ptr<CostVolume>> consumers;//the cost volumes using this image as inputs (non-base)
    Ptr<CostVolume> cv;// the cost volume(if any) that uses this image as the base image
    Ptr<DepthmapDenoiseWeightedHuber> optimizer;
//     Tracker tracker;
//     vector<Mat> pyramid;
    Ptr<vector<Mat> > pyramid;
    Frame():fid(0),im(),gray(),R(),T(),relPose2d(),relPose3d(),reg2d(0),reg3d(0),gt(0),parent(),cv(),optimizer(),pyramid(){}
};

#endif