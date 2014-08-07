#ifndef FRAME_HPP
#define FRAME_HPP

struct Frame{
    FrameID fid;
    Ptr<Mat> im;
    Mat R;
    Mat T;
    
    Mat relPose2d;
    Mat relPose3d;
//     //Ptr<Mat> CameraMatrix;
    int reg2d;//2d registered?
    int reg3d;//3d registered
    int gt;//ground truth
//     Mat pose;//the global pose
//     Ptr<CostVolume> parent;//the cost volume whose cv was used for registration
//     vector<Ptr<CostVolume>> consumers;//the cost volumes using this image as inputs (non-base)
//     Ptr<CostVolume> child;// the cost volume(if any) that uses this image as the base image
//     Optimizer optimizer;
//     Tracker tracker;
//     vector<Mat>& pyramid;
    
};

#endif