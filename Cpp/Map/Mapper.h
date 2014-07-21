// Free for non-commercial, non-military, and non-critical 
// use unless incorporated in OpenCV. 
// Inherits OpenCV Licence if in OpenCV.

#ifndef MAPPER_H
#define MAPPER_H

// Mapper has slots for cost volumes, and an optimizer
// Mapper exposes a depth map from the current cost volume
// Mapper is otherwise a thin shell, does not understand the existance 
// of multiple keyframes/maps. Need a DTAM object to 




class Mapper
{
    int threadId;//the unique id of the thread this is being used by. 0 implies no threading
    int stop();//setting this to true will tell internal components to stop
    int running();//returns whether this or any subcomponents are running
    void pushFrame(Mat Image,Mat R,Mat T);//adds a frame to the framelist
    void pushFrame(Frame3D frame);
    void consumeFrames();//offers the queued frames to the current cost volume if hungry
    void setKeyFrame(Mat Image, Mat R, Mat T);//sets the current cost volume's keyframe
    void addKeyFrame();
    void optimize();//optimizes the current cost volume
    CostVolume costVolume();//return the active cost volume
    MapOptimizer optimizer();//return the current optimizer
    Mat_<float> depthMap();//return the best depth map so far
    
    
};

#endif // MAPPER_H
