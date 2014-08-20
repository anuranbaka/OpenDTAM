#include "Track.hpp"
#include "utils/utils.hpp"
using namespace cv;
using namespace std;
Track::Track(Cost cost){
    rows=cost.rows;
    cols=cost.cols;
    baseImage=lastFrame=thisFrame=cost.baseImage;
    cameraMatrix=Mat(cost.cameraMatrix);
    depth=cost.depthMap();
    PToLie(Mat(cost.pose),basePose);
    pose=basePose.clone();

}
Track::Track(CostVolume cost){
    rows=cost.rows;
    cols=cost.cols;
    cost.baseImage.download(thisFrame);
    baseImage=lastFrame=thisFrame;
    cameraMatrix=Mat(cost.cameraMatrix);
    RTToLie(cost.R,cost.T,basePose);
    pose=basePose.clone();

}
void Track::addFrame(cv::Mat frame){
    lastFrame=thisFrame;
    thisFrame=frame;
}
