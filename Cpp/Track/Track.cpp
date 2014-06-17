#include "Track.hpp"
#include <Cost.h>
#include "utils/utils.h"
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
void Track::addFrame(cv::Mat frame){
    lastFrame=thisFrame;
    thisFrame=frame;
}