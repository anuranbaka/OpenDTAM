#include <opencv2/core/core.hpp>
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <fstream>



//Mine
#include "convertAhandaPovRayToStandard.h"
#include "CostVolume/utils/reproject.hpp"
#include "CostVolume/utils/reprojectCloud.hpp"
#include "CostVolume/Cost.h"
#include "CostVolume/CostVolume.hpp"
#include "Optimizer/Optimizer.hpp"
#include "DepthmapDenoiseWeightedHuber/DepthmapDenoiseWeightedHuber.hpp"
// #include "OpenDTAM.hpp"
#include "graphics.hpp"
#include "set_affinity.h"
#include "Track/Track.hpp"

#include "utils/utils.hpp"


//debug
#include "tictoc.h"





const static bool valgrind=0;

//A test program to make the mapper run
using namespace cv;
using namespace cv::gpu;
using namespace std;

int App_main( int argc, char** argv );

void myExit(){
    ImplThread::stopAllThreads();
}
int main( int argc, char** argv ){

    initGui();

    int ret=App_main(argc, argv);
    myExit();
    return ret;
}


int App_main( int argc, char** argv )
{
    srand(time(NULL));
    rand();
    rand();
    cv::theRNG().state = rand();
    int numImg=500;

#if !defined WIN32 && !defined _WIN32 && !defined WINCE && defined __linux__ && !defined ANDROID
    pthread_setname_np(pthread_self(),"App_main");
#endif

    char filename[500];
    Mat image, cameraMatrix, R, T;
    vector<Mat> images,Rs,Ts,Rs0,Ts0;
    Mat ret;//a place to return downloaded images to
    
    ofstream file("outscale.csv");
    double reconstructionScale=5/5.;
    int inc=1;
    for(int i=0;i>0||inc>0;i+=inc){
        Mat tmp;
        sprintf(filename,"../../Trajectory_30_seconds/scene_%03d.png",i);
        convertAhandaPovRayToStandard("../../Trajectory_30_seconds",
                                      i,
                                      cameraMatrix,
                                      R,
                                      T);
        Mat image;
        cout<<"Opening: "<< filename << endl;
        if(inc>0){
        imread(filename, -1).convertTo(image,CV_32FC3,1.0/65535.0);
        resize(image,image,Size(),reconstructionScale,reconstructionScale);
        
        images.push_back(image.clone());
        }
        else
        {
            images.push_back(images[i]);
        }
        Rs.push_back(R.clone()*Mat::eye(3,3,R.type()));
        Ts.push_back(T.clone()*0.0);
        Rs0.push_back(R.clone());
        Ts0.push_back(T.clone());
        if(i==numImg-1)
            inc=-1;
    }
    numImg=numImg*2-2;
    cout<<LieSub(RTToLie(Rs[0],Ts[0]),RTToLie(Rs[1],Ts[1]))<<endl;
    randu(Ts[1] ,Scalar(-1),Scalar(1));
    Ts[1]=Ts[0]+Ts[1];
    cout<<Ts[1]-Ts[0]<<endl;
    Rs[1]=Rs[0].clone();
    CudaMem cret(images[0].rows,images[0].cols,CV_32FC1);
    ret=cret.createMatHeader();
    //Setup camera matrix
    double sx=reconstructionScale;
    double sy=reconstructionScale;
    cameraMatrix+=(Mat)(Mat_<double>(3,3) <<    0.0,0.0,0.5,
                                                0.0,0.0,0.5,
                                                0.0,0.0,0.0);
    cameraMatrix=cameraMatrix.mul((Mat)(Mat_<double>(3,3) <<    sx,0.0,sx,
                                                                0.0,sy ,sy,
                                                                0.0,0.0,1.0));
    cameraMatrix-=(Mat)(Mat_<double>(3,3) <<    0.0,0.0,0.5,
                                                0.0,0.0,0.5,
                                                0.0,0.0,0);
    int layers=256;
    int imagesPerCV=1;
    int desiredImagesPerCV=500;
    CostVolume cv(images[0],(FrameID)0,layers,0.015,0.0,Rs[0],Ts[0],cameraMatrix);;

//     //New Way (Needs work)
//     OpenDTAM odm(cameraMatrix);
//     odm.addFrameWithPose(images[0],Rs[0],Ts[0]);
//     odm.addFrameWithPose(images[10],Rs[10],Ts[10]);
//     for (int imageNum=2;imageNum<=numImg;imageNum++){
//         odm.addFrame(images[imageNum]);
//         usleep(100000);
//     }
    
    //Old Way
    int imageNum=0;
    
    
    cv::gpu::Stream s;
    double totalscale=1.0;
    int tcount=0;
    for (int imageNum=1;imageNum<numImg;imageNum=(++imageNum)%numImg){

        T=Ts[imageNum].clone();
        R=Rs[imageNum].clone();
        image=images[imageNum];

        if(cv.count<imagesPerCV){
            
            cv.updateCost(image, R, T);
            cudaDeviceSynchronize();
//             gpause();
//             for( int i=0;i<layers;i++){
//                 pfShow("layer",cv.downloadOldStyle(i), 0, cv::Vec2d(0, .5));
//                 usleep(1000000);
//             }
        }
        else{
            cudaDeviceSynchronize();
            //Attach optimizer
            Ptr<DepthmapDenoiseWeightedHuber> dp = createDepthmapDenoiseWeightedHuber(cv.baseImageGray,cv.cvStream);
            DepthmapDenoiseWeightedHuber& denoiser=*dp;
            Optimizer optimizer(cv);
            optimizer.initOptimization();
            GpuMat a(cv.loInd.size(),cv.loInd.type());
//             cv.loInd.copyTo(a,cv.cvStream);
            cv.cvStream.enqueueCopy(cv.loInd,a);
            GpuMat d;
            denoiser.cacheGValues();
            ret=image*0;
//             pfShow("A function", ret, 0, cv::Vec2d(0, layers));
//             pfShow("D function", ret, 0, cv::Vec2d(0, layers));
//             pfShow("A function loose", ret, 0, cv::Vec2d(0, layers));
//             pfShow("Predicted Image",ret,0,Vec2d(0,1));
//             pfShow("Actual Image",ret);
            
            cv.loInd.download(ret);
            pfShow("loInd", ret, 0, cv::Vec2d(0, layers));
//                waitKey(0);
//                gpause();
            
            

            bool doneOptimizing; int Acount=0; int QDcount=0;
            do{
//                 cout<<"Theta: "<< optimizer.getTheta()<<endl;
//
//                 if(Acount==0)
//                     gpause();
//                a.download(ret);
//                pfShow("A function", ret, 0, cv::Vec2d(0, layers));
                
                

                for (int i = 0; i < 10; i++) {
                    d=denoiser(a,optimizer.epsilon,optimizer.getTheta());
                    QDcount++;
                    
//                    denoiser._qx.download(ret);
//                    pfShow("Q function:x direction", ret, 0, cv::Vec2d(-1, 1));
//                    denoiser._qy.download(ret);
//                    pfShow("Q function:y direction", ret, 0, cv::Vec2d(-1, 1));
//                    d.download(ret);
//                    pfShow("D function", ret, 0, cv::Vec2d(0, layers));
                }
                doneOptimizing=optimizer.optimizeA(d,a);
                Acount++;
            }while(!doneOptimizing);
//             optimizer.lambda=.05;
//             optimizer.theta=10000;
//             optimizer.optimizeA(a,a);
            optimizer.cvStream.waitForCompletion();
            a.download(ret);
               pfShow("A function loose", ret, 0, cv::Vec2d(0, layers));
//                gpause();
//             cout<<"A iterations: "<< Acount<< "  QD iterations: "<<QDcount<<endl;
//             pfShow("Depth Solution", optimizer.depthMap(), 0, cv::Vec2d(cv.far, cv.near));
//             imwrite("outz.png",ret);

            Track tracker(cv);
            Mat out=optimizer.depthMap();
            double m;
            minMaxLoc(out,NULL,&m);
            m=mean(out)[0]*3;

            double sf=(.66*cv.near/m);
            tracker.depth=out;
           
            imageNum=((imageNum-imagesPerCV)%numImg+numImg)%numImg;
            assert(imageNum>=0);
//             if (imageNum>5)
//                 if(imagesPerCV==1)
            if (tcount>10)
                    imagesPerCV=desiredImagesPerCV;
//                 else
//                     imagesPerCV=1;

            for(int i0=0;i0<=imagesPerCV;i0++){
                int i=(imageNum+i0)%numImg;
                tracker.addFrame(images[i]);
                if(!tracker.align())
                    imagesPerCV=max(abs(i-imageNum)-1,1);
                LieToRT(tracker.pose,R,T);
                Rs[i]=R.clone();
                Ts[i]=T.clone();
                
                Mat p,tp;
                p=tracker.pose;
                tp=RTToLie(Rs0[i],Ts0[i]);
                {//debug
                    cout << "True Pose: "<< tp << endl;
                    cout << "True Delta: "<< LieSub(tp,tracker.basePose) << endl;
                    cout << "Recovered Pose: "<< p << endl;
                    cout << "Recovered Delta: "<< LieSub(p,tracker.basePose) << endl;
                    cout << "Pose Error: "<< p-tp << endl;
                }
                cout<<i<<endl;
                reprojectCloud(images[i],images[cv.fid],tracker.depth,RTToP(Rs[cv.fid],Ts[cv.fid]),RTToP(Rs[i],Ts[i]),cameraMatrix);
            }
            cv=CostVolume(images[imageNum],(FrameID)imageNum,layers,cv.near/sf,0.0,Rs[imageNum],Ts[imageNum],cameraMatrix);
            totalscale*=sf;
            file<<sf<<", "<<endl;
//             file.sync_with_stdio();
            if(tcount==7){
                totalscale=1.0f;
            }
            tcount++;
            cout<<"CV #: "<<tcount<<" Total Scale: "<<totalscale<<endl;
            s=optimizer.cvStream;
//             for (int imageNum=0;imageNum<numImg;imageNum=imageNum+1){
//                 reprojectCloud(images[imageNum],images[0],optimizer.depthMap(),RTToP(Rs[0],Ts[0]),RTToP(Rs[imageNum],Ts[imageNum]),cameraMatrix);
//             }
            a.download(ret);
            
        }
        s.waitForCompletion();// so we don't lock the whole system up forever
    }
    exit:
    s.waitForCompletion();
    Stream::Null().waitForCompletion();
    return 0;
}


