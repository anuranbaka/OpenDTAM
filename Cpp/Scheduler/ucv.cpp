//parameter: level to ucv to
#include "Frame.hpp"
Tucv{
    registerThread();
    while(1){
        wait for ucv
        lock(ucvq);
            if(ucvq.size()>0){//no other thread got it yet
                [frame1,frame2]=pop(ucvq);//ucvs use two frames to build a cv as fast as possible
            }
            else{
                continue;//someone else dealt with it(What? How can this happen?)
            }
        release(ucvq);
        if(!(myUcv=ucv(frame1,frame2))){
            error! couldn't build a ucv! What? how can ucv fail!?
        }
        lock(ucvd);
            ucvd.push(myUcv);
        release(ucvd);
    }
}

cv::gpu::CudaMem imageContainer;

Ptr<CostVolume> ucv(Frame& base,Frame& alt){
    Mat cameraMatrix=this->cameraMatrix.clone();
    gpuMat b,a;
    
    Size2d s0=base.im.size();
    resize(base.im,b,Size(),sc,sc);
    resize(alt.im,a,Size(),sc,sc);
    Size2d sn=b.size();
    double sx=(sn.width/s0.width);
    double sy=(sn.height/s0.height);
    cameraMatrix+=(Mat)(Mat_<double>(3,3) << 0,0.0,0.5,
                                        0.0,0.0,0.5,
                                        0.0,0.0,0);
    cameraMatrix=cameraMatrix.mul((Mat)(Mat_<double>(3,3) <<    sx,0.0,sx,
                                            0.0,sy ,sy,
                                            0.0,0.0,1.0));
    cameraMatrix-=(Mat)(Mat_<double>(3,3) << 0,0.0,0.5,
                                        0.0,0.0,0.5,
                                        0.0,0.0,0);
        
    Ptr<CostVolume> cvp(new CostVolume(b,base.fid,32,0.015,0.0,base.R,base.T,cameraMatrix));
    CostVolume& cv=*cvp;
    imageContainer.create(a.rows,a.cols,CV_8UC4);
    Mat tmp,ret;
    cvtColor(b,tmp,CV_RGB2RGBA);
    Mat imageContainerRef=imageContainer;//Required by ambiguous conversion rules
    tmp.convertTo(imageContainerRef,CV_8UC4,255.0);
    cv.updateCost(imageContainer, R, T);
    DepthmapDenoiseWeightedHuber optimizer(cv);
    optimizer.initOptimization();
    
    bool doneOptimizing;
    do{
//         cout<<"Theta: "<< optimizer.theta<<endl;
//         optimizer._a.download(ret);
//         pfShow("One A Opt Soln", ret, 0, cv::Vec2d(0, 32));

//         optimizer.cacheGValues();
//         optimizer._gy.download(ret);
//         pfShow("G function", ret, 0, cv::Vec2d(0, 1));
//         gpause();
        for (int i = 0; i < 10; i++) {
            optimizer.optimizeQD();
//             cudaDeviceSynchronize();
//             optimizer._qx.download(ret);
//             pfShow("Qx function", ret, 0, cv::Vec2d(-1, 1));
//             optimizer._gy.download(ret);
//             pfShow("Gy function", ret, 0, cv::Vec2d(0, 1));
//             optimizer._d.download(ret);
//             pfShow("D function", ret, 0, cv::Vec2d(0, 32));
//             gpause();
            
        }
//         cudaDeviceSynchronize();
        doneOptimizing=optimizer.optimizeA();
    }while(!doneOptimizing);
    optimizer.cvStream.waitForCompletion();
    return cvp;
}
    
