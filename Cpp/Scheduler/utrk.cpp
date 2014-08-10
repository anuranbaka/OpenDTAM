//thread for small high speed tracking

//parameter: level to utrack to 
void* Tutrk(void*){
    while(1){
        Ptr<Frame> myframe=utrkq.pop();
        if(!utrk(myFrame)){
            utrkq.readStall();//prevent others from wasting time trying to track until got a new cv
            std::vector<Ptr<Frame> > frames=utrkd.peekn(2);
            Ptr<Frame> newUcvFrame=ucv(&(frames[0]),&(frames[1]));
            ucvd.push(newUcvFrame);
            if(!utrk(myFrame)){//try again
                error! couldn't recover tracking
            }
            utrkq.readUnstall();
	    }
	    lock(utrkd);
	        utrkd.push(myFrame);
        release(utrkd);
	    lock(trkq);
	        trkq.push(myFrame);
        release(trkq);
        signal trk to run;
    }
    return 0;
}

bool utrk(Frame& frame){
    
    
    
    
    //Find the last frame before this one
    Ptr<Frame> lfp;
    {
        fs.mutex.lock();
        assert(frame.fid-1>=0)
        lfp=fs.q[frame.fid-1];
        fs.mutex.unlock();
    }
    Frame & lf = *lfp;
    
    // Find a depthmap to track from
    Ptr<Frame> basep;
    {
        ucvd.mutex.lock();
        if(ucvd.q.size()==0)//no ucv yet! might be at init
            return 0;
        basep = ucvd.q.back();
        ucvd.mutex.unlock();
    }
    Frame & base = *basep;
    
    Mat p=Mat::zeros(1,6,CV_64FC1);
    Mat p2d=Mat::zeros(1,6,CV_64FC1);
    
    //Do 2D tracking
    static Mat Z(lf.im->rows,lf.im->cols,CV_32FC1);
    for (level=utrkLevel2DStart; level<=utrkLevel2DEnd; level++){
        int iters=1;
        for(int i=0;i<iters;i++){
            //HACK: use 3d alignment with depth disabled for 2D. ESM would be much better, but I'm lazy right now.
            bool success = align_level_largedef_gray_forward(   (*lf.grayPyr)[level],//Total Mem cost ~185 load/stores of image
                                                                Z,
                                                                (*frame.grayPyr)[level],
                                                                cameraMatrixPyr[level],//Mat_<double>
                                                                p2d,                //Mat_<double>
                                                                CV_DTAM_FWD,
                                                                1,
                                                                3);
            if (!success)
                return 0;
        }
    }
    p=LieAdd(p2d,p);

    //Do 3D tracking
    for (level=utrkLevel3DStart; level<=utrkLevel3DEnd; level++){
        int iters=1;
        for(int i=0;i<iters;i++){
            bool success = align_level_largedef_gray_forward(   (*base.grayPyr)[level],//Total Mem cost ~185 load/stores of image
                                                                base.optimizer.depthMap(),
                                                                (*frame.grayPyr)[level],
                                                                cameraMatrixPyr[level],//Mat_<double>
                                                                p,                //Mat_<double>
                                                                CV_DTAM_FWD,
                                                                1,
                                                                3);
            if (!success)
                return 0;
        }
    }
    frame.relPose2d=p2d;
    frame.relPose3d=p;
    Lie2RT(p,frame.R,frame.T);
    
}
