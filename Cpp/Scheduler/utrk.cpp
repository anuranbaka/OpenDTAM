//thread for small high speed tracking

//parameter: level to utrack to 
Tutrk{
    registerThread();
    while(1){
        Ptr<Frame> myframe=utrkq.pop();
        if(!utrk(myFrame)){
            utrkq.readStall();//prevent others from wasting time trying to track until got a new cv
            std::vector<Ptr<Frame> > frames=utrkd.peekn(2);
            Ptr<CostVolume> newUcv=ucv(&(frames[0]),&(frames[1]));
            ucvs.push(newUcv);
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
}
