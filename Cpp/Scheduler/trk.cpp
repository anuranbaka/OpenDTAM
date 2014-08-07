//parameter: level to track to

Ttrk{
    registerThread();
    while(1){
        wait for trk
        lock(trkq);
            if(trkq.size()>0){//no other thread got it yet, trkq cannot stall
                myFrame=pop(trkq);
            }
            else{
                continue;//someone else dealt with them
            }
        release(trkq);
        if(!trk(myFrame)){
            error! high quality tracking failed!
	    }
	    lock(trkd);
	        trkd.push(myFrame);
        release(trkd);
    }
}
