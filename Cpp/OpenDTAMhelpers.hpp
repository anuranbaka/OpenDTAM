void createPyramid(const Mat& image,vector<Mat>& pyramid,int& levels){
    
    Mat in=image;
    if(levels==0){//auto size to end at >=15px tall (use height because shortest dim usually)
        for (float scale=1.0; scale>=15.0/image.rows; scale/=2, levels++);
    }
    assert(levels>0);
    int l2=levels-1;
    pyramid.resize(levels);
    pyramid[l2--]=in;
    
    for (float scale=0.5; l2>=0; scale/=2, l2--) {
        Mat out;

        
        resize(in,out,Size(),.5,.5,CV_INTER_AREA);
        pyramid[l2]=out;
        in=out;
    }
    
}