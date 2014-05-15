//in Cost.cpp
void updateCost(Mat& image, 
                Matx44d& currentCameraPose)
{
    
    for(int n=0; n < depth.size(); ++n){
        Mat_<PixelType> plane;
        Mat mask;
        reproject(image, cameraMatrix, pose, currentCameraPose, depth[n], plane, mask);
        
        //
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                if(mask(i,j)){//valid point
                    data[n+j*layers+i*layers*cols]=photoCost(plane(i,j),baseImage(i,j));
                    hits(i,j)++;
                }
            }
        }
    }
    
    
}
