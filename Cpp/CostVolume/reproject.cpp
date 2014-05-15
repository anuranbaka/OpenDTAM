//in Cost.cpp
void reproject( const Mat_<double>& src
                const Matx33d& cameraMatrix,
                const Matx44d& cameraAffinePoseBase,
                const Matx44d& cameraAffinePoseAlternate,
                float e,
                Mat_<double>& dst,
                Mat_<double>& mask){//e is inverse depth
           
    Mat_<double> cA_Apln(Mat_<double>(3,4) << cameraMatrix.inv() << 0,0,e);
    Mat_<double> cW_A=cameraAffinePoseAlternate.inv();//optimizable
    Mat_<double> & cB_W=cameraAffinePoseBase;
    Mat_<double> & cBpln_B;
    hconcat(cameraMatrix,(Mat_<double>(3,1) << 0,0,0), cBpln_B);// the usual cam matrix operating on 
    Mat_<double> cBpln_Apln=cBpln_B*cB_W*cW_A*cA_Apln;
    Mat_<double> dst;
    warpPerspective(src, dst, cBpln_Apln, src.size(), );
    mask = (dst == 0);
}

