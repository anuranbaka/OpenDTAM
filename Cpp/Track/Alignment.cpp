/*
 *  Align_part.cpp
 *  
 *
 *
 */
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#include "utils/utils.hpp"







const static float FAIL_FRACTION=0.30;

enum alignment_modes{CV_DTAM_REV,CV_DTAM_FWD,CV_DTAM_ESM};
const double small0=.1;//~6deg, not trivial, but hopefully enough to make the translation matter

static void getGradient(const Mat& image,Mat & grad);





static Mat paramsToProjection(const Mat & p,const Mat& _cameraMatrix){
    //Build the base transform
    assert(p.type()==CV_64FC1);
    Mat dR=rodrigues(p.colRange(Range(0,3)));
    Mat dT=p.colRange(Range(3,6)).t();
    Mat dA;
    hconcat(dR,dT,dA);
    dA=make4x4(dA);
    Mat cameraMatrix=make4x4(_cameraMatrix);
    assert(cameraMatrix.type()==CV_64FC1);
    Mat proj=cameraMatrix*dA*cameraMatrix.inv();
//     cout<<"p: "<<"\n"<< p<< endl;
//     cout<<"Proj: "<<"\n"<< proj<< endl;
    //The column swap
    Mat tmp=proj.colRange(2,4).clone();
    tmp.col(1).copyTo(proj.col(2));
    tmp.col(0).copyTo(proj.col(3));
    //The row drop
    proj=proj.rowRange(0,3);
    return proj;
}



static void getGradient(const Mat& image,Mat & grad){
    //Image gradients for alignment
    //Note that these gradients have theoretical problems under the sudden 
    // changes model of images. It might be wise to blur the images before 
    // alignment, to avoid sudden changes, but that makes occlusion more 
    // problematic.
    grad.create(2,image.rows*image.cols,CV_32FC1);
    Mat gray;
    if (image.type()==CV_32FC1) {
        gray=image;
    }else {
        cvtColor(image, gray, CV_BGR2GRAY);
        gray.convertTo(gray,CV_32FC1);
    }
    Mat grad_x(image.rows,image.cols,CV_32FC1,grad.row(0).data);
    Scharr( gray, grad_x, CV_32FC1, 1, 0, 1.0/26.0, 0, BORDER_REPLICATE );
    Mat grad_y(image.rows,image.cols,CV_32FC1,grad.row(1).data);
    Scharr( gray, grad_y, CV_32FC1, 0, 1, 1.0/26.0, 0, BORDER_REPLICATE);
}

static void getGradient_8(const Mat& image,Mat & grad){
    //Image gradients for alignment
    //Note that these gradients have theoretical problems under the sudden 
    // changes model of images. It might be wise to blur the images before 
    // alignment, to avoid sudden changes, but that makes occlusion more 
    // problematic.
    grad.create(2,image.rows*image.cols,CV_32FC1);
    Mat gray;
    if (image.type()==CV_32FC1) {
        gray=image;
    }else {
        cvtColor(image, gray, CV_BGR2GRAY);
        gray.convertTo(gray,CV_32FC1);
    }
    Mat grad_x(image.rows,image.cols,CV_32FC1,grad.row(0).data);
    Scharr( gray, grad_x, CV_32FC1, 1, 0, 1.0/26.0, 0, BORDER_REPLICATE );
    Mat grad_y(image.rows,image.cols,CV_32FC1,grad.row(1).data);
    Scharr( gray, grad_y, CV_32FC1, 0, 1, 1.0/26.0, 0, BORDER_REPLICATE);
}

static void getGradientInterleave(const Mat& image,Mat & grad){
    //Image gradients for alignment
    //Note that these gradients have theoretical problems under the sudden 
    // changes model of images. It might be wise to blur the images before 
    // alignment, to avoid sudden changes, but that makes occlusion more 
    // problematic.
    grad.create(2,image.rows*image.cols,CV_32FC1);
    Mat gray;
    if (image.type()==CV_32FC1) {
        gray=image;
    }else {
        cvtColor(image, gray, CV_BGR2GRAY);
        gray.convertTo(gray,CV_32FC1);
    }
    Mat gradX(image.rows,image.cols,CV_32FC1);
    Scharr( gray, gradX, CV_32FC1, 1, 0, 1.0/26.0, 0, BORDER_REPLICATE );
    Mat gradY(image.rows,image.cols,CV_32FC1);
    Scharr( gray, gradY, CV_32FC1, 0, 1, 1.0/26.0, 0, BORDER_REPLICATE);
    Mat src [2]={gradY,gradX};
    merge(src,2,grad);
}

static void Mask(const Mat& in,const Mat& m,Mat& out){
    Mat tmp;
    
    m.convertTo(tmp,in.type());
    out=out.mul(tmp/255);
}


static bool align_level_largedef_gray_forward(const Mat& T,//Total Mem cost ~185 load/stores of image
                          const Mat& d,
                          const Mat& _I,
                          const Mat& cameraMatrix,//Mat_<double>
                          const Mat& _p,                //Mat_<double>
                          int mode,
                          float threshold,
                          int numParams
                                      )
{

    int r=_I.rows;
    int rows=r;
    int c=_I.cols;
    int cols=c;
    const float small=small0;
    //Build the in map (Mem cost 3 layer store:3)
    
    Mat_<Vec3f> idMap3;
    if(numParams>3){
        idMap3.create(r,c);//[rows][cols][3]
        float* id3=(float*) (idMap3.data);
        float* dp=(float*) (d.data);
        int offset=0;
        for(int i=0;i<r;i++){
            for(int j=0;j<c;j++,offset++){
                id3[offset*3+0]=j;
                id3[offset*3+1]=i;
                id3[offset*3+2]=dp[offset];
            }
        }
    }else{
        idMap3.create(r,c);//[rows][cols][3]
        float* id3=(float*) (idMap3.data);
        int offset=0;
        for(int i=0;i<r;i++){
            for(int j=0;j<c;j++,offset++){
                id3[offset*3+0]=j;
                id3[offset*3+1]=i;
                id3[offset*3+2]=0.0f;
            }
        }
    }
        
   
    
    //Build the unincremented transform: (Mem cost 2 layer store,3 load :5)
    Mat baseMap(rows,cols,CV_32FC2);
    {
        Mat tmp=_p.clone();
        Mat baseProj=paramsToProjection(_p,cameraMatrix);
        perspectiveTransform(idMap3,baseMap,baseProj);
        assert(baseMap.type()==CV_32FC2);
    }
    
    
    // reproject the gradient and image at the same time (Mem cost >= 24)
    Mat gradI;
    Mat I(r,c,CV_32FC1);
    {
        getGradient(_I,gradI); //(Mem cost: min 2 load, 2 store :4)
        Mat toMerge[3]={_I,
                        Mat(r,c,CV_32FC1,(float*)gradI.data),
                        Mat(r,c,CV_32FC1,((float*)gradI.data)+r*c)};
        Mat packed;
        merge(toMerge,3,packed); //(Mem cost: min 3 load, 3 store :6)
        Mat pulledBack;
        
        remap( packed, pulledBack, baseMap,Mat(), CV_INTER_LINEAR, BORDER_CONSTANT,0.0 );//(Mem cost:?? 5load, 3 store:8)
        gradI.create(r,c,CV_32FC2);

        int from_to[] = { 0,0, 1,1, 2,2 };
        Mat src[1]=pulledBack;
        Mat dst[2]={I,gradI};
        
        mixChannels(src,1,dst,2,from_to,3);// extract the image and the resampled gradient //(Mem cost: min 3 load, 3 store :6)
        
        
        if(cv::countNonZero(I)<rows*cols*FAIL_FRACTION){//tracking failed!
            return false;
//             cout<<"TRACKING FAILURE, REBASING"<<endl;
//             pose=basePose.clone();
//             cout << "True Pose: "<< pose << endl;
//             cout << "Base Pose: "<< basePose << endl;
//             
//             LieSub(pose,basePose).copyTo(_p);// the Lie parameters 
//             bool improved = align_level_largedef_gray_forward(  T,//Total Mem cost ~185 load/stores of image
//                                                                 d,
//                                                                 _I,
//                                                                 cameraMatrix,//Mat_<double>
//                                                                 _p,                //Mat_<double>
//                                                                 mode,
//                                                                 threshold,
//                                                                 3);
//             //gpause();
//             return improved;
        }
        
    }
    
    // Calculate the differences and build mask for operations (Mem cost ~ 8)
    Mat fit;
    absdiff(T,I,fit);
    Mat mask=(fit<threshold)&(I>0);
    Mat err=T-I;
    
    //debug
    {
//         if (numParams==3){
        pfShow("Before iteration",_I);
//         if(I.rows==480){
            Mask(I,fit<.05,I);
            pfShow("Tracking Stabilized With Occlusion",I,0,Vec2d(0,1));
// //             gpause();
//         }
//         else{
            pfShow("After Iteration",I,0,Vec2d(0,1));
            pfShow("To match",T);
//             gpause(); 
//         }
//         }
    }
    
   
    
    
    
    // Build Jacobians:
    Mat Jsmall;
    Jsmall.create(numParams,rows*cols,CV_32FC1);
    int OM_OFFSET=0;//128;//offset helps keep cache from being clobbered by load/stores
    Mat outContainer;
    outContainer.create(numParams,rows*cols*2+OM_OFFSET,CV_32FC1);
    
    //TODO: Whole loop cacheable except J multiplies if CV_DTAM_REV (Mem cost whole loop 17/itr: 102)
    for (int paramNum=0; paramNum<numParams; paramNum++) {
        
        
        
        //Build the incremented transform
        assert(_p.type()==CV_64FC1);
        Mat_<double> p=_p.clone();
        p(0,paramNum)+=small;
        Mat proj=paramsToProjection(p,cameraMatrix);
        
        //get a row of dmap/dp
        Mat outMap(rows,cols,CV_32FC2,((float*)outContainer.data)+rows*cols*2*paramNum+OM_OFFSET);
        
        perspectiveTransform(idMap3,outMap,proj);//outmap=baseMap+dMap/dp*small (Mem cost 5)
        
        
        //subtract off the base to make a differential  (Mem cost 6)
        //this cannot be done below in the J loop because it would need 5 pointers
        // which is bad for cache (4-way set associative)
//         Mat t1,t2;
//         Mat tmp[2]={t1,t2};
//         split(outMap,tmp);
//         char s[500];
//         sprintf(s,"diff0:%d",paramNum);
//         pfShow(s,tmp[0]);
//         pfShow("diff1",tmp[1]);
        outMap-=baseMap;//outmap=dMap/dp*small
//         split(outMap,tmp);
//         pfShow("diff2",tmp[0]);
//         pfShow("diff3",tmp[1]);
        //multiply by appropriate gradient
        
        
        //want:J*small=dI/dMap*dMap/dp*small
        //do: Jsmall=sumChannels((outmap-idMap2).mul(merge(gradient[0],gradient[1])))
        const float * om=(const float*) (outMap.data);
        const float * bm=(const float*) (baseMap.data);
        const float * gi=(const float*) (gradI.data);
        const uchar * m = mask.data;
        float* Jp=(float*) (Jsmall.row(paramNum).data);//the row of the jacobian we're computing
        int offset = 0;
        //TODO: this loop may work faster unrolled or hand sse/avx optimized (Mem cost 6)
        //Warning this loop uses all 4 mappings in a 4 way cache!
        //Unrolling to cache line size would allow a 5th pointer to be used.
        for(float i=0;i<rows;i++){
            for(float j=0;j<cols;j++,offset++){
                float jx,jy;
                jx = om[offset*2+0]*gi[offset*2+0];
                jy = om[offset*2+1]*gi[offset*2+1];
                Jp[offset]=m[offset]?jx+jy:0;
            }
        }
        //usleep(1000000);
    }
    //now want: dp=(J'J)^-1*J'*(T-I)
    //          dp=small*(Jsmall*Jsmall')^-1*Jsmall*(T-I) since Jsmall is already transposed
    //          dp=small*Hsmallsmall^-1*Jsmall*(T-I)
    Mat Hss=Jsmall*Jsmall.t(); //Hessian (numParams^2) (Mem cost 6-36 depending on cache)
    Hss.convertTo(Hss,CV_64FC1);
    Mat Hinv=small*Hss.inv(DECOMP_SVD);  //TODO:cacheable for CV_DTAM_REV 
    Hinv.convertTo(Hinv,CV_32FC1);
    err=err.reshape(0,r*c);

    Mat dp=(Hinv*(Jsmall*err)).t();//transpose because we decided that p is row vector (Mem cost 7)
    dp.convertTo(dp,CV_64FC1);
//     cout<<"Je: \n"<<Jsmall*err<<endl;
//     cout<<"H: "<<"\n"<< Hss<< endl;
//     cout<<"Hinv: "<<"\n"<< Hinv<< endl;
//     cout<<"dp: "<<"\n"<< dp<< endl;
    
    
    //Check amount of motion
    {
        
    }
    
    //Check error
    //For the pixels that are within threshold, the average error should go down (Expensive!)
//     {
//         Mat tmp=_p.clone();
//         tmp.colRange(0,numParams)+=dp;
//         Mat newMap,newBack;
//         Mat newProj=paramsToProjection(tmp,cameraMatrix);
//         perspectiveTransform(idMap3,newMap,newProj);
//         remap( _I, newBack, newMap, Mat(), CV_INTER_LINEAR, BORDER_CONSTANT,-1.0/0.0 );
//         Mat newFit;
//         absdiff(T,newBack,newFit);
//         Mat fitDiff;
//         subtract(fit,newFit,fitDiff,mask & (newBack>0));
//         double deltaErr=sum(fitDiff)[0];
//         cout<<"Delta Err: "<< deltaErr<<endl;
//         if (deltaErr<0)
//             return false;
//     }
    _p.colRange(0,numParams)+=dp;
    return true;
}











// void align_level_largedef(const Mat& T,
//                           const Mat& d,
//                           const Mat& I,
//                           const Mat& cameraMatrix,//Mat_<double>
//                           Mat& _p,                //Mat_<double>
//                           int mode,
//                           int numParams,
//                           float threshold)
// {
//     int rows=T.rows,cols=T.cols;
//     Mat_<float> gradT;//[2][rows][cols]
//     Mat_<float> gradI;//[2][rows][cols]
//     
//     
//     const float small=.0000001;
//     
//     //Build the in map
//     Mat_<Vec3f> idMap3;
//     {
//         // Mat_<Vec2f> idMap2;                        //UNUSED
//         idMap3.create(rows,cols);//[rows][cols][3]
//         // idMap2.create(rows,cols);//[rows][cols][2]//UNUSED
//         float* id3=(float*) (idMap3.data);
//         // float* id2=(float*) (idMap2.data);        //UNUSED
//         float* dp=(float*) (d.data);
//         int offset=0;
//         for(int i=0;i<rows;i++){
//             for(int j=0;j<cols;j++,offset++){
//                 id3[offset*3+0]=j;
//                 id3[offset*3+1]=i;
//                 id3[offset*3+2]=dp[offset];
//                 // id2[offset*2+0]=j;//UNUSED
//                 // id2[offset*2+1]=i;//UNUSED
//             }
//         }
//     }
//     
//     //Build the unincremented transform:
//     Mat baseMap(rows,cols,CV_32FC2);
//     {
//         Mat baseProj=paramsToProjection(_p,cameraMatrix);
//         Mat baseMap(rows,cols,CV_32FC2);
//         perspectiveTransform(idMap3,baseMap,baseProj);
//     }
//     
//     //Build the image space gradient
//     Mat gradient;
//     switch (mode) {
//         case CV_DTAM_ESM:
//         {
//             cerr << "Error: ESM not supported with large deformation"<<endl;
//             assert(false);
//         }break;
//         case CV_DTAM_FWD://SLOW!
//         {
//             gradI.create(2,rows*cols);
//             getGradient(I,gradI);
//             //pull back the gradient
//             Mat gradimage;
//             Mat gradLayers[2] = {gradI.row(0),gradI.row(1)};
//             merge(gradLayers,2,gradimage);
//             
//             Mat gradPullback;
//             remap( gradimage, gradPullback, baseMap, Mat(), CV_INTER_LINEAR, BORDER_REPLICATE);// Might be faster on 3 channels
//             split(gradPullback,gradLayers);
//             gradient=gradI;
//         }break;
//         case CV_DTAM_REV:
//         {
//             gradT.create(2,rows*cols);
//             getGradient(T,gradT);
//             gradient=gradT;
//         }break;
//         default:
//         {
//             cerr << "Error: Unsupported gradient type for tracking"<<endl;
//             assert(false);
//         }break;
//     }
//     
//     //Find the difference between the costs
//     Mat dimg = T-I;
//     Mat adimg;
//     absdiff(T,I,adimg);
//     
//     
//     Mat Jsmall;
//     Jsmall.create(numParams,rows*cols,CV_32FC1);
//     
//     Mat outContainer;
//     outContainer.create(numParams,rows*cols*2,CV_32FC1);
//     
//     //TODO: Whole loop cacheable except J multiplies if CV_DTAM_REV
//     for (int paramNum=0; paramNum<numParams; paramNum++) {
//         
//         
//         
//         //Build the incremented transform
//         assert(_p.type()==CV_64FC1);
//         Mat_<double> p=_p.clone();
//         p(0,paramNum)+=small;
//         Mat proj=paramsToProjection(p,cameraMatrix);
//         
//         //get a row of dmap/dp
//         Mat outMap(rows,cols,CV_32FC2,((float*)outContainer.data)+rows*cols*2*paramNum);
//         
//         perspectiveTransform(idMap3,outMap,proj);//outmap=baseMap+dMap/dp*small
//         
//         
//         //subtract off the base to make a differential
//         //this cannot be done below in the J loop because it would need 5 pointers
//         // which is bad for cache (4-way set associative)
//         outMap-=baseMap;//outmap=dMap/dp*small
//         
//         //multiply by appropriate gradient
//         
//         
//         //want:J*small=dI/dMap*dMap/dp*small
//         //do: Jsmall=sumChannels((outmap-idMap2).mul(merge(gradient[0],gradient[1])))
//         const float * om=(const float*) (outMap.data);
//         const float * bm=(const float*) (baseMap.data);
//         const float * gx=(const float*) (gradient.data);
//         const float * gy=((const float*) (gradient.data))+rows*cols;
//         float* Jp=(float*) (Jsmall.row(paramNum).data);//the row of the jacobian we're computing
//         int offset = 0;
//         //TODO: this loop may work faster unrolled or hand sse/avx optimized
//         //Warning this loop uses all 4 mappings in a 4 way cache!
//         //Unrolling to cache line size would allow a 5th pointer to be used.
//         for(float i=0;i<rows;i++){
//             for(float j=0;j<cols;j++,offset++){
//                 float jx,jy;
//                 jx = om[offset*2+0]*gx[offset];
//                 jy = om[offset*2+1]*gy[offset];
//                 Jp[offset]=jx+jy;
//             }
//         }
//         
//         // remove occluded pixels:
//         offset = 0;
//         for(float i=0;i<rows;i++){
//             for(float j=0;j<cols;j++,offset++){
//                 float test=adimg.at<float>(0,offset) ;
//                 
//                 if(threshold<test ){
//                     Jp[offset]=0;
//                 }
//             }
//         }
//     }
//     
//     // Now we have constructed the gradient for all pixels assuming small deformation.
//     // We still have problems with occluded pixels and with the fact that the deformation
//     // is not smooth. 
//     
//     
//     //now want: dp=(J'J)^-1*J'*(T-I)
//     //          dp=small*(Jsmall*Jsmall')^-1*Jsmall*(T-I) since Jsmall is already transposed
//     //          dp=small*Hsmallsmall^-1*Jsmall*(T-I)
//     Mat Hsmallsmall=Jsmall*Jsmall.t(); //Hessian (numParams^2) 
//         Mat Hinv=small*Hsmallsmall.inv();  //TODO:cacheable for CV_DTAM_REV 
//         
//         
//         
//         Mat dp=(Hinv*Jsmall*(T-I)).t();//transpose because we decided that p is row vector
//         dp.convertTo(dp,CV_64FC1);
//         _p+=dp;
// }








// void align_level_smalldef(const Mat& T,
//                           const Mat& d,
//                           const Mat& I,
//                           const Mat& cameraMatrix,//Mat_<double>
//                           Mat& _p,                //Mat_<double>
//                           int mode,
//                           float threshold,
//                           int numParams
// )
// {
//     
//     
//     int rows=T.rows,cols=T.cols;
// 
//     Mat_<float> gradT;//[2][rows][cols]
//     Mat_<float> gradI;//[2][rows][cols]
//     
//     
//     const float small=0.0000001;
//     
//     //Build the in map and map to subtract
//     Mat_<Vec3f> idMap3;
//     {
//         // Mat_<Vec2f> idMap2;                        //UNUSED
//         idMap3.create(rows,cols);//[rows][cols][3]
//         // idMap2.create(rows,cols);//[rows][cols][2]//UNUSED
//         float* id3=(float*) (idMap3.data);
//         // float* id2=(float*) (idMap2.data);        //UNUSED
//         float* dp=(float*) (d.data);
//         int offset=0;
//         for(int i=0;i<rows;i++){
//             for(int j=0;j<cols;j++,offset++){
//                 id3[offset*3+0]=j;
//                 id3[offset*3+1]=i;
//                 id3[offset*3+2]=dp[offset];
//                 // id2[offset*2+0]=j;//UNUSED
//                 // id2[offset*2+1]=i;//UNUSED
//             }
//         }
//     }
//     
//     Mat Jsmall;
//     Jsmall.create(numParams,rows*cols,CV_32FC1);
//     
//     Mat outContainer;
//     outContainer.create(numParams,rows*cols*2,CV_32FC1);
//     
//     
//     for (int paramNum=0; paramNum<numParams; paramNum++) {
//         
//         //Build the incremented transform
//         Mat_<double> p=Mat::zeros(0,numParams,CV_64FC1);
//         p(0,paramNum)+=small;
//         Mat dR=rodrigues(p.colRange(Range(0,3)));
//         Mat dT=p.colRange(Range(3,6));
//         Mat dA;
//         hconcat(dR,dT,dA);
//         dA=make4x4(dA);
//         
//         Mat proj=cameraMatrix*dA*cameraMatrix.inv();
//         //The column swap
//         Mat tmp=proj.colRange(2,4).clone();
//         tmp.col(1).copyTo(proj.col(2));
//         tmp.col(0).copyTo(proj.col(3));
//         //The row drop
//         proj=proj.rowRange(0,3);
//         
//         //get a row of dmap/dp
//         Mat outMap(rows,cols,CV_32FC2,((float*)outContainer.data)+rows*cols*2*paramNum);
//         perspectiveTransform(idMap3,outMap,proj);//outmap=idMap2+dMap/dp*small
//         //TODO: all of outContainer is cacheable
//         // even if gradient is not.
//         
//         //subtract off the identity and multiply by appropriate gradient
//         Mat gradient;
//         switch (mode) {
//             case CV_DTAM_ESM:
//                 gradT.create(2,rows*cols);
//                 getGradient(T,gradT);
//                 gradI.create(2,rows*cols);
//                 getGradient(I,gradI);
//                 gradient=(gradT+gradI)/2.0;
//                 break;
//             case CV_DTAM_FWD:
//                 gradI.create(2,rows*cols);
//                 getGradient(I,gradI);
//                 gradient=gradI;
//                 break;
//             case CV_DTAM_REV:
//                 gradT.create(2,rows*cols);
//                 getGradient(T,gradT);
//                 gradient=gradT; //can optimize out since doesn't change between iterations
//                 break;
//             default:
//                 assert(false/*unsupported gradient type*/);
//                 break;
//         }
//         //want:J*small=dI/dMap*dMap/dp*small
//         //do: Jsmall=sumChannels((outmap-idMap2).mul(merge(gradient[0],gradient[1])))
//         const float * om=(float*) (outMap.data);
//         const float * gx=(const float*) (gradient.data);
//         const float * gy=((const float*) (gradient.data))+rows*cols;
//         float* Jp=(float*) (Jsmall.row(paramNum).data);//the row of the jacobian we're computing
//         int offset = 0;
//         //TODO: this loop may work faster unrolled or hand sse/avx optimized
//         //TODO: presubtracting the i's and j's will be better if outContainer is cached
//         //Warning this loop uses all 4 mappings in a 4 way cache!
//         for(float i=0;i<rows;i++){
//             for(float j=0;j<cols;j++,offset++){
//                 float jx,jy;
//                 jx = (om[offset*2+0]-j)*gx[offset];
//                 jy = (om[offset*2+1]-i)*gy[offset];
//                 Jp[offset]=jx+jy;
//             }
//         }
//     }
//     
//     // now we have constructed the gradient for all pixels assuming small deformation
//     // we still have problems with occluded pixels and with the fact that the deformation
//     // is not smooth. 
//     
//     //now want: dp=(J'J)^-1*J'*(T-I)
//     //          dp=small*(Jsmall*Jsmall')^-1*Jsmall*(T-I) since Jsmall is already transposed
//     //          dp=small*Hsmallsmall^-1*Jsmall*(T-I)
//     Mat Hsmallsmall=Jsmall*Jsmall.t(); //Hessian (numParams^2) 
//     Mat Hinv=small*Hsmallsmall.inv();  //TODO:cacheable for CV_DTAM_REV 
//     Mat dp=(Hinv*Jsmall*(T-I)).t();//transpose because we decided that p is row vector
//     dp.convertTo(dp,CV_64FC1);
//     _p+=dp;
// }






// void align_level_largedef(const Mat& T,
//                           const Mat& d,
//                           const Mat& I,
//                           const Mat& cameraMatrix,//Mat_<double>
//                           Mat& _p,                //Mat_<double>
//                           int mode,
//                           float threshold)
// {
