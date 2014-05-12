local vars
pyramid[]


//define: img(x,y)=ref(mapx(x,y),mapy(x,y))

//Find a d(mapx,mapy) such that: dimg(mapx,mapy)/d(mapx,mapy)*d(mapx,mapy)=ref-img
//dimg(x,y)/dmapx(x,y)=(ref(mapx+1,mapy)-ref(mapx,mapy))= grad_x(ref)
//now find a dr from the lie group to make d(mapx,mapy) equal the right value
//i.e. find r -> R -> mapx,mapy -> make new image match the reference

small=0.000001//something small enough to make a differential
Rodrigues([1 0 0]*small,R1)
Rodrigues([0 1 0]*small,R1)
Rodrigues([0 0 1]*small,R1)
//solve for [dx,dy]/[dr1,dr2,dr3]
// the magic line:
initUndistortRectifyMap(cameraMatrix, 
				distCoeffs, 
				R0,//the rotation matrix p_new=R*p_old
                                getOptimalNewCameraMatrix(	cameraMatrix, 
								distCoeffs, 
								imageSize, 
								1, 
								imageSize, 
								0),
                                imageSize, 
				CV_32FC1, 
				x0, 
				y0);
initUndistortRectifyMap(cameraMatrix, 
				distCoeffs, 
				R1,//the rotation matrix p_new=R*p_old
                                getOptimalNewCameraMatrix(	cameraMatrix, 
								distCoeffs, 
								imageSize, 
								1, 
								imageSize, 
								0),
                                imageSize, 
				CV_32FC1, 
				x1, 
				y1);
initUndistortRectifyMap(cameraMatrix, 
				distCoeffs, 
				R2,//the rotation matrix p_new=R*p_old
                                getOptimalNewCameraMatrix(	cameraMatrix, 
								distCoeffs, 
								imageSize, 
								1, 
								imageSize, 
								0),
                                imageSize, 
				CV_32FC1, 
				x2, 
				y2);
initUndistortRectifyMap(cameraMatrix, 
				distCoeffs, 
				R3,//the rotation matrix p_new=R*p_old
                                getOptimalNewCameraMatrix(	cameraMatrix, 
								distCoeffs, 
								imageSize, 
								1, 
								imageSize, 
								0),
                                imageSize, 
				CV_32FC1, 
				x3, 
				y3);

dx_dr1=(x1-x0)*(1/small)
dx_dr2=(x2-x0)*(1/small)
dx_dr3=(x3-x0)*(1/small)
dy_dr1=(y1-y0)*(1/small)
dy_dr2=(y2-y0)*(1/small)
dy_dr3=(y3-y0)*(1/small)

[gx,gy]=grad(src)
//[dsrc/dx*dx/dr<whatever>+dref/dy*dy/dr<whatever>]
//this thing is a Numpixels x 3 matrix
dsrc_dr=[gx(:).*dx_dr1(:)+gy(:).*dy_dr1(:)     gx(:).*dx_dr2(:)+gy(:).*dy_dr2(:)     gx(:).*dx_dr3(:)+gy(:).*dy_dr3(:)];


//solve dsrc/dr * dr = dst - src
//in least square sense
c = dst - src;
dr = dsrc_dr\c(:);
dRotationMatrix=Lie2Rotation(dr) 


Lie2Rotation(dr){
//this turns out to just be rodrigues
}


