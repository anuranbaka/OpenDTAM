

// this file converts the pov-ray perspective projection into a matrix
#define sq(x) (x*x)


load:
cam_pos      
cam_dir      
cam_up       
cam_lookat  
cam_sky   
cam_right  
cam_fpoint
cam_angle    



//convert to radians
cam_angle=cam_angle*M_PI/180;





//povray uses camright for scaling. We want scaling only in the camera matrix, so we recompute camright as a cross product
cam_right[0]=cam_dir[1]*cam_up[2]-cam_dir[2]*cam_up[1];
cam_right[1]=cam_dir[2]*cam_up[0]-cam_dir[0]*cam_up[2];
cam_right[2]=cam_dir[0]*cam_up[1]-cam_dir[1]*cam_up[0];

cx=w/2-.5;
cy=h/2-.5;
fx=fy=tan(cam_angle/2)*w/2;//should be 320 for the example

//takes camera frame to image frame
Mat cam_matrix = (Mat_<double>(3,3) << fx,0,cx,
										0,fy,cy,
										0,0,1);

//takes rotated camera frame to camera frame
Mat cam_rot=(Mat_<double>(3,3) <<
cam_right.x, cam_up.x, cam_dir.x, 0,
cam_right.y, cam_up.y, cam_dir.y, 0,
cam_right.z, cam_up.z, cam_dir.z, 0,
0, 0, 0, 1).inv();

//takes world frame to rotated camera frame
Mat cam_trans=(Mat_<double>(3,3) <<
1, 0, 0, -cam_pos.x,
0, 1, 0, -cam_pos.y,
0, 0, 1, -cam_pos.z,
0, 0, 0, 1);


//so now imgpoint=cam_matrix*cam_rot*cam_trans*worldpoint

//so to project an image point out to a depth d and then back to another camera


//project to depth d
x_cw=(x-cx)/f*d
y_cw=(x-cx)/f*d
z_cw=d

//matrix to project from camera to alternative camera


//to project back out to world:
[cfrx]
[cfry]
[cfrz]
[1]
=
[fx 0   0][1 0 -cx] [x]
[0 fy   0][0 1 -cy] [y]
[0 0    1][0 0 1  ] [1]
[0 0    1]             ;

//to project from world back to other image:






















