class Track{
	
	timestamp //timestamp pf last pose
	lastframe
	velocity //last velocity
	pose
	camera
	calibration
	map
	
	
	Track(map, calibration) //no abilities
	Track(map, camera, calibration)
	Track(map, camera, calibration, timestamp, velocity, frame, pose)
	
	setPose()
	getPose()//gives last pose
	getPose(timestamp)//extrapolates pose to timestamp given
	
	update(frame)//finds the pose of the new frame, no velocity/timestamp update
	update(frame, timestamp)//finds the pose of the new frame and updates velocity
	
	
	
	
	
	
	
}