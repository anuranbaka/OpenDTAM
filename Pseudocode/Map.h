class Map{
	//things not set to null should be assigned in constructor
	//external linkages
	tracker = NULL
	camera
	calibration
	
	// input queue
	frames
	poses
	
	//internal data structure
	keyframe
	keypose
	cost
	depth = NULL
	
	addKeyframe()
	addKeyframe(frame,pose)
	addFrame()//pulls a frame from the camera, asks the tracker for a pose
	addFrame(frame,pose) //adds a frame manually 
	setTracker(tracker);
	updateCost() //updates the cost with the accumulated frames and poses
	resetCost() //zeros out the cost volume
	resetIteration() //sets theta back to theta_0
	iterate()
	converge() // calls iterate over and over until converged
	
	//constants
	lambda //could change by frame, but I probably won't
	epsilon
	alpha_G
	beta_G
	beta_theta
	
	
	//used by iterator core
	D // a copy of the depth to be used in the iterations 
	Q
	A
	C := cost // and alias to the cost volume's cost data
	G
	sigma_q
	sigma_d
	theta
	n_theta
	
	Map(camera,calibration):
			camera,calibration{
		keyframe = camera.getFrame();
		init();
	}
	
	Map(keyframe,keypose,initialTracker,camera,calibration):
			keyframe,keypose,tracker(initialTracker),camera,calibration{
		init();
	}
	
	
	private init(){
		resetCost();
		computeG();
		[sigma_q, sigma_d] = computeSigmas();
	}
	
	addKeyframe(){//for use when already have frames to build a keyframe from
		addFrame();//pull frame from camera
		addKeyframe(frames.end, poses.end);
		init(); //need to reinitialize because all previous work is invalid
		assert(frames.length() >= NUM_FRAMES_BEFORE_INIT);// need to have built up enough frames to create a new keyframe well
		updateCost();
	}
	
	addKeyframe(keyframe, keypose):keyframe,keypose{
	}
	
	addFrame(){
		camera.getFrame();
		timestamp = now();
		tracker.update(frame,timestamp);
		addFrame(frame,tracker.getPose());
		updateCost();
		iterate();
	}
		
	addFrame(frame,pose){
		synchronize(frames){
			frames.push(frame);
			poses.push(pose);
		}
	}
	
	setTracker(tracker):tracker{}
	
	updateCost(){
		for [frame,pose]:[frames,poses]{
			updateCost(frame,pose);
			synchronize(frames){
				frames.pop();
				poses.pop();
			}
		}
	}
	
	updateCost(frame,pose){
		for column:cost{
			for cell:column{
				if(!cellProjectsIntoFrame(cell, frame, pose)) continue;
				color = sampleProjectedIntoFrame(cell, frame, pose);
				value = dissimilarity(color, column.color);
				cell.sum = cell.sum + value;
				cell.count++;
				column.max = max(column.max, cell.sum/cell.count);
			}
		}
	}
	
	resetCost(){
		fill cost with zeros
	}
	
	resetIteration(){
		D = A = argmin C over columns;
		Q.x = zeros(size(D));
		Q.y = zeros(size(D));
		theta = THETA_START;
		n_theta=0;
	}
	
	done = iterate(){
		iteratorCore(D, Q, A, C, G, sigma_q, sigma_d, theta);
		if (theta > THETA_END){
			theta *= (1-beta_theta*n_theta);
			n_theta++;
			done = 0;
		}
		else{
			done = 1;
			depth = D;
		}
	}
	
	converge(){
		while(!iterate());
	}
}