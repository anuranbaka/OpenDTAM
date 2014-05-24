



























/*//the complex internals of the iterate function
iteratorCore(D, Q, A, C, G, sigma_q, sigma_d, theta){
	//lowercase lettes are elements of uppercase letters at a given column
	
	//Q update
	grad_D=grad(D)
	for all q vectors:
		//q = PiFunc( (q+sigma_q*g*grad_d)/(1 + sigma_q*epsilon) )
		q.x = (q.x + sigma_q*g*grad_d.x)/(1 + sigma_q*epsilon)
		q.y = (q.y + sigma_q*g*grad_d.y)/(1 + sigma_q*epsilon)
		pd = PiDenom(q.x,q.y)
		q.x = q.x/pd
		q.y = q.y/pd
		
	//D update
	div_Q = div(Q)
	for all d scalars:
		d = (d + sigma_d*(g*div_q + 1/theta*a))/(1 + sigma_d/theta)
	
	//A update
	for all a scalars:
		a = argmin (1/(2*theta)*(d - a)^2+ lambda*c(a)) with a in feasible region of column c
}





// Notice that the description of both the Huber norm and
// the Pi function are inconsistent with the implementation.
// The actual implementation of the Huber norm is a parabola
// that turns smoothly into a cone at radius epsilon.
// This is actually the correct form if we want the equation to 
// be rotation invariant.
// PiFunc(vec) := vec./max(1,sqrt(sum(vec.^2)))
PiDenom(x,y){
	return max(1, sqrt(x*x+y*y))
}

computeG(){
	G = exp(-alpha_G*sqnorm2(grad(keyframe))^(beta_G/2))
}

[sigma_q, sigma_g] = computeSigmas(){
	L = 4*max(G) // can probably save time by figuring an analytical limit on this
	             // max(G) = sqrt(2) if colors are in 0,1 range
	lambda = 1/theta  	//sigma_d source
	alpha  = epsilon    //sigma_q source
	
	gamma = lambda
	delta = alpha
	
	mu      = 2*sqrt(gamma*delta)/L
	theta_3 = 1/(1+mu)    //doesn't appear to be used in dtam
	
	
	rho   = mu/(2*gamma)
	sigma = mu/(2*delta)
	
	sigma_d = rho
	sigma_q = sigma
}

//Notes:
//a b c = A
//
//d e f = B
//
// grad(A).B = -div(B).*A
//
//(b-a)*d + (c-b)*e + (0-c)*f
//(d-0)*a + (e-d)*b + (f-e)*c
*/
