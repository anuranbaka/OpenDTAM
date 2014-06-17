///This file finds the special weighted divergence and gradient used in the optimizer
///Thie code in this file is not used, but a reference for understanding.
///The output after gradient is the "k" space(my own name, not used by the paper)
///It is like the gradient, but weirdly weighted by the g function.
///The average of the g function at the two ends of each gradient spring is the influence 
///of that spring.
///The input space to the gradient is the depth image space "d"
///
///The functions here provide these on a pointwise basis.
///The weighted gradient (used in Eq. 11 (ish because of bugs in the paper) ):
///k_x[here] =                     0*d[left] + -0.5*(g[here]+g[right])*d[here] + 0.5*(g[here]+g[right])*d[right]
///k_x[left] = -.5*(g[left]+g[here])*d[left] +  0.5*(g[left]+g[here]) *d[here] +                      0*d[right]
///
///k_y[here] =                     0*d[ up ] + -0.5*(g[here]+g[down])*d[here]  +  0.5*(g[here]+g[down])*d[down]
///k_y[ up ] = -.5*(g[ up ]+g[here])*d[ up ] +  0.5*(g[ up ]+g[here])*d[here]  +                      0*d[down]
///Only the [here] rows need to be evaluated obviously.
///
///And the transpose (used in Eq. 12):
///d[here] = 0.5*(g[ up ]+g[here])*k_y[up] + -0.5*(g[here]+g[down])*k_y[here] +  0.5*(g[left]+g[here])*k_x[left] + -0.5*(g[here]+g[right])*k_x[here]
///
///In practice I will include the 0.5 factor in the g function, but the above is idealized
///
///These equations have a kind of beautiful symmetry hidden by the words used to express them on paper.
///I wish I could draw a picture.

// A clear optimization would be to store g[here]+g[whatever] as gwhatever[here]


//Below are the obvious representations. Commented out because we use optimized ones 
//k functions(notice they can't be evaluated on the right or bottom edge respectively
inline float k_x(float& dhere, float& dright, float& ghere, float& gright){
    return (ghere+gright)*(dright-dhere);
}
inline float k_y(float& dhere, float& ddown, float& ghere, float& gdown){
    return  (ghere+gdown)*(ddown-dhere);
}


//d functions in order of usage:
//d_topleft      d_top        d_topright
//d_left         d_core          d_right
//d_bottomleft  d_bottom   d_bottomright

inline float d_topleft    (float& kxhere, float&kxleft, float& kyhere, float& kyup, float& ghere, float& gup, float& gdown, float& gleft, float& gright){
    return                 -(gdown+ghere)*kyhere                     -(ghere+gright)*kxhere;
}
inline float d_top        (float& kxhere, float&kxleft, float& kyhere, float& kyup, float& ghere, float& gup, float& gdown, float& gleft, float& gright){
    return                 -(gdown+ghere)*kyhere+(gleft+ghere)*kxleft-(ghere+gright)*kxhere;
}
inline float d_topright   (float& kxhere, float&kxleft, float& kyhere, float& kyup, float& ghere, float& gup, float& gdown, float& gleft, float& gright){
    return                 -(gdown+ghere)*kyhere+(gleft+ghere)*kxleft                      ;
}

inline float d_left       (float& kxhere, float&kxleft, float& kyhere, float& kyup, float& ghere, float& gup, float& gdown, float& gleft, float& gright){
    return (gup+ghere)*kyup-(gdown+ghere)*kyhere+                    -(ghere+gright)*kxhere;
}
inline float d_core       (float& kxhere, float&kxleft, float& kyhere, float& kyup, float& ghere, float& gup, float& gdown, float& gleft, float& gright){
    return (gup+ghere)*kyup-(gdown+ghere)*kyhere+(gleft+ghere)*kxleft-(ghere+gright)*kxhere;
}
inline float d_right      (float& kxhere, float&kxleft, float& kyhere, float& kyup, float& ghere, float& gup, float& gdown, float& gleft, float& gright){
    return (gup+ghere)*kyup-(gdown+ghere)*kyhere+(gleft+ghere)*kxleft                      ;
}

inline float d_bottomleft (float& kxhere, float&kxleft, float& kyhere, float& kyup, float& ghere, float& gup, float& gdown, float& gleft, float& gright){
    return (gup+ghere)*kyup                                          -(ghere+gright)*kxhere;
}
inline float d_bottom     (float& kxhere, float&kxleft, float& kyhere, float& kyup, float& ghere, float& gup, float& gdown, float& gleft, float& gright){
    return (gup+ghere)*kyup-(gdown+ghere)*kyhere+(gleft+ghere)*kxleft-(ghere+gright)*kxhere;
}
inline float d_bottomright(float& kxhere, float&kxleft, float& kyhere, float& kyup, float& ghere, float& gup, float& gdown, float& gleft, float& gright){
    return (gup+ghere)*kyup                                          -(ghere+gright)*kxhere;
}


//We would use these functions straight if we wanted absolute max performance, but the code gets uglier, so we'll just use padding to fill in the zeros



//Now for the forms we use. Notice that gwhatever is implicitly ghere+gwhatever
inline float k_x(float& dhere, float& dright,float& gright){
    return gright*(dright-dhere);
}
inline float k_y(float& dhere, float& ddown, float& gdown){
    return  gdown*(ddown-dhere);
}

inline float d_down(float& kdown,float& gdown){
    return -gdown*ky
}