// Free for non-commercial, non-military, and non-critical 
// use unless incorporated in OpenCV. 
// Inherits OpenCV Licence if in OpenCV.


#include "Optimizer.h"

void Optimizer::initOptimization(){
    QDruncount=0;
            Aruncount=0;
            thetaStart=500.0;
            thetaMin=0.01;
            running=false;

            epsilon=.1;
            lambda=.000001;
            thetaStep=.99;
}
void Optimizer::initQD(){
    QDruncount=0;

            thetaStart=500.0;
            thetaMin=0.01;
            running=false;

            epsilon=.1;
            lambda=.000001;
            thetaStep=.99;
}
