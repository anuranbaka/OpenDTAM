// Free for non-commercial, non-military, and non-critical 
// use unless incorporated in OpenCV. 
// Inherits OpenCV Licence if in OpenCV.


#ifndef TRACKER3D_H
#define TRACKER3D_H

class Tracker3D
{

public:
Tracker3D();
Tracker3D(const Tracker3D& other);
virtual Tracker3D& operator=(const Tracker3D& other);
};

#endif // TRACKER3D_H
