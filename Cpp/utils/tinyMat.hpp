// Free for non-commercial, non-military, and non-critical
// use unless incorporated in OpenCV.
// Inherits OpenCV Licence if in OpenCV.


#ifndef TINYMAT_HPP_
#define TINYMAT_HPP_

template<typename _Tp, int m, int n> union Matt
{
    _Tp d[m][n]; //< matrix elements
    _Tp l[m*n]; //< matrix elements
};
typedef Matt<float, 1, 2> Matt12f;
typedef Matt<double, 1, 2> Matt12d;
typedef Matt<float, 1, 3> Matt13f;
typedef Matt<double, 1, 3> Matt13d;
typedef Matt<float, 1, 4> Matt14f;
typedef Matt<double, 1, 4> Matt14d;
typedef Matt<float, 1, 6> Matt16f;
typedef Matt<double, 1, 6> Matt16d;

typedef Matt<float, 2, 1> Matt21f;
typedef Matt<double, 2, 1> Matt21d;
typedef Matt<float, 3, 1> Matt31f;
typedef Matt<double, 3, 1> Matt31d;
typedef Matt<float, 4, 1> Matt41f;
typedef Matt<double, 4, 1> Matt41d;
typedef Matt<float, 6, 1> Matt61f;
typedef Matt<double, 6, 1> Matt61d;

typedef Matt<float, 2, 2> Matt22f;
typedef Matt<double, 2, 2> Matt22d;
typedef Matt<float, 2, 3> Matt23f;
typedef Matt<double, 2, 3> Matt23d;
typedef Matt<float, 3, 2> Matt32f;
typedef Matt<double, 3, 2> Matt32d;

typedef Matt<float, 3, 3> Matt33f;
typedef Matt<double, 3, 3> Matt33d;

typedef Matt<float, 3, 4> Matt34f;
typedef Matt<double, 3, 4> Matt34d;
typedef Matt<float, 4, 3> Matt43f;
typedef Matt<double, 4, 3> Matt43d;

typedef Matt<float, 4, 4> Matt44f;
typedef Matt<double, 4, 4> Matt44d;
typedef Matt<float, 6, 6> Matt66f;
typedef Matt<double, 6, 6> Matt66d;

#endif /* TINYMAT_HPP_ */
