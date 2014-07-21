#ifndef TIC_TOC
#define TIC_TOC

#include <iostream>
#include <ctime>

// Just a static variable that will be visible only within this file

static struct timespec start_time;
static void tic() {
clock_gettime(CLOCK_MONOTONIC, &start_time);

}

static double toc() {
  struct timespec tv2;

  if (clock_gettime(CLOCK_MONOTONIC, &tv2))
    tv2.tv_sec = tv2.tv_nsec = -1;

  double  sec = static_cast<double>(tv2.tv_sec - start_time.tv_sec);
  double nsec = static_cast<double>(tv2.tv_nsec - start_time.tv_nsec);

  double elapsed= (sec + 1.0e-9 * nsec);

    time_t end_time=time(0);
    std::cout<<"Elapsed time is "<<elapsed<<" seconds."<<std::endl;
    return elapsed;
}
static double tocq() {
    struct timespec tv2;
    
    if (clock_gettime(CLOCK_MONOTONIC, &tv2))
        tv2.tv_sec = tv2.tv_nsec = -1;
    
    double  sec = static_cast<double>(tv2.tv_sec - start_time.tv_sec);
    double nsec = static_cast<double>(tv2.tv_nsec - start_time.tv_nsec);
    
    double elapsed= (sec + 1.0e-9 * nsec);
    
    time_t end_time=time(0);
    return elapsed;
}

#endif //TIC_TOC
