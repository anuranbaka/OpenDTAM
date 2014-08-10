#include <pthread.h>

#ifndef SET_AFFINITY_H
#define SET_AFFINITY_H
static void set_affinity(int cpuid){
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpuid, &cpuset);
    pthread_setaffinity_np(pthread_self(),1, &cpuset);
}
#endif