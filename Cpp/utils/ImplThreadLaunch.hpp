#ifndef IMPL_THREAD_HPP
#define IMPL_THREAD_HPP

#if !defined WIN32 && !defined WINCE
#  include <pthread.h>
#  include <opencv2/core/core.hpp>
#  include <map>
#  include <iostream>
#include <Scheduler/ImplMutex.hpp>
namespace ImplThread{
    extern std::map<const pthread_t,int* > mymap;
    extern ImplMutex mutex;
    void stopAllThreads();
    unsigned int startThread(void (*_func)(int*) , const std::string& _name="ODMThread",int affinity=-1);
}
template <typename Object>
class ImplThreadLauncher{
    
    struct Pass{
        std::string name;
        int cpuid;
        void (Object::*func)(int*);
        Object* instance;
        int * stop;
    };
    
    
    static void* launch (void* data){
        Pass* pass=(Pass*)data;
        pthread_setname_np(pthread_self(),pass->name.c_str());
        if (pass->cpuid>0){
            set_affinity(pass->cpuid);
        }
        Object* object=(Object*)pass->instance;

        (object->*(pass->func))(pass->stop);
        delete pass;
    }
    
    static void set_affinity(int cpuid){
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        static int cores=cv::getNumberOfCPUs();
        CPU_SET(cpuid%cores, &cpuset);
        pthread_setaffinity_np(pthread_self(),1, &cpuset);
    }
    
    
public:
    static unsigned int startThread(Object& object,void (Object::*_func)(int*), const std::string& _name="ODMThread",int affinity=-1){
        ImplThread::mutex.lock();
        using namespace ImplThread;
        Pass* pass=new Pass;
        pass->name=_name;
        pass->cpuid=affinity;
        pass->func=_func;
        pass->instance=&object;
        pass->stop=new int(0);
        int* stopp=pass->stop;
        pthread_t thread;
        
        pthread_create( &thread, NULL,launch, (void*) pass);
        std::cout<<" Thread Requested: "<<_name<<" : "<<thread<<":"<< stopp<< std::endl;
        mymap[thread]=stopp;
        ImplThread::mutex.unlock();
        return thread;
    }
    static void stopThread(unsigned int thread_id){
        ImplThread::mutex.lock();
        using namespace ImplThread;
        if(mymap.count(thread_id)){
            *mymap.at(thread_id)=1;
            std::cout<<" Thread Stop: "<< thread_id<< std::endl;
            pthread_join(thread_id,NULL);
            mymap.erase(thread_id);   
        }
        ImplThread::mutex.unlock();
    }
    
};



#else
#error OpenDTAM ImplThreadLauncher not implemented on this system
#endif

#endif