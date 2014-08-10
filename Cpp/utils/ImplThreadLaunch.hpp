#ifndef IMPL_THREAD_HPP
#define IMPL_THREAD_HPP

#if !defined WIN32 && !defined WINCE
#  include <pthread.h>
#  include <opencv2/core/core.hpp>
#  include <map>
#  include <iostream>
namespace ImplThread{
    extern std::map<const pthread_t,typename cv::Ptr<int> > mymap;
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
        delete pass;
        (object->*(pass->func))(pass->stop);
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
        using namespace ImplThread;
        Pass* pass=new Pass;
        pass->name=_name;
        pass->cpuid=affinity;
        pass->func=_func;
        pass->instance=&object;
        pass->stop=new int(0);
        cv::Ptr<int> stopp=pass->stop;
        pthread_t thread;
        
        pthread_create( &thread, NULL,launch, (void*) pass);
        mymap[thread]=stopp;
        return thread;
    }
    static void stopThread(unsigned int thread_id){
        using namespace ImplThread;
        *mymap.at(thread_id)=1;
        pthread_join(thread_id,NULL);
        mymap.erase(thread_id);
    }
    static void stopAllThreads(){
        using namespace ImplThread;
        for (volatile std::map<const pthread_t,typename cv::Ptr<int> >::iterator it=mymap.begin(); it!=mymap.end();){
            std::cout<<"Iter: "<<(int)it<<":"<<(int)mymap.end()<<std::endl;
            stopThread(it->first);
            it=mymap.begin();
            std::cout<<"Iter: "<<(int)it<<":"<<(int)mymap.end()<<std::endl;
        }
    }
};



#else
#error OpenDTAM ImplThreadLauncher not implemented on this system
#endif

#endif