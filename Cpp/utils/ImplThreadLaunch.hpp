#if !defined WIN32 && !defined WINCE
#  include <pthread.h>
#  include <opencv2/core/mat.hpp>
#  include <map>
namespace ImplThread{
    std::map<const pthread_t,typename cv::Ptr<int> > mymap;
    void stopAllThreads(){
        using namespace ImplThread;
        for (std::map<const pthread_t,typename cv::Ptr<int> >::iterator it=mymap.begin(); it!=mymap.end();){
            unsigned int thread_id=it->first;
            *mymap.at(thread_id)=1;
            pthread_join(thread_id,NULL);
            mymap.erase(thread_id);
        }
    }
    void*LAUNCH_THREAD(void* in){
        std::pair<void (*)(int*),int*>* p=(std::pair<void (*)(int*),int*>*)in;
        (p->first)(p->second);
        
    };
    unsigned int startThread(void (*_func)(int*), const std::string& _name="ODMThread",int affinity=-1){
        cv::Ptr<int> stopp=new int(0);
        pthread_t thread;
        void* in=new std::pair<void (*)(int*),int*>(_func,stopp);
        pthread_create( &thread, NULL,LAUNCH_THREAD, in);
        pthread_setname_np(thread,_name.c_str());
        if (affinity>0){
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            static int cores=getNumberOfCPUs();
            CPU_SET(affinity%cores, &cpuset);
            pthread_setaffinity_np(thread,1, &cpuset);
        }
        mymap[thread]=stopp;
        return thread;
    }
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
        static int cores=getNumberOfCPUs();
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
    static void allStop(){
        using namespace ImplThread;
        for (std::map<const pthread_t,typename cv::Ptr<int> >::iterator it=mymap.begin(); it!=mymap.end();){
            stopThread(it->first);
        }
    }
};



#else
#error OpenDTAM ImplThreadLauncher not implemented on this system
#endif
