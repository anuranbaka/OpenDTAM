#if !defined WIN32 && !defined WINCE
#  include <pthread.h>

template <typename Object>
class ImplThreadLauncher{
    struct Pass{
        std::string name;
        int cpuid;
        void (Object::*func)(void);
        Object* instance;
    };
    
    
    static void* launch (void* data){
        Pass* pass=(Pass*)data;
        pthread_setname_np(pthread_self(),pass->name.c_str());
        if (pass->cpuid>0){
            set_affinity(pass->cpuid);
        }
        Object* object=(Object*)pass->instance;
        delete pass;
        (object->*(pass->func))();
    }
    
    static void set_affinity(int cpuid){
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        static int cores=getNumberOfCPUs();
        CPU_SET(cpuid%cores, &cpuset);
        pthread_setaffinity_np(pthread_self(),1, &cpuset);
    }
    
    
public:
    static void startThread(Object& object,void (Object::*_func)(void), const std::string& _name="ODMThread",int affinity=-1){
        Pass* pass=new Pass;
        pass->name=_name;
        pass->cpuid=affinity;
        pass->func=_func;
        pass->instance=&object;
        pthread_t thread;
        
        pthread_create( &thread, NULL,launch, (void*) pass);
    }
};
#else
#error OpenDTAM ImplThreadLauncher not implemented on this system
#endif
