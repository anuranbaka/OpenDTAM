#if !defined WIN32 && !defined WINCE
#  include "ImplThreadLaunch.hpp"

using namespace std;
namespace ImplThread{
    ImplMutex mutex;
    std::map<const pthread_t,int* > mymap;
    void stopAllThreads(){
        mutex.lock();
        using namespace ImplThread;
        for ( std::map<const pthread_t,int*>::iterator it=mymap.begin(); it!=mymap.end();it++){
            pthread_t thread_id=it->first;
            *(it->second)=1;
            cout<<" Thread Stop: "<< thread_id<<":"<<it->second<< endl;
            
            timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_sec += 5;//5 sec timeout
            if(pthread_timedjoin_np(thread_id,NULL,&ts))
                pthread_cancel(thread_id);
            
//             it=mymap.begin();
        }
         mymap.erase(mymap.begin(),mymap.end());
         mutex.unlock();
    }
    void*LAUNCH_THREAD(void* in){
        std::pair<void (*)(int*),int* >* p=(std::pair<void (*)(int*),int* >*)in;
        void (*func)(int*)=p->first;
        cout<<" Thread Start: "<< p->second<< endl;
        func(p->second);
        
    };
    unsigned int startThread(void (*_func)(int*) , const std::string& _name,int affinity){
        mutex.lock();
        int* stopp=new int(0);
        pthread_t thread;
        void* in=new std::pair<void (*)(int*),int* >(_func,stopp);
        
        pthread_create( &thread, NULL,LAUNCH_THREAD, in);
        pthread_setname_np(thread,_name.c_str());
        if (affinity>0){
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            static int cores=cv::getNumberOfCPUs();
            CPU_SET(affinity%cores, &cpuset);
            pthread_setaffinity_np(thread,1, &cpuset);
        }
        cout<<" Thread Requested: "<<_name<<" : "<<thread<<":"<< stopp<< endl;
        mymap[thread]=stopp;
        
        mutex.unlock();
        return thread;
    }
}



#else
#error OpenDTAM ImplThread not implemented on this system
#endif
