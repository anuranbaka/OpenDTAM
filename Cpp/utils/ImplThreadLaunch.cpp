#if !defined WIN32 && !defined WINCE
#  include "ImplThreadLaunch.hpp"

using namespace std;
namespace ImplThread{
    std::map<const pthread_t,typename cv::Ptr<int> > mymap;
    void stopAllThreads(){
        using namespace ImplThread;
        for ( std::map<const pthread_t,typename cv::Ptr<int> >::iterator it=mymap.begin(); it!=mymap.end();){
            pthread_t thread_id=it->first;
            *(it->second)=1;
            pthread_join(thread_id,NULL);
//            
//             it=mymap.begin();
        }
         mymap.erase(mymap.begin(),mymap.end());
    }
    void*LAUNCH_THREAD(void* in){
        std::pair<void (*)(int*),typename cv::Ptr<int> >* p=(std::pair<void (*)(int*),typename cv::Ptr<int> >*)in;
        void (*func)(int*)=p->first;
        func(p->second);
        
    };
    unsigned int startThread(void (*_func)(int*) , const std::string& _name,int affinity){
        cv::Ptr<int> stopp=new int(0);
        pthread_t thread;
        void* in=new std::pair<void (*)(int*),typename cv::Ptr<int> >(_func,stopp);
        pthread_create( &thread, NULL,LAUNCH_THREAD, in);
        pthread_setname_np(thread,_name.c_str());
        if (affinity>0){
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            static int cores=cv::getNumberOfCPUs();
            CPU_SET(affinity%cores, &cpuset);
            pthread_setaffinity_np(thread,1, &cpuset);
        }
        mymap[thread]=stopp;
        return thread;
    }
}



#else
#error OpenDTAM ImplThread not implemented on this system
#endif
