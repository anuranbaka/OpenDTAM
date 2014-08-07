#if !defined WIN32 && !defined WINCE
#  include <pthread.h>
#endif

class ImplMutex
{
    friend class ImplCondVar;
public:
    ImplMutex() { init(); }
    ~ImplMutex() { destroy(); }

    void init();
    void destroy();

    void lock();
    bool trylock();
    void unlock();

    struct Impl;
protected:
    Impl* impl;

private:
    ImplMutex(const ImplMutex&);
    ImplMutex& operator = (const ImplMutex& m);
};

#if defined WIN32 || defined _WIN32 || defined WINCE
#  include <windows.h>
struct ImplMutex::Impl
{
    void init()
    {
#if (_WIN32_WINNT >= 0x0600)
        ::InitializeCriticalSectionEx(&cs, 1000, 0);
#else
        ::InitializeCriticalSection(&cs);
#endif
        refcount = 1;
    }
    void destroy() { DeleteCriticalSection(&cs); }

    void lock() { EnterCriticalSection(&cs); }
    bool trylock() { return TryEnterCriticalSection(&cs) != 0; }
    void unlock() { LeaveCriticalSection(&cs); }

    CRITICAL_SECTION cs;
    int refcount;
};

#ifndef __GNUC__
static int _interlockedExchangeAdd(int* addr, int delta)
{
#if defined _MSC_VER && _MSC_VER >= 1500
    return (int)_InterlockedExchangeAdd((long volatile*)addr, delta);
#else
    return (int)InterlockedExchangeAdd((long volatile*)addr, delta);
#endif
}
#endif // __GNUC__

/*#elif defined __APPLE__

#include <libkern/OSAtomic.h>

struct ImplMutex::Impl
{
    void init() { sl = OS_SPINLOCK_INIT; refcount = 1; }
    void destroy() { }

    void lock() { OSSpinLockLock(&sl); }
    bool trylock() { return OSSpinLockTry(&sl); }
    void unlock() { OSSpinLockUnlock(&sl); }

    OSSpinLock sl;
    int refcount;
};

/#elif defined __linux__ && !defined ANDROID

struct ImplMutex::Impl
{
    void init() { pthread_spin_init(&sl, 0); refcount = 1; }
    void destroy() { pthread_spin_destroy(&sl); }

    void lock() { pthread_spin_lock(&sl); }
    bool trylock() { return pthread_spin_trylock(&sl) == 0; }
    void unlock() { pthread_spin_unlock(&sl); }

    pthread_spinlock_t sl;
    int refcount;
};
*/
#else

struct ImplMutex::Impl
{
    void init() { pthread_mutex_init(&sl, 0); refcount = 1; }
    void destroy() { pthread_mutex_destroy(&sl); }

    void lock() { pthread_mutex_lock(&sl); }
    bool trylock() { return pthread_mutex_trylock(&sl) == 0; }
    void unlock() { pthread_mutex_unlock(&sl); }

    pthread_mutex_t sl;
    int refcount;
};

#endif

void ImplMutex::init()
{
    impl = (Impl*)malloc(sizeof(Impl));
    impl->init();
}
void ImplMutex::destroy()
{
    impl->destroy();
    free(impl);
    impl = NULL;
}
void ImplMutex::lock() { impl->lock(); }
void ImplMutex::unlock() { impl->unlock(); }
bool ImplMutex::trylock() { return impl->trylock(); }


class ImplCondVar{
public:
    ImplCondVar() { init(); }
    ~ImplCondVar() { destroy(); }
    
    void init();
    void destroy();
    
    void signal();
    void broadcast();
    void wait(ImplMutex& mutex);
    
    struct Impl;
    protected:
        Impl* impl;

    private:
        ImplCondVar(const ImplCondVar&);
        ImplCondVar& operator = (const ImplCondVar& m);
};

#if defined WIN32 || defined _WIN32 || defined WINCE
    #error No implementation of OpenDTAM for Windows yet!
#else
    struct ImplCondVar::Impl{
        void init() { pthread_cond_init(&c, 0); refcount = 1; }
        void destroy() { pthread_cond_destroy(&c); }

        void wait(ImplMutex& mutex) { pthread_cond_wait(&c,&mutex.impl->sl); }
        void signal() { pthread_cond_signal(&c); }
        void broadcast() { pthread_cond_broadcast(&c); }

        pthread_cond_t c;
    int refcount;
    };
#endif


void ImplCondVar::init()
{
    impl = (Impl*)malloc(sizeof(Impl));
    impl->init();
}
void ImplCondVar::destroy()
{
    impl->destroy();
    free(impl);
    impl = NULL;
}
void ImplCondVar::signal(){impl->signal();}
void ImplCondVar::broadcast(){impl->broadcast();}
void ImplCondVar::wait(ImplMutex& mutex){impl->wait(mutex);}

