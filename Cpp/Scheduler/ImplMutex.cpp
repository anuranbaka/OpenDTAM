#include "ImplMutex.hpp"
//Special purpose mutex and cond var implementation
// The cond var does not guarantee fairness, and may wake threads that were queued AFTER when signal was called instead of the correct one
// For me these failure modes don't matter
#if !defined WIN32 && !defined WINCE
#  include <pthread.h>
#endif


#if defined WIN32 || defined _WIN32 || defined WINCE
    #  include <windows.h>
    struct ImplMutex::Impl
    {
        void init()
        {
    #if (_WIN32_WINNT >= 0x0600)
            ::InitializeCriticalSectionEx(&sl, 1000, 0);
    #else
            ::InitializeCriticalSection(&sl);
    #endif
            refcount = 1;
        }
        void destroy() { DeleteCriticalSection(&sl); }

        void lock() { EnterCriticalSection(&sl); }
        bool trylock() { return TryEnterCriticalSection(&sl) != 0; }
        void unlock() { LeaveCriticalSection(&sl); }

        CRITICAL_SECTION sl;
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

    #elif defined __linux__ && !defined ANDROID

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

        void lock() {pthread_mutex_lock(&sl); }
        bool trylock() { return pthread_mutex_trylock(&sl) == 0; }
        void unlock() { pthread_mutex_unlock(&sl); }

        pthread_mutex_t sl;
        int refcount;
        
        
        
    };

#endif

void ImplMutex::init()
{
    impl = new Impl;
    impl->init();
}
void ImplMutex::destroy()
{
    impl->destroy();
    delete impl;
    impl = NULL;
}
void ImplMutex::lock() { impl->lock(); }
void ImplMutex::unlock() { impl->unlock(); }
bool ImplMutex::trylock() { return impl->trylock(); }



#if defined WIN32 || defined _WIN32 || defined WINCE
    //Reimplement pthreads cond vars as shown in http://www.cse.wustl.edu/~schmidt/win32-cv-1.html
    typedef CRITICAL_SECTION pthread_mutex_t;
    typedef struct
    {
        u_int waiters_count_;
        // Count of the number of waiters.
        
        CRITICAL_SECTION waiters_count_lock_;
        // Serialize access to <waiters_count_>.

        // Same as before...
    } pthread_cond_t;
    
    int 
    pthread_cond_init (pthread_cond_t *cv, 
                    const pthread_condattr_t *)
    {
        // Initialize the count to 0.
        cv->waiters_count_ = 0;

        // Create an auto-reset and manual-reset event, as before...
    }
    int 
    pthread_cond_wait (pthread_cond_t *cv,
                    pthread_mutex_t *external_mutex)
    {
        // Avoid race conditions.
        EnterCriticalSection (&cv->waiters_count_lock_);
        cv->waiters_count_++;
        LeaveCriticalSection (&cv->waiters_count_lock_);

        // It's ok to release the <external_mutex> here since Win32
        // manual-reset events maintain state when used with
        // <SetEvent>.  This avoids the "lost wakeup" bug...
        LeaveCriticalSection (external_mutex);

        // Wait for either event to become signaled due to <pthread_cond_signal>
        // being called or <pthread_cond_broadcast> being called.
        int result = WaitForMultipleObjects (2, ev->events_, FALSE, INFINITE);

        EnterCriticalSection (&cv->waiters_count_lock_);
        cv->waiters_count_--;
        int last_waiter =
            result == WAIT_OBJECT_0 + BROADCAST 
            && cv->waiters_count_ == 0;
        LeaveCriticalSection (&cv->waiters_count_lock_);

        // Some thread called <pthread_cond_broadcast>.
        if (last_waiter)
            // We're the last waiter to be notified or to stop waiting, so
            // reset the manual event. 
            ResetEvent (cv->events_[BROADCAST]); 

        // Reacquire the <external_mutex>.
        EnterCriticalSection (external_mutex, INFINITE);
    }
    int 
    pthread_cond_signal (pthread_cond_t *cv)
    {
        // Avoid race conditions.
        EnterCriticalSection (&cv->waiters_count_lock_);
        int have_waiters = cv->waiters_count_ > 0;
        LeaveCriticalSection (&cv->waiters_count_lock_);

        if (have_waiters)
            SetEvent (cv->events_[SIGNAL]);
    }
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
    impl = new Impl;
    impl->init();
}
void ImplCondVar::destroy()
{
    impl->destroy();
    delete impl;
    impl = NULL;
}
void ImplCondVar::signal(){impl->signal();}
void ImplCondVar::broadcast(){impl->broadcast();}
void ImplCondVar::wait(ImplMutex& mutex){impl->wait(mutex);}

