#ifndef IMPL_MUTEX_HPP
#define IMPL_MUTEX_HPP
//Special purpose mutex and cond var implementation
// The cond var does not guarantee fairness, and may wake threads that were queued AFTER when signal was called instead of the correct one
// For me these failure modes don't matter
#if !defined WIN32 && !defined WINCE
#  include <pthread.h>
#endif

class ImplMutex
{
    friend class ImplCondVar;//so ImplCondVar::Impl::wait can access
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

class ScopeLock{
    ImplMutex* m;
public:
ScopeLock(ImplMutex& mutex)
{
    m=&mutex;
    mutex.lock();
}
~ScopeLock()
{
    m->unlock();
}

};

#endif
