#include "ImplMutex.hpp"

template <class T>
class StallableSynchronizedQueue{
public:
    //void push(T &in);
    //T pop();
    //void readStall();
    //void readUnstall();
    ImplMutex mutex;
    
private:
    int _readStall;
    ImplCondVar cond;
    std::deque<T> q;
public:
    void push(T& in){
        mutex.lock();
        q.push_back(in);
        cond.signal();
        mutex.unlock();
    }
    
    T pop(){// a real pop that takes the element off the end
        T out;
        mutex.lock();
        while(q.size()==0||_readStall){
            cond.wait(mutex);
        }
        out=q.front();
        q.pop_front();
        if (q.size()>0){
            cond.signal();
        }
        mutex.unlock();
        return out;
    }
    void readStall(){
        _readStall=1;
    }
    void readUnstall(){
        _readStall=0;
        mutex.lock();
        cond.broadcast();//will this really work?
        mutex.unlock();
    }
};

template <class T>
class StallableSynchronizedStack{
public:
    //void push(T& in);
    //T pop();
    //std::vector<T> peekn(int i=1);
    //void readStall();
    //void readUnstall();
    ImplMutex mutex;
    std::deque<T > q;
    
private:
    int _readStall;
    ImplCondVar cond;
    
public:
    void push(T& in){
        mutex.lock();
        q.push_back(in);
        cond.signal();
        mutex.unlock();
    }
    
    T pop(){// a real pop that takes the element off the end
        T out;
        mutex.lock();
        while(q.size()==0||_readStall){
            cond.wait(mutex);
        }
        out=q.back();
        q.pop_back();
        if (q.size()>0){
            cond.signal();
        }
        mutex.unlock();
        return out;
    }
    T peek(){
        mutex.lock();
        while(q.size()<1||_readStall){
            cond.wait(mutex);
        }
        T out = q.back();
        mutex.unlock();
        return out;
    }
    std::vector<T> peekn(int n=1){
        std::vector<T> out;
        mutex.lock();
        
        while(q.size()<n||_readStall){
            cond.wait(mutex);
        }
        std::reverse_iterator<typename std::deque<T>::iterator> it=q.rbegin();
        for(int i=0;i<n;i++){
            out.push_back(q[q.size()-n+i]);
        }

        mutex.unlock();
        return out;
    }
    void readStall(){
        _readStall=1;
    }
    void readUnstall(){
        _readStall=0;
        mutex.lock();
        cond.broadcast();//will this really work?
        mutex.unlock();
    }
};
