// REFERENCE
// https://blog.andreiavram.ro/job-scheduler-cpp/

#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <thread>
#include <chrono>
#include <condition_variable>

// Make generic function type
using Job = void (*)();

// Make generic error type, which is a function that
// takes a reference to an exception as the input
using Error = void(*)(const std::exception &);


// Scheduler class
class Scheduler
{
    private:
    std::condition_variable condition;
    std::mutex mutex;
    size_t size;
    const Error error;

    // value initialization, in most cases the value will be zero, so
    // this would be zero initialization
    // https://www.learncpp.com/cpp-tutorial/variable-assignment-and-initialization/
    size_t count{};

    public:

    Scheduler(size_t size, Error error)
    {
    }
    // https://www.geeksforgeeks.org/explicitly-defaulted-deleted-functions-c-11/
    // Do not allow constructor function that except null as second argument
    // A pointer that is null will checked at runtime
    // This is usually done to implicit functions
    Scheduler(size_t size, nullptr_t) = delete;

    void schedule(Job f, long n);

    // after scheduling the jobs, the scheduler can wait until all jobs are finished
    void wait();

    // Prevent copying
    Scheduler(const Scheduler &) = delete;

    void Scheduler::schedule(const Job f, long n)
    {
        std::unique_lock<std::mutex> lock(this->m)
    };

};


int main()
{
    Scheduler my_scheduler;
    return 0;
}