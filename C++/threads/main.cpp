#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <numeric>
#include <cmath>
#include <sstream>
#include <thread>
#include <chrono>
#include <ctime>
#include <mutex>

int get_random(int max)
{
    srand(time(NULL));
    return rand() % max;
}

void execute_thread(int id)
{
// Get current time
    auto nowTime = std::chrono::system_clock::now(); 
    
    // Convert to a time we can output
    std::time_t sleepTime = 
            std::chrono::system_clock::to_time_t(nowTime);
    
    // Convert to current time zone
    tm myLocalTime = *localtime(&sleepTime);
    
    // Print full time information
    std::cout << "Thread " << id << 
            " Sleep Time : " <<
            std::ctime(&sleepTime) << "\n";
    
    // Get separate pieces
    std::cout << "Month : " <<
            myLocalTime.tm_mon << "\n";
    std::cout << "Day : " <<
            myLocalTime.tm_mday << "\n";
    std::cout << "Year : " <<
            myLocalTime.tm_year + 1900 << "\n";
    std::cout << "Hours : " <<
            myLocalTime.tm_hour << "\n";
    std::cout << "Minutes : " <<
            myLocalTime.tm_min << "\n";
    std::cout << "Seconds : " <<
            myLocalTime.tm_sec << "\n\n";


    std::this_thread::sleep_for(std::chrono::seconds(get_random(3)));
    nowTime = std::chrono::system_clock::now();
    sleepTime = std::chrono::system_clock::to_time_t(nowTime);

        std::cout << "Thread " << id << 
            " Awake Time : " <<
            std::ctime(&sleepTime) << "\n";


}


int main()
{
    std::thread th1(execute_thread,1);
    th1.join();
    std::thread th2(execute_thread,2);
    th2.join();


    return 0;
}