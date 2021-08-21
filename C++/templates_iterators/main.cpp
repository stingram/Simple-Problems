#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <numeric>
#include <cmath>
#include <sstream>
 
#include <deque>
#include <iterator>

#include "animal.h"

#define PI 3.14159
#define AREA_CIRCLE(radius) (PI * (std::pow(radius, 2)))


// We use templates to create functions or classes
// that can work with many types
// Templates differ from function overloading in that
// instead of having a function that does similar
// things with different objects a template does the
// same thing with different objects
 
// This says this is a function template that generates
// functions that except 1 parameter
template<typename T>
void times_2(T val)
{
    std::cout << val << " * 2 = " <<
                    val * 2 << "\n";
}

// Receive multiple parameters and return a value
template<typename T>
T add(T val, T val2)
{
    return val + val2;
}

// Work with chars and strings
template<typename T>
T max(T val, T val2)
{
    return (val < val2) ? val2 : val;
}

// Template classes are classes that can work with 
// different data types
 
// You can define that you may receive parameters
// with different types, but they don't have to
// be different
template<typename T, typename U>
class Person
{
    public:
        T height;
        U weight;
        static int num;
        Person(T h, U w)
        {
            height = h;
            weight = w;
            num++;
        }
        void get_data()
        {
            std::cout << "Height : " <<
                height << " and weight : " << weight << "\n";
        }
};

// required for statics!!
// You have to initialize static class members
template<typename T, typename U> int Person<T,U>::num;





int main()
{
    Animal spot;
    spot.name = "Spot";
    std::cout << "The animal is named " <<
            spot.name << "\n";


    std::cout << "Circle area: " <<
                AREA_CIRCLE(5) << "\n";

    times_2(5);
    times_2(5.3);

    std::cout << "5 + 4 = " << add(5,4) << "\n";
    std::cout << "5.5 + 4.6 = " << add(5.5,4.6) << "\n";

    std::cout << "Max of 4 or 8 = " << max(4,8) << "\n";
    std::cout << "Max of A or B = " << max('A','B') << "\n";
    std::cout << "Max of Dog or Cat = " << max("Dog","Cat") << "\n";


    Person<double, double> mikeTyson(5.83, 216);
    mikeTyson.get_data();
    std::cout << "Number of people : " << mikeTyson.num << "\n";

    // Iterators are used to point at container
    // memory locations
    std::vector<int> num2 = {1,2,3,4};

    // Define an iterator as the same type
    std::vector<int>::iterator itr;

    // Refer to the vectors begin and end while
    // incrementing the iterator
    for(itr = num2.begin(); itr < num2.end(); itr++)
    {
        // Get value at the pointer
        std::cout << *itr << "\n";
    }

    // You can also increment a set number of spaces
    // Create an iterator and point it at the beginning
    // of the vector
    std::vector<int>::iterator itr2 = num2.begin();
    advance(itr2, 2);

    // Advance 2 spaces
    std::cout << *itr2 << "\n";

    // Next works like advance, but it returns an 
    // iterator
    auto itr3 = next(itr2,1);
    std::cout << *itr3 << "\n";

    // Previous moves a set number of indexes and
    // returns an iterator
    auto itr4 = prev(itr2,1);
    std::cout << *itr4 << "\n"; 

    // You can also insert at a defined index
    std::vector<int> nums3 = {1,4,5,6};
    std::vector<int> nums4 = {2,3};
    auto itr5 = nums3.begin();
    advance(itr5, 1);
    copy(nums4.begin(), nums4.end(), inserter(nums3, itr5));
    for(int &i: nums3)
        std::cout << i << "\n";


    return 0;
}