// # cons(a, b) constructs a pair, and car(pair) and cdr(pair) returns the first and last element of that pair.
// # For example, car(cons(3, 4)) returns 3, and cdr(cons(3, 4)) returns 4.

// # GIVEN
// def cons(c, d):
//     def pair(f):
//         return f(c, d)
//     return paira

#include <functional>
#include <iostream>

// SOLUTION REFERENCE
// https://stackoverflow.com/a/22623564

// Captures
// https://www.learncpp.com/cpp-tutorial/lambda-captures/


// cons function
auto cons = [](auto a, auto b)
    {
        // Needs = in brackets
        return [=](auto func) 
        {
            return func(a,b);
        };
    };


// return first element
auto car = [](auto func) 
    {
        return func([] (auto a, auto b) 
        {
            return a;
        });
    };

// return second element
auto cdr = [](auto func) 
    {
        return func([] (auto a, auto b) 
        {
            return b;
        });
    };



int main()
{
    auto p = cons(3,4);
    std::cout << "car: " << car(p) << ".\n";
    std::cout << "cdr: " << cdr(p) << ".\n";
    return 0;
}