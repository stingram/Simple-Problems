#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <string>
#include <numeric>
#include <sstream>
#include <ctime>
#include <cmath>


int fibo(int index);

double area(double radius);
double area(double height, double width);

int main()
{
    int index;
    std::cout << "Get Fibonacci Index : ";
    std::cin >> index;
    std::cout << "Fib(" << index << ") = " << fibo(index);

    std::cout << "Area circle (c) or Rectangle (r) : ";
    char areaType;
    std::cin >> areaType;

    switch(areaType)
    {
        case 'c':
            std::cout << "Enter radius : ";
            double radius;
            std::cin >> radius;
            std::cout << "Area = " << area(radius) << "\n";
            break;
        case 'r':
            std::cout << "Enter height : ";
            double height, width;
            std::cin >> height;
            std::cout << "Enter width : ";
            std::cin >> width;
            std::cout << "Area = " << area(height, width) << "\n";
            break;
        default:
            std::cout << "Please enter c or r \n";    
    }

}

int fibo(int index)
{
    if(index < 2)
    {
        return index;
    }
    else
    {
        return fibo(index - 1) + fibo(index - 2);
    }
    
}

double area(double radius)
{
    return 3.14159 * std::pow(radius, 2);
}

double area(double height, double width)
{
    return height * width;
}