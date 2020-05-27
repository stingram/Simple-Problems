#include "math_utils.h"

namespace utils{
    double power(double base, int pow) // default is in header file
    {
        int total = 1;
        for(int i=0;i<pow;i++)
        {
            total *= base;
        }
        return total;
    }

    double area(Rectangle rect)
    {
        return rect.length * rect.width;
    }

    double area(const double length, double width) // default is in header file
    {
        return length*width;
    }
};

