
#ifndef MATH_UTILS
#define MATH_UTILS
struct Rectangle 
{
    double width;
    double length;
};

namespace utils{
    double power(double base, int pow=2); // default here, but not in definiton

    double area(Rectangle rect);

    double area(const double length, double width=2); // default here, but not in definiton

};


#endif