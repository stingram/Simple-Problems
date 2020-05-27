#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <numeric>
#include <cmath>

struct Shape
{
    double length, width;
    Shape(double l=1, double w=1)
    {
        length = l;
        width = w;
    }

    double area()
    {
        return length*width;
    }
private:
    int id;
};


struct Circle : Shape
{
    Circle(double width)
    {
        this->width = width;
    }
    double area()
    {
        return 3.14159 * std::pow((width/2), 2);
    }

};

class Customer
{
private:
    friend class Get_Customer_Data;
    std::string name;
public:
    Customer(std::string name)
    {
        this->name = name;
    }
};

class Get_Customer_Data
{
public:
    static std::string get_name(Customer& customer)
    {
        return customer.name;
    }
};


class shape
{
protected:
    double height;
    double width;
public:
    shape(double length)
    {
        height = length;
        width = length;
    }
    shape(double h, double w)
    {
        height = h;
        width = w;
    }
    virtual double area()
    {
        return height*width;
    }
};

class circle : public shape
{
public:
    circle(double w) : shape(w){}
    double area()
    {
        return 3.14159*std::pow((width/2), 2);
    }
};

void show_area(shape& shape)
{
    std::cout << "Area : " << shape.area() << "\n";
}

 // ----- ABSTRACT BASE CLASS -----------------------
class abstract_shape
{
public:
    virtual double area() = 0;
};

class circle_v2 : public abstract_shape
{
protected:
    double width;

public:
    circle_v2(double w)
    {
        width = w;
    }

    // override forces compiler to check if base class virtual
    // function is same as subclass
    // best to do it always
    double area() override 
    {
        return 3.14159*std::pow((width/2), 2);
    }
};

class rect : public abstract_shape
{
protected:
    double height, width;
public:
    rect(double h, double w)
    {
        height = h;
        width = w;
    }
    double area() override final // means that a subclass will use this
    {
        return height * width;
    }    
};

void show_area_v2(abstract_shape& shape)
{
    std::cout << "Area : " << shape.area() << "\n";
}

/*
class square_v3 : public rect
{
public:
    square_v3(double h, double w) : rect(h,w){}
    double area() override      // can't do this since final
    {
        return height * 2;
    }
};
*/

int main()
{
    Shape shape1(10,10);
    std::cout << "Square area: " <<
        shape1.area() << "\n";

    Circle circle1(10);
    std::cout << "Circle area: " <<
        circle1.area() << "\n";

    Shape rectangle{10, 15};    // can also construct with aggregate
    std::cout << "Rectangle area: " <<
        rectangle.area() << "\n";


    Customer tom("tom");
    Get_Customer_Data getName;
    std::cout << "Name : " <<
        getName.get_name(tom) << "\n";


    shape square(10,5);
    circle circ(10);
    show_area(square);
    show_area(circ);

    rect rect1(10,5);
    circle_v2 circ2(10);
    show_area_v2(rect1);
    show_area_v2(circ2);

    return 0;
}