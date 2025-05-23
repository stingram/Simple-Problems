#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>
#include <ctime>
#include <string>
#include <numeric>


class Box
{
    public:
        double length, width, breadth;
        std::string box_string;

        Box()
        {
            length = 1, width = 1, breadth = 0;
        }
        Box(double l, double w, double b)
        {
            length = l, width = w, breadth = b;
        }
        Box& operator ++ ()
        {
            length++;
            width++;
            breadth++;
            return *this;
        }
        operator const char*()
        {
            std::ostringstream boxStream;
            boxStream << "Box : " <<
                length << ", " <<
                width << ", " <<
                breadth;
            box_string = boxStream.str();
            return box_string.c_str();
        }

        Box operator + (const Box& box2)
        {
            Box boxsum;
            boxsum.length = length + box2.length;
            boxsum.width = width + box2.width;
            boxsum.breadth = breadth + box2.breadth;
            return boxsum;
        }

        double operator [] (int x)
        {
            if(x==0)
            {
                return length;
            }
            else if(x==1){
                return width;
            }
            else if(x==2)
            {
                return breadth;
            }
            else{
                return 0;
            }
        }

        bool operator == (const Box& box2)
        {
            return ((length == box2.length) && 
                        (width == box2.width) &&
                        (breadth == box2.breadth));
        }

        bool operator < (const Box& box2)
        {
            double thissize = length+width+breadth;
            double box2size = box2.length+box2.width+box2.breadth;
            if(thissize < box2size)
            {
                return true;
            }
            else
            {
                return false;
            }
            
        }

        bool operator > (const Box& box2)
        {
            double thissize = length+width+breadth;
            double box2size = box2.length+box2.width+box2.breadth;
            if(thissize > box2size)
            {
                return true;
            }
            else
            {
                return false;
            }
            
        }

        void operator = (const Box& boxtocopy)
        {
            length = boxtocopy.length;
            width = boxtocopy.width;
            breadth = boxtocopy.breadth;
        }


    private:
};

int main()
{
    Box box(10,10,10);
    ++box;
    std:: cout << box << "\n";
    Box box2(5,5,5);
    std::cout << "Box1 + Box2 = " << box + box2 << "\n";

    std::cout << "Box length: " <<
                box[0] << "\n";
    std::cout << std::boolalpha;
    std::cout << "Arge boxes equal : " <<
            (box == box2) << "\n";
    std::cout << "Is box < box2 : " <<
            (box < box2) << "\n";
    std::cout << "Is box > box2 : " <<
            (box > box2) << "\n";
    box = box2;
    std::cout << box << "\n";


    return 0;
}
