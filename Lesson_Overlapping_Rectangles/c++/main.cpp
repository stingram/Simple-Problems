#include <iostream>
#include <string>
#include <cmath>
#include <cstdlib>
#include <algorithm>

class Rectangle
{
    private:


    public:
        float x_min;
        float y_min;
        float x_max;
        float y_max;
        
        Rectangle(float x_min, float y_min, float x_max, float y_max)
        {
            this->x_min = x_min;
            this->y_min = y_min;
            this->x_max = x_max;
            this->y_max = y_max;
        }

        double area()
        {
            if((x_max > x_min) && (y_max > y_min))
            {
                return (x_max-x_min)*(y_max-y_min);
            } 
            return 0;
        }
};

double overlap_rect(Rectangle& a, Rectangle& b)
{
    float x_min = std::max(a.x_min,b.x_min);
    float y_min = std::max(a.y_min,b.y_min);
    float x_max = std::min(a.x_max,b.x_max);
    float y_max = std::min(a.y_max,b.y_max);

    return Rectangle(x_min,y_min,x_max,y_max).area();
}

int main()
{
    Rectangle a = Rectangle(1,1,3,3);
    Rectangle b = Rectangle(1,1,4,2);

    std::cout << "Overlapping area: " << overlap_rect(a,b) << ".\n";
}