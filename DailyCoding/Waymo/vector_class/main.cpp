#include <iostream>
#include <algorithm>
#include <vector>

class MyVector {
    MyVector(){}
};

int main() {

    std::vector<int> source = {1,2,3,4,5};

    // This prevents a deep copy, which could be very expensive.
    // Ownership is transferred here, and source no longer can be used to reference data.
    std::vector<int> destination(std::move(source)); 

    std::cout << "destination: " << destination[0] << ".\n";
    destination[0] += 1;

    // Will make a seg fault
    // std::cout << "source: " << source[0] << ".\n";

    std::vector<int> destination1(destination); // this is a copy
    destination[0] += 1;
    std::cout << "destination: " << destination[0] << ".\n";
    std::cout << "destination1: " << destination1[0] << ".\n";


    return 0;
}