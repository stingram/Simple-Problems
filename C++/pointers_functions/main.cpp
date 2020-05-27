#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <ctime>

int main ()
{

    std::vector<int> my_vec(10);
    std::iota(std::begin(my_vec), std::end(my_vec), 0);
    for(int val : my_vec)
    {
        std::cout << val << " "; 
    }
    std::cout << "\n";
    return 0;
}