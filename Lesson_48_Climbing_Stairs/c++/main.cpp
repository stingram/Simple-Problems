#include <cstdlib>
#include <iostream>
#include <string>
#include <numeric>
#include <algorithm>

class Solution
{
    public:
    int climb_stairs(int n)
    {
        if(n == 0 || n == 1){
            return 1;
        }
        int first = 1;
        int second = 1;
        int third;
        for(int i=2;i<n+1;i++)
        {
            third = first + second;
            first = second;
            second = third;
        }
        return third;
    }
};


int main()
{
    int n = 8;
    std::cout << Solution().climb_stairs(n) << "\n";
    return 0;
}