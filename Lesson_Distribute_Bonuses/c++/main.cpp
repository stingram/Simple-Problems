#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>



std::vector<int> bonus(const std::vector<int>& nums)
{
    std::vector<int> b(nums.size(), 1);

    for(int i=1;i<nums.size();i++)
    {
        if(nums[i-1] < nums[i])
        {
            b[i] = b[i-1]+1;
        }
    }
    for(int i = nums.size()-2;i>-1;i--)
    {
        if(nums[i+1] < nums[i])
        {
            b[i] = std::max(b[i],b[i+1]+1);
        }
    }

    return b;
}

void print(const std::vector<int> & bs)
{
    for(int b: bs)
    {
        std::cout << b << ", ";
    }
    std::cout << "\n";
}

int main()
{

    print(bonus({1, 2, 3, 4, 3, 1}));
    print(bonus({4,3,2,1}));
    return 0;
}
