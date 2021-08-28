// Given a list of numbers and a number k, return whether any two numbers from the list add up to k.
// For example, given [10, 15, 3, 7] and k of 17, return true since 10 + 7 is 17.
// Bonus: Can you do this in one pass?

#include <unordered_map>
#include <vector>
#include <iostream>


class Solution
{
    public:
    Solution()
    {

    }
    bool sum_k(const std::vector<int>& arr, const int k)
    {
        std::unordered_map<int,bool> comps;
        for(int num: arr)
        {
            if(comps.find(k-num) == comps.end())
            {
                comps[k-num] = true;
            }
            if(comps.find(num) != comps.end())
            {
                return true;
            }
        }
        return false;
    }
};

int main()
{
    std::vector<int> arr = {3,3,3,3};
    int k = 3;
    std::cout << std::boolalpha << Solution().sum_k(arr,k);
    return 0;
}