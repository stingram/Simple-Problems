// Given an array of integers, return a new array such that each element at index i 
// of the new array is the product of all the numbers in the original array except the one at i.

// For example, if our input was [1, 2, 3, 4, 5], the expected output would be 
// [120, 60, 40, 30, 24]. If our input was [3, 2, 1], the expected output would be [2, 3, 6].


// [1,2,3,4,5]
// [120,60,20,5,1] -> a
// [1,1,2,6,24] -> b
// prod[i] = a[i]*b[i]
#include <vector>
#include <iostream>
#include <string>



class Solution
{

    public:
    Solution()
    {

    }
    std::vector<int> products(const std::vector<int>& nums)
    {
        std::vector<int> res(nums.size(),1);
        std::vector<int> left(nums.size(),1);
        std::vector<int> right(nums.size(),1);        
        
        // build right
        for(int i=nums.size()-2;i>-1;i--)
        {
            right[i]=right[i+1]*nums[i+1];
        }

        // build left
        for(int i=1;i<nums.size();i++)
        {
            left[i]=left[i-1]*nums[i-1];
        }

        // build result
        for(int i=0;i<nums.size();i++)
        {
            res[i] = left[i]*right[i];
        }
        return res;
    }

};


int main()
{
    std::vector<int> nums = {1,2,3,4,5};
    std::vector<int> res = Solution().products(nums);
    std::cout << "Results: ";
    for(int i: res)
    {
        std::cout << i <<", ";
    }
    std::cout << "\n";

    return 0;
}