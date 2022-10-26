// This problem was asked by Airbnb.
// Given a list of integers, write a function
// that returns the largest sum of non-adjacent numbers. Numbers can be 0 or negative.

// For example, [2, 4, 6, 2, 5] should return 13,
// since we pick 2, 6, and 5. [5, 1, 1, 5] should return 10, since we pick 5 and 5.

// Follow-up: Can you do this in O(N) time and constant space?

// Need two values as we traverse array:
// excl
// incl

// during first iteration:
// incl = arr[0]
// excl = 0

// during all subsequent iterations:
// temp = incl
// incl = max(incl, excl + arr[i])
// excl = temp

#include <vector>
#include <iostream>


class Solution
{
    public:
    int max_non_adj_sum(const std::vector<int>& nums)
    {
        if(nums.size() < 1)
        {
            return 0;
        }
        int incl = nums[0];
        int excl = 0;
        int temp;

        for(int i = 1; i < nums.size(); i++)
        {
            temp = incl;
            incl = std::max(incl, excl+nums[i]);
            excl = temp;
        }

        return std::max(incl,excl);
    }
};


int main()
{
    std::vector<int> nums = {2, 4, 6, 2, 5};
    std::cout << "Max sum: " << Solution().max_non_adj_sum(nums) << ".\n"; // 13

    nums.clear();
    nums = {5, 1, 1, 5};
    std::cout << "Max sum: " << Solution().max_non_adj_sum(nums) << ".\n"; // 10

    return 0;
}