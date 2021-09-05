// # Given an array of integers, find the first missing positive integer in linear time and constant space. 
// # In other words, find the lowest positive integer that does not exist in the array. 
// # The array can contain duplicates and negative numbers as well.

// # For example, the input [3, 4, -1, 1] should give 2. The input [1, 2, 0] should give 3.

// # You can modify the input array in-place.
// # O(n) time, O(1) space

#include <vector>
#include <iostream>
#include <cmath>

class Solution
{
    public:

        int find_smallest(std::vector<int>& nums)
        {
            // set negative values to zero
            for(int i =0;i<nums.size();i++)
            {
                if(nums[i] < 0)
                {
                    nums[i] = 0;
                } 
            }

            // Mark values as negative or out of bands in input array
            for(int i=0;i<nums.size();i++)
            {
                // get index based on value (and subtract 1, because solution set = [1,len(nums)])
                // then to map into input array, we need to subtract 1
                int val = std::abs(nums[i]);
                int ind = val - 1;

                // check if in valid range
                // If we have a valid index then we mark value given by index as a negative value
                // Meaning, a negative value in our input array corresponds to a positive number
                // from our solution set exists
                if(ind >= 0 && ind < nums.size())
                {
                    // if value at in is 0, set that value to be negative and out of bounds
                    if(nums[ind] == 0)
                    {
                        nums[ind] = -1*(nums.size()+1);
                    }
                    // else, make sure value at ind will be negative
                    else
                    {
                        nums[ind] = -1*std::abs(nums[ind]);
                    }
                }
            }

            // loop through solution set and select first non-negative value
            for(int i=1;i<nums.size()+1;i++)
            {
                if(nums[i-1] >= 0)
                {
                    return i;
                }
            }
            // Got to end, must return nums.size()+1
            return nums.size()+1;
        }
};

int main()
{
    std::vector<int> nums = {1,2,0}; 
    std::cout << Solution().find_smallest(nums) << "\n"; // 3
    
    nums = {3, 4, -1, 1};
    std::cout << Solution().find_smallest(nums) << "\n"; // 2
    return 0;
}