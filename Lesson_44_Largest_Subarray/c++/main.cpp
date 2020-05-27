#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <numeric>
#include <algorithm>


class Solution{
    public:
    int max_subarray(const std::vector<int>& nums)
    {
        if(nums.size() == 0){
            return 0;
        }
        int res = nums[0];
        int currMax = 0;
        for(int n : nums)
        {
            if(currMax + n < 0)
            {
                currMax = 0;
                res = std::max(n, res);
            }
            else
            {
                currMax += n;
                res = std::max(currMax, res);
            }
        }
        return res;
    }
};

int main() {
    std::vector<int> nums = {-2,1,-3,4,-1,2,1,-5,4};
    std::cout << Solution().max_subarray(nums) << std::endl;
    return 0;
}



//     def max_subarray_v2(self, nums):
//         if len(nums) == 0:
//             return 0
//         res = nums[0]
//         currMax = 0
//         for n in nums:
//             if currMax + n < 0:
//                 currMax = 0
//                 res = max(n, res)
//             else:
//                 currMax += n
//                 res = max(currMax, res)
//         return res
    
    
// arr = [-2,1,-3,4,-1,2,1,-5,4]
// print(Solution().max_subarray(arr))
