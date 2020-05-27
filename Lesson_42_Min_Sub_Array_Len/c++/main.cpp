#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <limits>
#include <algorithm>

class Solution
{
    public:
        int min_subarray_len_v2(const std::vector<int>& arr, int s)
        {
            float res = std::numeric_limits<float>::max();
            int sum = 0;
            int left = 0;
            int right = 0;
            while(right < arr.size())
            {
                sum += arr[right];
                while(sum >= s){
                    // updating the results
                    res = std::min(res, float(right - left + 1));
                    // increment left pointer
                    sum -= arr[left];
                    left += 1;
                }
                // increment right pointer since our sub array is smaller than s
                right += 1;
            }
            if(res == std::numeric_limits<float>::max()){
                return 0;
            }
            return int(res);
        }
};

int main()
{
    std::vector<int> arr = {2,1,3,2,4,3};
    int res = Solution().min_subarray_len_v2(arr, 7);
    std::cout << "Result: " << res << "\n";
    return 0;
}

    
    // def min_subarray_len_v2(self, nums, s):
    //     res = float('inf')
    //     sum = 0
    //     left = 0
    //     right = 0
    //     while right < len(nums):
    //         sum += nums[right]
    //         while sum >= s:
    //             # updating the results
    //             res = min(res, right - left + 1)
    //             # increment left pointer
    //             sum -= nums[left]
    //             left += 1
    //         # increment right pointer since sum is smaller than s
    //         right += 1
    //     if res == float('inf'):
    //         return 0
    //     return res