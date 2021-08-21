#include <string>
#include <cstdlib>
#include <vector>
#include <iostream>

class Solution
{
    public:
    std::vector<std::string> list_ranges(const std::vector<int>& nums)
    {
        if(nums.size() == 0)
            return std::vector<std::string>({" "});
        std::vector<std::string> res;
        int low = nums[0];
        int high = nums[1];
        for(int n : nums){
            if(high + 1 < n)
            {
                res.push_back(std::to_string(low) + "-" + std::to_string(high));
                low = n;
            }
            high = n;

        }
        res.push_back(std::to_string(low) + "-" + std::to_string(high));
        return res;
    }
};


int main() {
    std::vector<int> nums = {0,1,2,5,7,8,9,9,10,11,15};
    std::vector<std::string> out = Solution().list_ranges(nums);
    for(auto r : out){
        std::cout << r << std::endl;
    }

    return 0;
}


// def list_ranges_v2(self, nums: List[int]) -> List[str]:
//     ranges =[]
//     if not nums:
//         return []
//     low = nums[0]
//     high = nums[0]
//     for n in nums:
//         if high + 1 < n:
//             ranges.append(str(low) + '-' + str(high))
//             low = high = n
//         else:
//             high = n
//     ranges.append(str(low) + '-' + str(high))
//     return ranges
