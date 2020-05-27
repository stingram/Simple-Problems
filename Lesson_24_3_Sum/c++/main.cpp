#include <cstdlib>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>





class three_sum
{
    public:
        std::vector<std::vector<int>> three_sum_v1(const std::vector<int> nums, int k)
        {
            // brute force
            std::vector<std::vector<int>> output;
            return output;
        }

        // TIME - O(n*lg(n)) + O(n^2)
        // SPACE - O(1)
        std::vector<std::vector<int>> three_sum_v2(std::vector<int>& nums)
        {
            std::vector<std::vector<int>> output;
            sort(nums.begin(), nums.end());
            int j, k;
            std::vector<int> temp;
            for (int i=0; i < nums.size() -2; i++)
            {
                if(i > 0 && (nums[i] == nums[i-1]))
                    continue;
                // TWO SUM
                j = i + 1;
                k = nums.size() - 1;
                while(j < k)
                {
                    if(nums[i] + nums[j] + nums[k] == 0)
                    {
                        temp = {nums[i], nums[j], nums[k]};
                        output.push_back(temp);
                        while(j < k && (nums[j] == nums[j + 1]))
                            j += 1;
                        while(j < k && (nums[k] == nums[k - 1]))
                            k -= 1;
                        j += 1;
                        k -= 1;
                    }
                    else if(nums[i] + nums[j] + nums[k] > 0)
                        k -= 1;
                    else
                        j += 1;
                }
                // end two sum
            }
            return output;
        }
};


int main()
{
    return 0;
}