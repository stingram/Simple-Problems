#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>

int jump_to_end(const std::vector<int>& nums)
{
    int n = nums.size();
    std::vector<int> hops(n,std::numeric_limits<int>::max());

    hops[0] = 0;
    int i = 0;
    for(int num : nums)
    {
        for(int j=1;j<num+1;j++)
        {
            if(i+j<n)
            {
                hops[i+j] = std::min(hops[i+j],hops[i]+ 1);
            }
            else{
                break;
            }
        }
        i++;
    }


    return hops[n-1];
}

int main()
{

    std::vector<int> nums = {3, 2, 5, 1, 1, 9, 3, 4};
    std::cout << jump_to_end(nums) << "\n";
    return 0;
}

// print(jump_to_end_v2([3, 2, 5, 1, 1, 9, 3, 4]))
// 2