#include <vector>
#include <iostream>
#include <numeric>
#include <cmath>
#include <algorithm>




int lis(const std::vector<int>& nums)
{
    int n = nums.size();
    std::vector<int> cache(n, 1);
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<i;j++)
        {
            if(nums[i]> nums[j])
            {
                cache[i] = std::max(cache[i], cache[j]+1);
            }
        }
    }

    auto it = std::max_element(cache.begin(), cache.end()); // c++11
    return *it;
}






int main()
{
    // Should print 5
    std::cout << lis({0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3}) << "\n";
    
    return 0;
}
