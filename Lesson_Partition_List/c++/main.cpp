#include <vector>
#include <string>
#include <iostream>


void partition(std::vector<int>& nums, int k)
{
    int i = 0;
    int low = 0;
    int high = nums.size() - 1;
    int temp, n;
    while(i<=high)
    {
        n = nums[i];
        if(n > k)
        {
            nums[i] = nums[high];
            nums[high] = n;
            high--;
        }
        if(n < k)
        {
            nums[i] = nums[low];
            nums[low] = n;
            low++;
            i++;
        }
        if(n==k)
        {
            i++;
        }
    }
}



int main()
{
    std::vector<int> nums = {8,9,2,4,1,0};
    int k = 3;
    partition(nums,k);
    for (int n: nums)
    {
        std::cout << n << ", ";
    }
    std::cout << "\n";
    return 0;
}