#include <vector>
#include <iostream>


void sort_nums(std::vector<int>& nums)
{
    int low = 0;
    int index = low;
    int high = nums.size()-1;
    int temp;

    while(index<high)
    {
        if(nums[index] == 1)
        {
            // swap elements
            temp = nums[index];
            nums[index] = nums[low];
            nums[low] = temp;
            // increment low
            low++;
            index++;
        }
        if(nums[index] == 2)
        {
            index++;
        }
        if(nums[index] == 3)
        {
            // swap elements
            temp = nums[index];
            nums[index] = nums[high];
            nums[high] = temp;
            // decrement high
            high--;
        }
    }


    return;
}

int main()
{
    std::vector<int> nums = {3, 3, 2, 1, 3, 2, 1};
    sort_nums(nums);
    for(int num: nums)
    {
        std::cout << num << ", ";
    }
    std::cout << "\n";
    return 0;
}
// [1, 1, 2, 2, 3, 3, 3]