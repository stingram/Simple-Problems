#include <vector>
#include <iostream>
#include <string>

int bst(const std::vector<int>& nums)
{
    int m;
    int L = 0;
    int R = nums.size();
    while(L<R)
    {
        m = int((L + (R-1))/2);
        if(nums[m] == m)
            return m;
        else if(nums[m]> m)
            R = m - 1;
        else if(nums[m]< m)
            L = m + 1;
    }
    return -1;
}



int main()
{
    std::cout << bst({-5, 1, 3, 4}) << "\n";
    // 1

    std::cout << bst({-5, 1, 3, 4}) << "\n";
    // 1
    return 0;
}