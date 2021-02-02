#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>


std::string maximum_number(const std::vector<int>& nums)
{
    std::sort(nums.begin(), nums.end(), [] (const int& lhs, const int& rhs)
    {
        //std::string clhs, crhs;
        //clhs = lhs+rhs;
        //crhs = rhs+lhs;
        return lhs < rhs; //clhs.compare(crhs);
    });
    std::string res = "";
    for(int num : nums)
    {
        res += std::to_string(num);
    }
    return res;

}

int main()
{
    std::vector<int> nums = {17, 7, 2, 45, 72};
    std::string res = maximum_number(nums);
    std::cout << res << "\n";
    // 77245217
    return 0;
}