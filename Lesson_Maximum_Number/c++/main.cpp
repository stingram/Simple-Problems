#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>


bool compare_function(const int& lhs, const int& rhs)
    {
        std::string clhs, crhs;
        clhs = std::to_string(lhs)+ std::to_string(rhs);
        crhs = std::to_string(rhs)+ std::to_string(lhs);

        int ilhs = std::stoi(clhs);
        int irhs = std::stoi(crhs);

        return ilhs > irhs; //clhs.compare(crhs);
    };

std::string maximum_number(std::vector<int>& nums)
{
    std::sort(nums.begin(), nums.end(), &compare_function);

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