#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <algorithm>



class ListSum
{
    public:
    std::vector<int> nums;
    std::vector<int> acc;
    ListSum(std::vector<int>& nums)
    {
        this->nums = nums;
        this->acc.push_back(0);
        int tsum = 0;
        for(int num: nums)
        {
            tsum += num;
            this->acc.push_back(tsum);
        }
    }
    int opt_sum(int ind1, int ind2)
    {
        return this->acc[ind2] - this->acc[ind1];
    }

};


int main()
{
    std::vector<int> vnums = {1,2,3,4,5,6,7};
    ListSum* nums = new ListSum(vnums);
    // range is finding sum from [ind1, ind2), not [ind1, ind2]
    std::cout << nums->opt_sum(2,5) << "\n"; // should be 12
    return 0;
}