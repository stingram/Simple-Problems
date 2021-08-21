#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <string>


std::vector<int> sorted_squares(std::vector<int>& nums)
{
    // initialization
    std::vector<int> res(nums.size(),0);
    int ind_i = -1; 
    int ind_j = nums.size();

    // get first non-negative index
    int c = 0;
    for(int num : nums)
    {
        if(num>=0)
        {
            ind_i = c - 1;
            ind_j = c;
            break;
        }
        c++;
    }

    // merge and compute squares
    int k =0;
    int n,p;
    while(ind_i>=0 && ind_j<nums.size())
    {
        n = nums[ind_i]*nums[ind_i];
        p = nums[ind_j]*nums[ind_j];
        if(n <p)
        {
            res[k]=n;
            ind_i -= 1;
        }
        else
        {
            res[k]=p;
            ind_j += 1;
        }
        k++;
        
    }
    // if left over negatives
    while(ind_i>=0)
    {
        n = nums[ind_i]*nums[ind_i];
        res[k]=n;
        ind_i -= 1;
        k++;
    }

    // if left over positives
    while(ind_j<nums.size())
    {
        p = nums[ind_j]*nums[ind_j];
        res[k]=p;
        ind_j += 1;
        k++;
    }
    return res;

}


int main()
{
    std::vector<int> nums = {-5,-3,-2,-1,0,3,5};
    std::vector<int> res = sorted_squares(nums);
    for(int r : res)
    {
        std::cout << r << ", ";
    }
    std::cout << "\n";


    return 0;
}