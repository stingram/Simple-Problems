#include <vector>
#include <algorithm>
#include <iostream>

// O(N) in space and time for setup
//O(1) in space and time for sum calculation


class Solution
{

    private:
    std::vector<int> sums;

    public:
    Solution(const std::vector<int>& L)
    {
        sums.push_back(0); 
        // compute sums
        int ind = 0;
        for(auto&val: L)
        {
            sums.push_back(sums[ind]+val);
            ind++;
        }
    }

    int sum(const int i, const int j)
    {
        if(i>=j)
        {
            return 0;
        }
        return sums[j]-sums[i];
    }
};

int main()
{
    // L = [1, 2, 3, 4, 5], sum(1, 3) should return sum([2, 3]), which is 5.
    std::vector<int> L = {1,2,3,4,5};
    std::cout << Solution(L).sum(0,5);

    return 0;
}