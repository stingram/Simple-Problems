#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

class Solution
{
    public:
    int buy_sell(const std::vector<int>& prices)
    {
        if(prices.size() < 1)
        {
            return 0;
        }
        int res = 0;
        int max_val = 0;
        int profit;

        for(int i=prices.size()-1;i>=0;i--)
        {
            max_val = std::max(max_val, prices[i]);
            profit = max_val - prices[i];
            res = std::max(res, profit);
        }
        return res;
    }
};

int main()
{
    std::vector<int> prices = {7,1,5,3,6,4};
    std::cout << "Profit: " << Solution().buy_sell(prices) << "\n";

    prices.assign({7,6,4,3,1});
    std::cout << "Profit: " << Solution().buy_sell(prices) << "\n";

    return 0;
}
