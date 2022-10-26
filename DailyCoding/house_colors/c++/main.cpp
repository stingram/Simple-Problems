// # A builder is looking to build a row of N houses that can be of K different colors. 
// # He has a goal of minimizing cost while ensuring that no two neighboring houses are of the same color.

// # Given an N by K matrix where the nth row and kth column represents the cost to build 
// # the nth house with kth color, return the minimum cost which achieves this goal.

// # Space is O(k), Time is O(N*k*k)


#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

class Solution
{
    private:
    int min_cost_prev(std::vector<int> prev_row, int j)
    {
        prev_row.erase(prev_row.begin()+j);
        return *std::min_element(prev_row.begin(),prev_row.end());
    }
    public:
    Solution()
    {}


    int find_min_total_cost(const std::vector<std::vector<int>>& house_cost_matrix)
    {
        
        int N = house_cost_matrix.size();
        int k = house_cost_matrix[0].size();

        std::vector<int> total_prev_costs(house_cost_matrix[0]);
        std::vector<int> total_curr_costs = std::vector<int>(k,0);


        for(int i = 1; i<N;i++)
        {
            for(int j=0;j<k;j++)
            {
                total_curr_costs[j] = house_cost_matrix[i][j] + min_cost_prev(total_prev_costs, j);
            }
            // update prev with curr
            std::copy(total_curr_costs.begin(), total_curr_costs.end(), total_prev_costs.begin());
        }

        return *std::min_element(total_curr_costs.begin(),total_curr_costs.end());

    }


};


int main()
{
    std::vector<std::vector<int>> house_cost_matrix = {{2,5,3},{1,6,2},{2,7,1},{4,3,3}};
    std:: cout << "Min cost is: " << Solution().find_min_total_cost(house_cost_matrix) << ".\n"; // Should print 8
    return 0;
}