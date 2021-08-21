#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

class Solution
{
    public:
    std::vector<std::vector<int>> merge_intervals(std::vector<std::vector<int>>& intervals)
    {
        std::sort(intervals.begin(), intervals.end());
        std::vector<std::vector<int>> results;

        for(auto interval : intervals)
        {
            // if we have no interval at OR 
            // if the End time in the last item of result is less
            // than the start time of this current interval
            if(results.size() < 1 || results.back()[1] < interval[0])
            {
                // add this interval to results since there isn't 
                // an overlap
                results.push_back(interval);
            }
            else
            {
                // make the end time of the last item in result be the
// #            // maximum of either the end time of the last item in result
// #            // OR the end time of this interval we are are looking at
                results.back()[1] = std::max(results.back()[1], interval[1]);
            }
            
        }
        return results;

    }
};

int main()
{
    std::vector<std::vector<int>> intervals = {{1,3},{2,6},{8,10},{15,18}};
    std::vector<std::vector<int>> res = Solution().merge_intervals(intervals);
    for(std::vector<int> l : res)
    {
        std::cout << "[" << l[0] << "," << l[1] << "]" << "\n";
    }


    return 0;
}
