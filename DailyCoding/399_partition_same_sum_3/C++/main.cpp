// O(N) in space and time for solution

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iterator>


class Solution
{

    public:
        Solution()
        {}

        std::vector<std::vector<int>> split_list(const std::vector<int> in_list)
        {
            std::vector<std::vector<int>> ret_val = {{},{},{}};
            int sum = std::accumulate(in_list.begin(),in_list.end(),0);
            if(sum % 3 != 0){
                return ret_val;
            }
            
            int target_sum = sum / 3;
            // setup for finding ranges
            int temp_sum = 0;
            bool LAST_SEGMENT = false;

            // work around since no enumerate
            int start_position = 0;
            int end_position = start_position + 1;
            int segment_number = 0;
            for(const auto& val :in_list)
            {
                temp_sum += val;

                if(temp_sum == target_sum)
                {
                    if(!LAST_SEGMENT)
                    {
                        std::copy(in_list.begin()+start_position, in_list.begin()+end_position, std::back_inserter(ret_val[segment_number]));
                        LAST_SEGMENT = true;
                        temp_sum = 0;
                        start_position = end_position;
                    }
                    else
                    {
                        std::copy(in_list.begin()+start_position, in_list.begin()+end_position, std::back_inserter(ret_val[segment_number]));
                        std::copy(in_list.begin()+end_position, in_list.end(), std::back_inserter(ret_val[segment_number+1]));
                        return ret_val;
                    }
                    segment_number++;
                }
                end_position++;
            }

        }
};

int main()
{
    std::vector<int> in_list = {3, 5, 8, 0, 8};
    std::vector<std::vector<int>> out_list = Solution().split_list(in_list);
    for(auto& segment:out_list)
    {
        for(auto& val:segment)
        {
            std::cout << val << ",";
        }
        std::cout << "\n";
    }
    return 0;
}