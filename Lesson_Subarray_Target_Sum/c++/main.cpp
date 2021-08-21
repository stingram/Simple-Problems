#include <unordered_map>
#include <iostream>
#include <string>
#include <vector>



std::vector<int> find_subarray(const std::vector<int>& nums, int target)
{
    std::vector<int> res;
    std::unordered_map<int,int> sums;
    int i =0;
    int acc = 0;
    for(int num : nums)
    {
        acc += num;
        // put into dictionary
        sums[acc] = i;

        // check if we have got subarray
        if(sums.find(acc-target)!= sums.end())
        {
            // found it
            int start = sums[acc-target]+1;
            int end = i + 1;
            res.insert(res.end(),nums.begin()+start,nums.begin()+end);
            return res;
        }

        // increment i
        i++;
    }
}



int main()
{
    std::vector<int> sub_array = find_subarray({1, 3, 2, 5, 7, 2}, 14);
    for (int num: sub_array)
    {
        std::cout << num << ",";
    }
    std::cout << "\n";
    return 0;
}