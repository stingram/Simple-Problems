#include <cstdlib>
#include <vector>
#include <iostream>
#include <queue>
#include <map>

class comparator
{
    public:
        int operator() (const std::pair<int, int>& p1,
                        const std::pair<int, int>& p2)
        {
            return p1.first > p2.first;
        }
};

class Solution
{
    private:
        std::map<int, int> counts;
        std::priority_queue<std::pair<int, int>,
                            std::vector<std::pair<int,int>>, comparator> heap;

    public:
        std::vector<int> top_k_frequent(std::vector<int> nums, int k)
        {
            // build dictionary of counts
            for(auto i : nums)
            {
                if(this->counts.find(i) == counts.end())
                {
                    counts[i] = 1;
                }
                else
                {
                    counts[i] += 1;
                }
                
            }
            // construct min heap
            for (auto pair : counts)
            {
                std::cout << "Pair: " << pair.first << ", " <<
                            pair.second << "\n";
                heap.push(std::pair<int,int>(pair.second, pair.first));
                if(heap.size() > k)
                    heap.pop();
            }

            // return as vector 
            std::vector<int> res(heap.size());
            std::pair<int, int> temp;
            int i = 0;
            while(heap.size()>0)
            {
                temp = heap.top();
                std::cout << "Temp: " << temp.first <<
                        ", " << temp.second << "\n";
                res[i] = temp.first;
                heap.pop();
                i++;
            }
            return res;
        }
};
int main ()
{
    std::vector<int> nums = {1,1,1,2,2,3,4,5,5,5,5,5,6};
    int k = 2;
    Solution my_sol;
    std::vector<int> output = my_sol.top_k_frequent(nums, k);
    for(auto i : output)
    {
        std::cout << i << "\n";
    }


    return 0;
}