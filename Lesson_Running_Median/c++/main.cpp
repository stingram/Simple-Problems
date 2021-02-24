#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <queue>


float compute_median(std::priority_queue<float, std::vector<float>, std::greater<float>>& min_heap, std::priority_queue<float>& max_heap)
{
    if(min_heap.size()>max_heap.size())
    {
        return min_heap.top();
    }
    if(min_heap.size()<max_heap.size())
    {
        return max_heap.top();
    }
    return (min_heap.top()+max_heap.top())/2.0;
}

void add(float num, std::priority_queue<float, std::vector<float>, std::greater<float>>& min_heap, std::priority_queue<float>& max_heap)
{
    // put first element into max heap, will rebalance later
    if(min_heap.size() == 0 && max_heap.size() == 0)
    {
        max_heap.push(num);
        return;
    }

    // if we have elements, we need to know current median, and then put in appropriate queue
    if(num > compute_median(min_heap, max_heap))
    {
        // put into min heap
        min_heap.push(num);
    }
    else
    {
        max_heap.push(num);
    }

    return;
}

void rebalance(std::priority_queue<float, std::vector<float>, std::greater<float>>& min_heap, std::priority_queue<float>& max_heap)
{
    if(min_heap.size()>max_heap.size()+1)
    {
        max_heap.push(min_heap.top());
        min_heap.pop();
    }
    if(max_heap.size()>min_heap.size()+1)
    {
        min_heap.push(max_heap.top());
        max_heap.pop();
    }
    return;
}



std::vector<float> running_median(const std::vector<float>& nums)
{
    
    // results
    std::vector<float> medians;
    
    // construct both heaps
    std::priority_queue<float> max_heap;
    std::priority_queue<float, std::vector<float>, std::greater<float>> min_heap;

    
    for(float num: nums)
    {
        // add num to heaps
        add(num, min_heap, max_heap);

        // rebalance
        rebalance(min_heap, max_heap);

        std::cout << min_heap.size() << ", " << max_heap.size() << "\n";


        // compute median
        medians.push_back(compute_median(min_heap,max_heap));

    }
    return medians;
}



int main()
{


    std::vector<float> medians = running_median({2, 1, 4, 7, 2, 0, 5});
    for(float median: medians)
    {
        std::cout << median << ", ";
    }
    std::cout << "\n";
    // [2, 1.5, 2, 3, 2, 2, 2]
    return 0;
}