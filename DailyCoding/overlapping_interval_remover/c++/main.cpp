#include <vector>
#include <iostream>
#include <algorithm>
#include <tuple>

// Given a collection of intervals, find the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.
// Intervals can "touch", such as [0, 1] and [1, 2], but they won't be considered overlapping.
// For example, given the intervals (7, 9), (2, 4), (5, 8), return 1 as the last interval can be removed and the first two won't overlap.
// The intervals are not necessarily sorted in any order.

// Solution Explanation
// https://www.youtube.com/watch?v=nONCGxWoUfM
 
bool sortAscending(const std::pair<int,int>& p1, const std::pair<int,int>& p2)
{
   return std::tie(p1.second, p1.first) < std::tie(p2.second, p2.first);
}

// TIME COMPLEXITY - O(N*lg(N)) due to sort
// SPACE - O(1)

// Remember we can pass as const because we are doing in-place sort
int num_intervals_to_remove(std::vector<std::pair<int,int>>& intervals)
{
    // result
    int num_to_remove = 0;
    // set up for loop
    int prev_end = 0;
    
    // Need to sort in ascending order
    std::sort(intervals.begin(), intervals.end());
    for(const std::pair<int,int>& pair : intervals)
    {
        // if we have overlap
        if(pair.first < prev_end)
        {
            // remove the segment that ends later
            prev_end = std::min({prev_end,pair.second});
            // Increment counter of intervals removed
            num_to_remove += 1;
        }
        else
        {
            // no overlap, update end of our non-overlapping section
            prev_end = pair.second;
        }
    }


    return num_to_remove;
}

int main()
{
    std::vector<std::pair<int,int>> intervals = {{7,9},{2,4},{5,8}};
    std::cout << "Num Intervals to Remove: " << num_intervals_to_remove(intervals) << ".\n"; 
    return 0;
}