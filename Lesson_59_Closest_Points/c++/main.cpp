#include <vector>
#include <cstdio>
#include <string>
#include <map>
#include <queue>
#include <iostream>


class myComparator 
{

private:
        float dist(const std::pair<int,int>& point)
        {
            return point.first*point.first + point.second*point.second;
        }
public: 
    int operator() (const std::pair<int,int>& point1, const std::pair<int,int>& point2) 
    { 
        return dist(point1) > dist(point2); 
    } 
}; 


class Solution
{
    private:


    public:
    std::vector<std::pair<int, int>> closest_point(const int k, const std::vector<std::pair<int, int>>& points)
    {
        std::priority_queue<std::pair<int,int>, std::vector<std::pair<int,int>>, myComparator> pq;
        for (auto point: points)
        {
            pq.push(point);
        }

        std::vector<std::pair<int, int>> result;

        for(int i=0;i<k;i++)
        {
            std::pair<int, int> point = pq.top();
            result.push_back(point);
            pq.pop();
        }
        return result;
    }

};

// import heapq
// from typing import List
// import math

// class Solution:
    
//     def _dist(self, point: List[int]):
//         # don't need sqrt, because it won't change the ordering of the points
//         return point[0]*point[0]+point[1]*point[1]
    
//     def closest_points(self, k: int, points: List[List[int]]) -> List[List[int]]:
//         '''
//         O(k*lg(n)) - Time
//         O(n) - Space
//         '''
//         # make heap array
//         data = []
        
//         for p in points:
//             # compute distance and push with point to array
//             data.append((self._dist(p),p))
        
//         # heapify
//         heapq.heapify(data)
//         result = []
        
//         # pop elements from heap
//         for i in range(k):
//             result.append(heapq.heappop(data)[1])
        
//         return result
    
//     def closest_points_slow(self, k: int, points: List[List[int]]) -> List[List[int]]:
//         '''
//         O(n*lg(n)) - Time
//         O(n) - Space
//         '''
//         return sorted(points, lambda x: self._dist(x))[:k]
    
// k = 3
// points = [[-1,-1], [1,1], [2,2], [3,3],[4,4]]

// print(Solution().closest_points(k, points))


int main()
{
    std::vector<std::pair<int,int>> points = {{-1,-1},{1,1},{2,2},{3,3}, {4,4}};
    int k = 3;
    std::vector<std::pair<int,int>> closest = Solution().closest_point(k, points);
    
    for(std::pair<int,int> point: closest)
    {
        std::cout << "Point: [" << point.first << ", " << point.second << "]" << "\n";
    }

    int* root = new int(10);
    int** node = &(root);
    int* root2 = root;
    std::cout << "Address where root is: " << &root << " \n";
    std::cout << "Address where node is: " << &node << " \n";
    std::cout << "Value stored at node: " << *node << "\n";
    std::cout << "Value stored at root: " << *root << "\n";
    std::cout << "Node value: " << node << " \n";
    std::cout << "Root value: " << root << " \n";
    std::cout << "Root2 value: " << root2 << "\n";
    
    return 0;
}