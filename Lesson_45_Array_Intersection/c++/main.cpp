#include <unordered_map>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <numeric>
#include <algorithm>

class Solution
{
    public:
    std::unordered_map<int, bool> build_dict(const std::vector<int>& arr)
    {
        std::unordered_map<int, bool> res;
        for(int num : arr)
        {
            if(res.find(num) == res.end())
            {
                res[num] = true;
            }
        }
        return res;
    }
    std::vector<int> array_intersection(const std::vector<int>& arr1,
                                        const std::vector<int>& arr2)
    {
        std::unordered_map<int, bool> set_dict;
        std::unordered_map<int, bool> a_dict = Solution::build_dict(arr1);
        std::vector<int> result;
        for(int num : arr2)
        {
            if(a_dict.find(num) != a_dict.end() && set_dict.find(num) == set_dict.end())
            {
                set_dict[num] = true;
            }
        }

        for(auto pair: set_dict)
        {
            int val = pair.first;
            result.push_back(val);
        }
        return result;
    }

};



// class Solution:
    
//     def _build_dict(self, a: List[int]) -> Dict[int, bool]:
//         a_dict = {}
//         for num in a:
//             if num not in a_dict:
//                 a_dict[num] = True
//         return a_dict
    
//     def array_intersection(self, a1: List[int], a2: List[int]) -> List[int]:
        
//         # build dictionary of longest list
//         set_dict = {}
//         a_dict = self._build_dict(a1)
//         for n in a2:
//             if n in a_dict and n not in set_dict:
//                 set_dict[n] = True
        
//         return [k for k in set_dict.keys()]
    
// a1 = [4,9,5,9]
// a2 = [9,4,9,8,4]
// print(Solution().array_intersection(a1, a2))


int main()
{

    std::vector<int> a1 = {4,9,5,9};
    std::vector<int> a2 = {9,4,9,8,4};
    auto res = Solution().array_intersection(a1,a2);
    for(int n: res)
    {
        std::cout << n << " ";
    }


    return 0;
}
