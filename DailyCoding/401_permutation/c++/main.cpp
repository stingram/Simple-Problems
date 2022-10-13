#include <vector>
#include <iostream>
#include <string>
#include <numeric>
#include <algorithm>

class Solution
{

    public:
    std::pair<int,int> next_cycle(const std::vector<int>& P)
    {
        int i = 0;
        std::pair<int, int> locations = {-1,-1};
        for(auto val: P)
        {
            if(val!=-1)
            {
                locations.first = i;
                locations.second = val;
            }
            i++;
        }
        return locations;
    }

    void cycle(std::vector<std::string>& arr, std::vector<int>& P, const std::pair<int,int> locations)
    {
        std::string temp;
        int old_loc = locations.first;
        int new_loc = locations.second;
        int start_loc = old_loc;
        while(new_loc!=-1)
        {
            temp = arr[new_loc];
            arr[new_loc] = arr[start_loc];
            arr[start_loc] = temp;

            P[old_loc] = -1;
            old_loc = new_loc;
            new_loc = P[new_loc];
        }
    }
 
    void permute(std::vector<std::string>& arr, std::vector<int>& P)
    {
        std::pair<int, int> locations  = {-1,-1};
        while(true)
        {
            locations = next_cycle(P);
            if(locations.first != -1)
            {
                cycle(arr,P, locations);
            }
            else
            {
                break;
            }
        }
        return;
    }
};
// https://stackoverflow.com/a/23397700
template<typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
    out << "{";
    size_t last = v.size() - 1;
    for(size_t i = 0; i < v.size(); ++i) {
        out << v[i];
        if (i != last) 
            out << ", ";
    }
    out << "}";
    return out;
}

int main()
{
    std::vector<int> P = {2, 1, 0};
    std::vector<std::string> arr = {"a", "b", "c"};
    std::vector<std::string> orig(arr);
    Solution().permute(arr, P);
    std::cout << "Permutation of array " << orig <<  "is : " << arr << ".\n";

    P = {4, 0, 1, 2, 3, 5, 6};
    arr = {"a", "b", "c", "d", "e", "f", "g"};
    orig.clear();
    std::copy(arr.begin(), arr.end(), std::back_inserter(orig)); 
    Solution().permute(arr, P);
    std::cout << "Permutation of array " << orig <<  "is : " << arr << ".\n";

    P = {4, 0, 1, 2, 3, 6, 5};
    arr = {"a", "b", "c", "d", "e", "f", "g"};
    orig.clear();
    std::copy(arr.begin(), arr.end(), std::back_inserter(orig)); 
    Solution().permute(arr, P);
    std::cout << "Permutation of array " << orig <<  "is : " << arr << ".\n";
    return 0;
}