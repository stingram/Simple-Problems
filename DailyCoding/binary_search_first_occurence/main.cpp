#include <algorithm> // copy
#include <iostream> // boolalpha, cin, cout, streamsize
#include <iterator> // back_inserter, istream_iterator
#include <limits> // numeric_limits
#include <sstream> // istringstream
#include <string> // getline, string
#include <vector> // vector

int find_first_occurrence(std::vector<int> arr, int target) {
    // WRITE YOUR BRILLIANT CODE HERE
    int L = 0;
    int R = std::size(arr) - 1;
    int first = -1;
    int mid;
    while(L<=R)
    {
        mid = L + (R-L)/2;
        if(arr[mid]==target)
        {
            first = mid;
            R = mid - 1;
        }
        else if(arr[mid] < target)
        {
             L = mid + 1;   
        }
        else if(arr[mid] > target)
        {
            R = mid - 1;
        }
    }
    return first;
}

int main()
{
    int target = 1;
    std::vector<int> arr = {4, 6, 7, 7, 7, 20};
    std::cout << "First occurrence of " << target << " is at index: " << find_first_occurrence(arr,target) << ".\n";
    return 0;

}