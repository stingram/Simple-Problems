#include <algorithm> // copy
#include <iostream> // boolalpha, cin, cout, streamsize
#include <iterator> // back_inserter, istream_iterator
#include <limits> // numeric_limits
#include <sstream> // istringstream
#include <string> // getline, string
#include <vector> // vector

int first_not_smaller(std::vector<int> arr, int target) {
    // WRITE YOUR BRILLIANT CODE HERE
    int L = 0;
    int R = std::size(arr);
    int first = -1;
    int mid;
    while(L<=R)
    {
        mid = L + (R-L)/2;
        if(arr[mid]>=target)
        {
            first = mid;
            R = mid - 1;
        }
        else
        {
            L = mid + 1;
        }
    }

    return first;
}



int main()
{
    std::vector<int> arr = {1,2,2,2,2,2,2,3,5,8,8,10};
    int target = 2;
    std::cout << "First number >= " << target << " is at index: " << first_not_smaller(arr,target) << ".\n";
    return 0;
}