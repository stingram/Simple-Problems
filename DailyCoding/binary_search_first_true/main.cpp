#include <algorithm> // copy
#include <iostream> // boolalpha, cin, cout
#include <iterator> // back_inserter, istream_iterator
#include <sstream> // istringstream
#include <string> // getline, string
#include <vector> // vector

int find_boundary(std::vector<bool> arr) {
    int L = 0;
    int R = std::size(arr);
    int mid;
    int first = -1;
    while(L<=R)
    {
        mid = L + (R-L) / 2;
        if(arr[mid] == true)
        {
             R = mid - 1;
             first = mid;
        }
        else if(arr[mid] == false)
        {
            L = mid + 1;
        }
    }
    
    return first;
}


int main()
{
    std::vector<bool> arr = {true, true, true, true, true};
    std::cout << "First true is at index: " << find_boundary(arr) << ".\n";
    return 0;
}