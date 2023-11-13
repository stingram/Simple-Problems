#include <algorithm> // copy
#include <iostream> // boolalpha, cin, cout, streamsize
#include <iterator> // back_inserter, istream_iterator
#include <limits> // numeric_limits
#include <sstream> // istringstream
#include <string> // getline, string
#include <vector> // vector

int binary_search(std::vector<int> arr, int target) {
    // WRITE YOUR BRILLIANT CODE HERE
    if(std::size(arr)  == 0)
        return -1;
    
    int arr_size = std::size(arr);
    int left = 0;
    int right = arr_size - 1;
    int mid;
    while(left<=right)
    {
      mid = left + (right-left)/2;
      if(arr[mid] == target)
      {
          return mid;
      }
      else if(arr[mid] > target){
           right = mid - 1;   
      }
      else {
          left = mid + 1; 
      }
    }
    
    return -1;
}


int main() {
    int target = 9;
    std::vector<int> arr = {1,3,5,7,8};
    int res = binary_search(arr, target);
    std::cout << "Index of " << target << " is: " << res << ".\n";
}