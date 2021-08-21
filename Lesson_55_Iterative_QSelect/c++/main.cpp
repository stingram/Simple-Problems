#include <vector>
#include <cstdlib>
#include <iostream>
#include <string>
#include <numeric>
#include <algorithm>


int partition(std::vector<int>& arr, int L, int R)
{
    int i = L - 1;
    int p = arr[R];
    for(int j=L;j<R;j++)
    {
        if(arr[j] <= p)
        {
            i++;
            std::swap(arr[i],arr[j]);
        }
    }
    std::swap(arr[i+1],arr[R]);
    return i+1;
}

int quick_select(std::vector<int>& arr, int kth)
{
    int k = arr.size() - kth;
    int L = 0;
    int R = arr.size() - 1;
    int p;
    while(L<=R)
    {
        p = partition(arr,L,R);
        if(p == k)
        {
            return arr[p];
        }
        else if (p > k)
        {
            R = p - 1;
        }
        else
        {
            L = p + 1;
        }
    }
    return -1;
}

int main()
{
    std::vector<int> arr = {8,7,2,3,4,1,5,6,9,0};
    int k = 3;
    std::cout << k << "th largest: " << quick_select(arr,k) << "\n";
    return 0;
}
// def partition(arr, L, R):
//     i = L - 1
//     p = arr[R]
//     for j in range(L,R):
//         if arr[j] <= p:
//             i += 1
//             arr[i], arr[j] = arr[j], arr[i]
//     arr[i+1], arr[R] = arr[R], arr[i+1]
//     return i+1

// def quick_select(arr, k):
    
//     # Kth LARGEST
//     k = len(arr) - k
    
//     L = 0
//     R = len(arr) - 1
    
//     while L <= R:
//         p = partition(arr, L,R)
//         if p == k:
//             return arr[k]
//         elif p > k:
//             R = p - 1
//         else:
//             L = p + 1
//     return -1

// print(quick_select([8,7,2,3,4,1,5,6,9,0], 3))