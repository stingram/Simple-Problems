#include <cstdlib>
#include <iostream>
#include <string>
#include <ctime>
#include <random>
#include <algorithm>
#include <vector>



int select(int list[], int l , int r, int index)
{
    
    if(l==r)
        return list[l];
    // select pivot
    srand(time(NULL));
    int pivot_index = -1;
    while(pivot_index < l || pivot_index > r)
        pivot_index = l + (rand() % int (r - l + 1));

    // Swap
    std::swap(list[pivot_index],list[l]);

    // partition block
    int i = l;
    for(int j=l+1;j<r+1;j++)
    {
        if(list[j] < list[i])
        {
            i++;
            std::swap(list[j], list[i]);
        }
    }
    std::swap(list[l], list[i]);

    // recursively partition on one side
    if(index == i)
        return list[i];
    else if (index < i)
        return select(list, l, i-1, index);
    else
        return select(list, i+1, r, index);

}

int kth_larest_qs(int arr[], int n, int l, int r, int k)
{
    return select(arr, 0, n- 1, n - k);
}


int main()
{
    int arr[] = {10,4,5,8, 3, 26, 2};
    int n = sizeof(arr) / sizeof(arr[0]);
    int k = 3;
    std::cout << "Kth largest: "  << kth_larest_qs(arr, n, 0, n-1, k) << 
            "\n";


    std::vector<int> input = {10,4,5,8, 3, 26, 2, 31};
    std::make_heap(input.begin(), input.end());
    for(int i=0;i<k-1;i++)
    {
        std::pop_heap(input.begin(),input.end());
        input.pop_back();
    }
    std::pop_heap(input.begin(),input.end());
    int out = input.back();
    std::cout << "TOP K: " << out << "\n";

    std::vector<int> easy = {10,4,5,8, 3, 26, 2, 31};
    std::sort(easy.begin(), easy.end(), std::greater<int>());

    std::cout << "Easy k: " << easy[k-1] << "\n";

    return 0;
}

