#include <vector>
#include <iostream>




int n_ways(int n)
{
    if(n == 0)
        return 0;
    std::vector<int> counts(n+1,0);
    
    counts[1] = 1;
    counts[2] = 2;

    for(int i=3;i<n+1;i++)
    {
        counts[i] = counts[i-1] + counts[i-2];
    }
    return counts[counts.size()-1];
}

int main()
{
    int n = 6;
    std::cout << n_ways(n) << "\n";
    return 0;
}