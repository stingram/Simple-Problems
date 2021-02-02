#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath> 

int perfect_squares(int n)
{
    std::vector<int> D(n+1,0);
    int minD;
    int tmin;
    for(int k=1;k<n+1;k++)
    {
        minD = std::numeric_limits<int>::max();
        for(int i=1;i<int(std::pow(k,0.5))+1;i++)
        {
            tmin = 1 + D[k-std::pow(i,2)];
            if(tmin<minD)
            {
                minD = tmin;
            }
        }
        D[k] = minD;
    }
    return D[n];
}

int main()
{
    int n = 13;
    std::cout << perfect_squares(n) << "\n";
    return 0;
}