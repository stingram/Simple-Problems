#include <iostream>


int fib(int n)
{
    if(n==0 || n == 1)
    {
        return n;
    }
    int n_1 = 1;
    int n_2 = 0;
    int res;
    for(int i=2;i<n+1;i++)
    {
        res = n_1 + n_2;
        n_2 = n_1;
        n_1= res;
    }
    return res;
}

int main()
{
    std::cout << fib(10) << "\n";
    return 0;
}