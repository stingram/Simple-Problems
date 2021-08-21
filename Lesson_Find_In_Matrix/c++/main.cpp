#include <vector>
#include <iostream>
#include <functional>
#include <numeric>
#include <algorithm>

bool find_in_matrix(const std::vector<std::vector<int>>& matrix, int value)
{
    if(matrix.size() == 0)
    {
        return false;
    }
    int rows = matrix.size();
    int cols = matrix[0].size();

    int L,m,R, r, c, mval;
    L = 0;
    R = rows*cols - 1;


    while(L<R)
    {
        // calculate m
        m = (R+(L-1))/2;

        // convert m to 2D
        r = int(m /cols);
        c = m % cols;

        mval = matrix[r][c];

        if(mval == value)
        {
            return true;
        }
        // matrix value is too big, so we need reduce upper bound search
        if(mval > value)
        {
            R = m;
        }
        // matrix value is too small, so we need to increase lower bound search
        else
        {
            L = m +1;
        }

    }


    return false;
}


int main()
{
    std::vector<std::vector<int>> mat = {{1,3,5,8},
                                         {10,11,15,16},
                                         {24,27,30,31}};

    std::cout << std::boolalpha << find_in_matrix(mat, 4) << "\n"; // False

    std::cout << find_in_matrix(mat, 10) << "\n"; // True

    return 0;
}

