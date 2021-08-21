#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>

class Solution
{
    private:
        std::vector<std::vector<int>> matrix;
    public:

    Solution(int m, int n)
    {
        matrix.resize(m, std::vector<int>(n,0));
        std::cout << "Rows:" << matrix.size() << "\n";
        std::cout << "Columns:" << matrix[0].size() << "\n";
        for(int i = 0; i<m; i++){
            for(int j = 0; j< n;j++){
                matrix[i][j] =0;
                std::cout << "val:" << matrix[i][j] << "\n";
            }
        }
        std::cout << "Done" << "\n";
    }
    int unique_paths(int m , int n)
    {      
        // set top row to one
        for(int j=0; j<n; j++)
        {
            matrix[0][j] = 1;
        }
         
        // set left column to one
        for(int i=0;i<m;i++)
        {
            matrix[i][0] = 1;
        }

        //update matrix
        for(int i = 1; i<m; i++){
            for(int j = 1; j<n; j++){
                matrix[i][j] = matrix[i-1][j]+ matrix[i][j-1];
                // std::cout << "val: " << matrix[i][j] << "\n";
            }
        }
                
        // return
        return matrix[m-1][n-1];
    }
};

int main(){
    int m,n;
    m=7;
    n=3;
    std::cout << Solution(m,n).unique_paths(7,3) << "\n";
}