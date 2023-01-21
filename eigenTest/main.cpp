#include "iostream"
#include <stdlib.h>
#include "/mnt/e/College/eigen/Eigen/Dense"

typedef Eigen::Matrix<signed char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix8i;

//batches:2, channels: 2, rows: 5, cols: 5
int input[100] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99};

// batches:3, channels: 2, rows: 3, cols: 3
int kernel[54] = {1};

// batches:2, channels: 2, rows: 5, cols: 5
int output[50] = {0};


struct test_struct
{
    int a;
    void* arr;
};

test_struct return_struct(void)
{
    test_struct my_eig;
    int num_elements = 16;
    signed char vals[num_elements] = {0};
    my_eig.arr = malloc(16);
    //my_eig.arr = vals;
    for(int i=0;i<num_elements;i++)
    {
        vals[i] = (signed char)0x04;
    }
    memcpy(my_eig.arr,vals,num_elements);
    // my_eig.arr = vals;
    return my_eig;
}


int main()
{
    test_struct my_eig = return_struct();
    Eigen::Map<Matrix8i> eig((signed char*)my_eig.arr, 4, 4);
    std::cout << eig << std::endl;    
    free(my_eig.arr);
}