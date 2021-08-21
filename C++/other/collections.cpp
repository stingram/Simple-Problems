#include <iostream>
#include <limits>
#include <vector>
#include <array>

namespace utilz
{

    void print_array(int arr[], int size)
    {
        //std::cout << sizeof(arr) << std::endl;          // This is actually the size of a pointer
        for(int i=0; i<size; i++)
        {
            std::cout << arr[i] << "\t";
        }
        std::cout << std::endl;
    }

    void print_vector(std::vector<int> & in_vec) // & pass by reference, don't need to do anything special when calling the function
    {
        for (int i=0; i< in_vec.size(); i++)
        {
            std::cout << in_vec[i] << "\t";
        }
        std::cout << "\n";
    }

    void print_array_std(const std::array<int, 20> & in_arr) // & pass by reference, don't need to do anything special when calling the function
    {
        for (int i=0; i< in_arr.size(); i++)
        {
            std::cout << in_arr[i] << "\t";
        }
        std::cout << "\n";
    }

    void test(std::vector<int> & data)
    {
        // range based for-loops - works for collections, like python! yay
        for(int n : data)
        {
            std::cout << n << "\t";
        }
    }

    void print_array_const(int data[], int size)
    {
        for (int i=0; i<size; i++)
        {
            std::cout << data[i] << "\t";
        }
        std::cout << "\n";
    }
};

int main()
{

    int twoD[][3] = {{1,2,3},{4,5,6},{7,8,9}};
    std::vector<std::vector<int>> vec2D = {{1,2,3},{4,5,6},{7,8,9}};

    for(int r =0;r<3;r++)
    {
        for(int c=0;c<3;c++)
        {
            // std::cout << twoD[r][c] << "\t";
            std::cout << vec2D[r][c] << "\t";
        }
        std::cout << std::endl;
    }

    std::vector<int> data = {1,2,3,4,5,6};


    utilz::test(data);
    /*
    std::array<int, 20> data_a = {1,2,3};
    print_array(data_a);

    std::vector<int> data = {1,2,3};
    print_vector(data); // notice it's still pass by reference because in function definition we use &, there's no need to use * operator, for example
    
    data.push_back(12);
    std::cout << data[data.size() -1] << std::endl;
    data.pop_back();
    std::cout << data.size() << std::endl; // size is different after pop_back 
    */



    /*
    const int SIZE = 100;
    int guesses[SIZE];
    int count = 0;
    for(int i=0; i<SIZE; i++)
    {
        if(std::cin >> guesses[i]) // return cin -> true if input worked, so we can stop by not giving an int
        {
            // inputn worked
            count++;
        }
        else
        {
            std::cin.clear();
            // std::cin.ignore(10000, '\n');
            // clear all the garbage by doing the following
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            break;
        }
    
        
    }
    print_array(guesses, count);
    */  
    /*
    int size = sizeof(guesses)/sizeof(guesses[0]);



    int guesses_v2[20];
    int num_elements = 5;
    
    guesses_v2[0] = 10;
    std::cin >> guesses_v2[0];
    std::cout << guesses_v2[3] << std::endl;

    std::cout << guesses[3] << std::endl; // guesses of 3
    guesses[3] = 300;
    std::cout << guesses[3] << std::endl;
    */
    return 0;
}