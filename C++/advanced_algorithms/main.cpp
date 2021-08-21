#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>
#include <ctime>
#include <string>
#include <numeric>
#include <fstream>
#include <functional>
#include <random>

double mult_by_2(double num)
{
    return num * 2;
}

// 2. Functions can receive other functions
// the first double is the return type followed
// by the data types for the parameter
double do_math(std::function<double(double)> func, double num)
{
    return func(num);
}

double mult_by_3(double num)
{
    return num * 3;
}


bool is_it_odd(int num)
{
    if(num % 2 == 0)
    {
        return false;
    }
    return true;
}


std::vector<int> change_list(std::vector<int> list,
                    std::function<bool(int)> func)
    {
        std::vector<int> odd_list;
        for(auto i: list)
        {
            if(func(i))
            {
                odd_list.push_back(i);
            }
        }
        return odd_list;
    }


std::vector<char> make_HT_list(std::vector<char>  poss_vals, int num)
{
    std::vector<char> out_list;
    srand(time(NULL));
    for(int x =0; x<num; x++)
    {
        int randind = rand() % 2;
        out_list.push_back(poss_vals[randind]);
    }
    return out_list;
}

int get_number_of_matches(std::vector<char> list, char val)
{
    int num = 0;
    for(char c: list)
    {
        if(c == val)
        {
            num++;
        }
    }
    return num;
}


int main()
{
    auto times2 = mult_by_2;
    std::cout << "5 * 2 = " <<
            times2(5) << "\n";

    std::cout << "6 * 2 = " <<
        do_math(times2, 6) << "\n";

    std::vector<std::function<double(double)>> funcs(2);
    funcs[0] = mult_by_2;
    funcs[1] = mult_by_3;
    std::cout << "2 times 10 = " << funcs[0](10) << "\n";


    // ------ PROBLEM ------------------
    std::vector<int> list_of_nums {1,2,3,4,5};
    std::vector<int> odd_list = change_list(list_of_nums,
                    is_it_odd);
    std::cout << "List of odd\n";
    for(auto i : odd_list)
    {
        std::cout << i << "\n";
    }

    // ----- PROBLEM 2 ------------------
    std::vector<char> possible_values{'H', 'T'};
    std::vector<char> heads_tails_list = make_HT_list(possible_values,
                                                      100);
    std::cout << "Number of heads : " <<
        get_number_of_matches(heads_tails_list, 'H') << "\n";
    std::cout << "Number of tails : " <<
        get_number_of_matches(heads_tails_list, 'T') << "\n";

    return 0;
}