#include <sstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <functional>

std::vector<int> generate_random_vector(int amount, int min, int max);

std::vector<int> gen_fibs(const int count);

int main()
{
    std::vector<int> rand_vec = generate_random_vector(10,1,50);
    std::sort(rand_vec.begin(), rand_vec.end(),
                [](int x, int y){return x < y;}); // ascending

    for(auto val: rand_vec)
    {
        std::cout << val << "\n";
    }

    rand_vec = generate_random_vector(10,1,50);
    std::vector<int> even_vals;
    std::copy_if(rand_vec.begin(),rand_vec.end(),
                std::back_inserter(even_vals),
                [](int x) {return (x % 2) == 0;});

    int sum = 0;
    std::for_each(rand_vec.begin(), rand_vec.end(),
                    [&] (int x) {sum += x;}); // capture by reference, so we can change sum and get it out
    // guess if we didn't do that, sum would be zero
    // the & is for sum, the lambda function only would have access to
    // x 


    int divisor;
    std::vector<int> vec_vals;
    std::cout << "List of values divisable by : ";
    std::cin >> divisor;

    std::copy_if(rand_vec.begin(),rand_vec.end(),
                std::back_inserter(vec_vals),
                [divisor](int x) {return (x % divisor) == 0;});
    for (auto val: vec_vals)
    {
        std::cout << val << "\n";
    }

    std::vector<int> double_vec;
    std::for_each(vec_vals.begin(), vec_vals.end(),
            [&] (int x){double_vec.push_back(x*2);}); // & because we need access to double vec

    std::vector<int> vec1 = {1,2,3,4,5};
    std::vector<int> vec2 = {1,2,3,4,5};
    std::vector<int> vec3(5);

    std::transform(vec1.begin(), vec1.end(),
                    vec2.begin(), vec3.begin(),
                    [](int x, int y){return x + y;});

    // ----- TERNARY -------
    int age = 21;
    bool can_i_vote = (age >= 18) ? true : false;
    std::cout.setf(std::ios::boolalpha);
    std::cout << "Can derek vote : " << can_i_vote << "\n";


    std::function<int(int)> Fib =
        [&Fib](int n) {return n < 2 ? n : 
            Fib(n-1 )+ Fib(n-2);};
    // This creates a callable that takes an int as argument,
    // This callable is a lambda function that has a reference to this
    // callable when needed and does computation shown in body

    std::cout << "Fib 4 : " << Fib(4) << "\n";


    // ------- GENERATE FIBONACCI LIST -----------
    std::vector<int> fibs = gen_fibs(10);
    for(auto val: fibs)
    {
        std::cout << val << "\n";
    }  


}



std::vector<int> generate_random_vector(int amount, int min, int max)
{
    std::vector<int> vals;
    srand(time(NULL)); // make seed
    int i =0, rand_val = 0;
    while(i < amount)
    {
        rand_val = min + std::rand() % ((max + 1) - min); // guarantee it falls in requested range
        vals.push_back(rand_val);
        i++;
    }

    return vals;
}

std::vector<int> gen_fibs(const int count)
{
    std::vector<int> fib_list;
    int i = 0;
    std::function<int(int)> Fib =
        [&Fib](int n) {return n < 2 ? n : 
            Fib(n-1 )+ Fib(n-2);};
    while (i < count)
    {
        fib_list.push_back(Fib(i));
        i++;
    }
    return fib_list;
}