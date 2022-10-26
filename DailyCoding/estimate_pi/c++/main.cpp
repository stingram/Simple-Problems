// # The area of a circle is defined as πr^2. Estimate π to 3 decimal places using a Monte Carlo method.

// # Hint: The basic equation of a circle is x2 + y2 = r2.


#include <iostream>
#include <random>
#include <cmath>
  
  
// REFERENCE
// https://stackoverflow.com/a/20136256    

double estimate_pi(int N = 10000)
{
    std::random_device rand_dev; 
    std::mt19937 generator(rand_dev());
    std::uniform_real_distribution<double> distribution(-1.0,1.0);
    
    double N_in_circle = 0;
    for(int i=0;i<N;i++)
    {
        // generate x and y values
        double x = distribution(generator);
        double y = distribution(generator);

        if(std::pow(x,2) + std::pow(y,2) < 1)
        {
            N_in_circle++;
        }
    }
    return 4.0*(N_in_circle/N);
}


int main()
{
    int N = 1000000;
    std::cout << "Estimate of pi: " << estimate_pi() << ".\n";
    return 0;
}