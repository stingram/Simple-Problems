#include <ctime>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <iostream>
#include <string>

std::vector<int> Range(int start, int max, int step);



int main()
{
    srand(time(NULL)); // create seed
    int secretNum = std::rand() % 11; // We will get 0 to 10 with this
    int guess = 0;

    do {
        std::cout << "Guess the number : ";
        std::cin >> guess;
        if (guess > secretNum) std::cout << "Too big\n";
        if (guess < secretNum) std::cout << "Too small\n";
    } while(secretNum != guess);

    std::cout << "You guessed it!\n";



    return 0;
}

std::vector<int> Range(int start, int max, int step)
{
    
    // Every while statement needs an index 
    // to start with
    int i = start;
    
    // Will hold returning vector
    std::vector<int> range;
    
    // Make sure we don't go past max value
    while(i <= max){
        
        // Add value to the vector
        range.push_back(i);
        
        // Increment the required amount
        i += step;
    }
    
    return range;
}    
