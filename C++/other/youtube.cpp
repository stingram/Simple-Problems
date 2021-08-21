#include <iostream>
#include <cmath>
#include <climits>
#include <float.h>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <array>
#include <fstream>

double power(double, int);


void print_array(std::array<int, 251> & arr, int size)
{
    for(int i=0;i<size;i++)
    {
        std::cout << arr[i] << "\t";
    }
    std::cout << std::endl;
}

void print_vector(std::vector<int> & vec)
{
    for(int i=0;i<vec.size();i++)
    {
        std::cout << vec[i] << "\t";
    }
    std::cout << std::endl;
}


void save_score(int count)
{

    std::ifstream input ("best_scores.txt");
    if(!input.is_open())
    {
        std::cout << "Unable to read file." << std::endl;
        return;
    }
    int best_score;
    input >> best_score;

    std::ofstream output ("best_score.txt");
    if(!output.is_open())
    {
        std::cout << "Unable to read file." << std::endl;
        return;
    }

    if(count < best_score)
    {
        output << count;
    }
    else
    {
        output << best_score;
    }
    
    input.close();
    output.close();
}

void play_game()
{
    // std::array<int, 251> guesses_a;
    std::vector<int> guesses;
    int random = rand() % 251; // Get something from 0 to 250
    std::cout << "Game is being played\n";
    std::cout << "Guess a number: ";
    int count = 0;
    while(true)
    {
        int guess;
        std::cin >> guess;
        // guesses_a[count++] = guess;
        count++;
        guesses.push_back(guess);
        if(guess == random)
        {
            std::cout << "You win!\n";
            break;
        } else if (guess < random)
        {
            std::cout << "Too low!\n";
        } else
        {
            std::cout << "Too high!\n";
        }
        
    }
    // print_array(guesses_a, count);
    save_score(count);
    print_vector(guesses);
}

int main()
{

    // GAME MENU
    srand(time(NULL)); // To get different numbers each time
    int choice;
    
    do
    {
        std::cout << "0. Quit" << std::endl << "1. Play Game\n";
        std::cin >> choice;

        switch (choice)
        {
            case 0:
                std::cout << "Thanks for nothing\n";
                break;
            case 1:
                std::cout << "Yo let's play.\n";
                break;
        }

    } while (choice != 0);






    int slices;

    /*
    std::cout << "How many slices?";
    std::cin >> slices;

    std::cout << "Hello world!" << std::endl;

    printf("%i\n", slices);

    int base, exponent;
    std::cout << "What is the base? ";
    std::cin >> base;
    std::cout << "What is the exponent? ";
    std::cin >> exponent; 
    double my_power = power(base,exponent);
    std::cout << my_power; 
    */

    short a;
    int b;
    long c;
    long long d;
    // short <= int <= long <= long long
    unsigned short aa;
    unsigned int bb;
    unsigned long cc;
    unsigned long long dd;

    std::cout << sizeof(int) << std::endl;
    std::cout << ULLONG_MAX << std::endl;

    char x = 'B';
    std::cout << (int) x << std::endl;

    char y = 65;
    std::cout << y << std::endl;

    unsigned char z = 129;

    bool found = false;
    std::cout << std::boolalpha << found << std::endl;

    float fa;
    double db = 7.7E4;
    long double ld;

    std::cout << std::fixed << db << std::endl;

    std::cout << DBL_DIG << std::endl;

    const int cx = 5; // symbolic constants, just call it const, creates a read-only variable
    enum { ey = 100}; // another way too make constant

    std::cout << -INFINITY << std::endl;
    std::cout << remainder(10,3.25) << std::endl; // Better modulus
    std::cout << fmax(10, 3.25) << fmin(10, 3.25) << std::endl; //min and max
    std::cout << ceil(3.25) << floor(3.25) << std::endl;
    std::cout << trunc(-1.5) << floor(-1.5) << std::endl; // output is: -1 -2

    std::string greeting = "hello";
    std::cout << greeting + " there" << std::endl;

    std::cout << greeting.length() << std::endl;        // method is a member function == function attached to object

    char name[] = "Hello"; // c string == array of characters "Caleb\0"
    // name = "t"; // not allowed 

    // cin only grabs one word at a time, it needs to know when to stop. It stops with whitespace.
    // Need getline!
    std::string better_greeting;
    std::getline(std::cin, better_greeting);
    better_greeting.insert(3, "   "); // inserts the string at the given index
    better_greeting.erase(3, 3); // starting at position 3, this will remove 3 characters
    better_greeting.erase(3); // remove everything from 3 on
    better_greeting.erase(better_greeting.length()-1); // remove last character
    better_greeting.pop_back(); // removes last character
    better_greeting.replace(0, 4, "Heaven"); // starting at 0, charnge 4 characters, replace those characters with "Heaven"
    better_greeting.replace(better_greeting.find("hell"), 4, "****"); //find returns starting index
    better_greeting.substr(5, 2); // Get 2 characters starting at position 5
    better_greeting.find_first_of("aeiou"); // find first occurence of any aeiou, will return npos, or -1 of unsigned long
    if(better_greeting.find_first_of("!") == -1)
     {
         std::cout << "NOT FOUND!" << std::endl;
     }
    better_greeting.compare("BLAH!"); // returns 0 if the same

    auto xx = 5U;
    auto test = 5.5L;

    int octal = 030; // prefix with 0 for octal
    int hex = 0x30; //prefix with 0x for hex
    std::cout << std::hex << hex << std::oct << octal << std::endl;

    int age;
    std::cin >> age;
    
    enum class Season{Summer, Spring, Fall, Winter}; // c++11
    Season Now = Season::Winter;
    switch(Now){
        case Season::Summer:
            std::cout << "You are: " << age << std::endl;
            break;
        case Season::Spring:
            break;

        case Season::Fall:
            break;
        case Season::Winter:
            break;

        default:
            std::cout << "UwU" << std::endl;
            break;

    }    



    enum season{summer, spring, fall, winter};
    season now = winter;

    switch(now){
        case summer:
            std::cout << "You are: " << age << std::endl;
            break;
        case spring:
            break;

        case fall:
            break;
        case winter:
            break;

        default:
            std::cout << "UwU" << std::endl;
            break;

    }


    // Has to take integral type as input to switch, double doens't work, etc.
    switch(age){
        case 13:
            std::cout << "You are: " << age << std::endl;
            break;
        case 14:

            break;

        default:
            std::cout << "UwU" << std::endl;
            break;

    }

    return 0;








}


double power(double base, int exponent)
{
    double result = 1;
    for(int i = 0; i < exponent; i++)
    {
        result = result*base;
    }

    return result;
}