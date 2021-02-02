#include <iostream>
#include <unordered_set>
#include <string>
#include <sstream>



char first_recurring_character(std::string in_str)
{
    std::unordered_set<char> seen;
    for (const char&c : in_str)
    {
        if(seen.find(c) != seen.end())
        {
            return c;
        }
        seen.insert(c);
    }
    return '0';
}





int main()
{
    std::cout << first_recurring_character("qwertty") << "\n";
    std::cout << first_recurring_character("qwerty") << "\n";
    return 0;
}