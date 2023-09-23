#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <benchmark/benchmark.h>

void find_combinations(const std::string& s, const std::string& curr, int index, std::vector<std::string>& result)
{
    // base case, we've finished processing the entire string and we can return the current combination
    if(index == s.length())
    {
        result.push_back(curr);
        return;
    }
    // Include the character at index
    find_combinations(s, curr+s[index], index+1, result);
    
    // Don't include the character at index
    find_combinations(s, curr, index+1, result);
}

std::vector<std::string> gen_combinations(const std::string& s)
{
    std::vector<std::string> result;
    result.reserve(std::pow(2,s.length())); // unsure what would happen here since a string can be any length
    // note how we start with
    find_combinations(s,"",0,result);
    return result;
}


int main()
{
    std::string s = "abc";
    std::vector<std::string> res = gen_combinations(s);
    for(const std::string& r: res)
    {
        std::cout << r << ", ";
    }
    std::cout << "\n";
    return 0;
}