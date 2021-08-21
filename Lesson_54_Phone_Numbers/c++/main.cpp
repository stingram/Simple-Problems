#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

std::unordered_map<int, std::vector<char>> letters_map {{1,{}},
                                                         {2,{'a','b','c'}},
                                                         {3,{'d','e','f'}},
                                                         {4,{'g','h','i'}},
                                                         {5,{'a','b','c'}},
                                                         {6,{'a','b','c'}},
                                                         {7,{'a','b','c'}},
                                                         {8,{'a','b','c'}},
                                                         {9,{'a','b','c'}},
                                                         {0,{}}};

std::vector<std::string> valid_words = {"dog", "fish", "cat", "dog"};

std::vector<std::string> make_words_helper(std::vector<int> digits, std::vector<char> letters)
{
    if(digits.size() < 1)
    {
        std::string word = "";
        for(char c: letters)
        {
            word += std::string(1,c);
        }
        if(std::find(valid_words.begin(), valid_words.end(), word) != valid_words.end())
        {
            return {word};
        }
        return {};
    }

    std::vector<std::string> results;
    std::vector<char> chars = letters_map[digits[0]];

    for (auto c : chars)
    {
        std::vector<int>   sub_digits(&digits[1],&digits[digits.size()-1]);
        letters.push_back(c);
        std::vector<std::string> res = make_words_helper(sub_digits, letters);
        results.insert(results.end(), res.begin(), res.end());
    }
    return results;
}

std::vector<std::string> make_words(std::string phone)
{
    std::vector<int> digits;
    for(auto p: phone)
    {
        digits.push_back(int(p));
    }
    return make_words_helper(digits, {});
}

int main()
{
    std::vector<std::string> out = make_words("364");
    for(std::string word : out)
    {
        std::cout << word << "\n";
    }
    return 0;
}