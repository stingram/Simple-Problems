#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <iterator>

class Solution
{
    public:
    bool is_cycle_dfs(std::unordered_map<char,std::vector<std::string>>& symbol,
                                        std::string current_word, std::string start_word, int length,
                                        std::unordered_set<std::string>& visited)
        {
            if(length == 1)
            {
                return start_word[0] == current_word[current_word.size()-1];
            }
            visited.insert(current_word);
            std::vector<std::string> word_list = symbol[current_word[current_word.size()-1]];
            for(std::string neighbor: word_list)
            {
                auto it = visited.find(neighbor);
                if(it == visited.end() && is_cycle_dfs(symbol, neighbor, start_word, length-1, visited))
                {
                    return true;
                }
            }
            visited.erase(current_word);
            return false;
        }
    bool chainedWords(const std::vector<std::string>& words)
    {
        std::unordered_map<char,std::vector<std::string>> symbol;
        std::unordered_set<std::string> visited;
        for (auto word : words)
        {
            auto it = symbol.find(word[0]);
            if(it != symbol.end())
            {
                it->second.push_back(word);
            }
            else
            {
                symbol.insert(std::pair<char,std::vector<std::string>>(word[0], {word}));
            }
        }
        return is_cycle_dfs(symbol, words[0], words[0], words.size(), visited);
    }
};

int main()
{
    std::vector<std::string> words = {"apple", "eager", "ege", "raa"};
    std::cout << Solution().chainedWords(words) << "\n";
    return 0;
}