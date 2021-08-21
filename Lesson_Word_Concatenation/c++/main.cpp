#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <string>


bool can_form(const std::unordered_set<std::string>& words_dict,const std::string& word, std::unordered_map<std::string,bool>& cache)
{
    if(cache.find(word) != cache.end())
    {
        return cache[word];
    }
    for(int i=1;i<word.size();i++)
    {
        // build prefix and suffix
        std::string prefix = word.substr(0,i);
        std::string suffix = word.substr(i,word.size()-i);

        if(cache.find(prefix)!=cache.end())
        {
            if(cache.find(suffix)!=cache.end() || can_form(words_dict,suffix,cache))
            {
                cache[word] = true;
                return true;
            }
        }
    }
    cache[word] = false;
    return false;
}


std::vector<std::string> word_concatenation(const std::vector<std::string>& words)
{
    std::unordered_set<std::string> words_dict(words.begin(),words.end());
    std::unordered_map<std::string,bool> cache;

    std::vector<std::string> res;
    for(std::string word : words)
    {
        if(can_form(words_dict,word,cache))
        {
            res.push_back(word);
        }
    }
    return res;

}


int main()
{
    std::vector<std::string> words = word_concatenation({"cat", "cats", "dog", "catsdog"});
    for(std::string word : words)
    {
        std::cout << word << ", ";
    }
    std::cout << "\n";
    return 0;
}