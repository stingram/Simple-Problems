#include <cstdlib>
#include <vector>
#include <iostream>
#include <string>
#include <map>


class Solution{
    private:
    std::string to_string(int* a, int size)
    {
        std::string s = "";
        for(int i =0;i<size;i++)
        {
            s += a[i];
        }
        return s;
    }


    std::string hash_key(const std::string& s1){
        int array[26] = {0};
        for(char c : s1){
            array[int(c) - int('a')] = 1;
        }
        return Solution::to_string(array, sizeof(array)/sizeof(int));

    }
    
    public:
    std::vector<std::vector<std::string>> group_anagrams(const std::vector<std::string>& words){
        std::map<std::string, std::vector<std::string>> groups; 
        for(std::string word : words)
        {
            std::string hash_key = Solution::hash_key(word);
            if(groups.find(hash_key) == groups.end())
            {
                std::vector<std::string> vword = {word};
                groups[hash_key] = vword;
            }
            else{
                groups[hash_key].push_back(word);
            }
        }

        std::vector<std::vector<std::string>> result;
        for(auto it = groups.begin(); it != groups.end(); it++)
        {
            result.push_back(it->second);
        }
        return result;
    }

};


    // def _hash_key_v2(self, s: str):
    //     arr = [0]* 26
    //     for char in s:
    //         arr[ord(char)- ord('a')] = 1
    //     return tuple(arr)
    
    // def group_anagrams_v3(self, words: List[str]) -> List[str]:
    //     groups = collections.defaultdict(list)
    //     for word in words:
    //         hash_key = self._hash_key_v2(word)
    //         groups[hash_key].append(word)
    //     return groups.values()
    
int main(){
    std::vector<std::string> test = {"abc", "bcd", "cba", "cbd", "efg"};
    auto result = Solution().group_anagrams(test);
    for(auto words : result)
    {
        for(auto word : words)
        {
            std::cout << word << " ";
        }
        std::cout << "\n";
    }
    return 0;
}

// print(Solution().group_anagrams(test))