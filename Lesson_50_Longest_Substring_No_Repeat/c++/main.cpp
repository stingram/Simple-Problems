#include <string>
#include <unordered_map>
#include <iostream>
#include <numeric>
#include <algorithm>

class Solution
{
    public:
    int longest_substring(std::string s)
    {
        std::unordered_map<std::string, int> letters;
        int tail = -1;
        int result = 0;
        int head = 0;
        while(head < s.size())
        {
            auto it = letters.find(std::string(1,s[head]));
            if(it != letters.end())
            {
                tail = std::max(tail, letters[std::string(1,s[head])]);
                letters[it->first] = head;
            }
            else{
                letters.insert({std::string(1,s[head]),head});
            }
            result = std::max(result, head - tail);
            head += 1;
        }
        return result;
    }
};

int main()
{
    std::string s = "pwwkewxyzawlmnopq";
    std::cout << Solution().longest_substring(s) << "\n";
    return 0;
}
