#include <unordered_map>
#include <vector>
#include <iostream>
#include <string>






bool character_mapping(const std::string& s1, const std::string& s2)
{
    std::unordered_map<char,char> chars;

    for(int i=0;i<s1.size();i++)
    {
        if(chars.find(s1[i]) == chars.end())
        {
            chars[s1[i]] = s2[i];
        }
        else
        {
            if(chars[s1[i]] != s2[i])
            {
                return false;
            }
        }
    }
    return true;
}



int main()
{
    std::string s1 = "abc";
    std::string s2 = "def";

    std::cout << std::boolalpha << character_mapping(s1,s2) << "\n";

    return 0;
}