#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>



std::vector<int> find_anagrams(const std::string s1, const std::string s2)
{
    std::vector<int> res;
    
    // build character map
    std::unordered_map<char,int> char_map;
    for(auto c : s2)
    {
        if(char_map.find(c)!= char_map.end())
        {
            char_map[c]+= 1;
        }
        else
        {
            char_map[c] = 1;
        }
    }

    // proceed over s1 to see if an anagram of s2
    int length = s1.size();
    for(int i=0;i<length;i++)
    {
        // get char
        char c = s1[i];

        // check if we exceeded length of s2
        if(i>=s2.size())
        {
            // get old char
            char old_c = s1[i-s2.size()];

            // add back to char map
            if(char_map.find(old_c)!=char_map.end())
            {
                char_map[old_c]+=1;
            }
            else
            {
                char_map[old_c] = 1;
            }
            if(char_map[old_c] == 0)
            {
                // remove from char_map
                char_map.erase(old_c);
            }
        }

        // subtract from char_map
        if(char_map.find(c)!=char_map.end())
        {
            char_map[c] -= 1;
        }
        else
        {
            char_map[c] = -1;
        }
        // remove from char_map 
        if(char_map[c]==0)
        {
            char_map.erase(c);
        }

        // if char_map is empty we have an anagram, but the index is at the last character
        // we need to get the index of the first character
        if(char_map.size()==0 && i>=s2.size()-1)
        {
            // want the beginning of the anagram so we to subtract s2.size() and add 1 from i
            res.push_back(i-s2.size()+1);
        }
    }
    return res;
}


int main()
{
    std::vector<int> inds = find_anagrams("acdbacdacb", "abc");
    for(int ind : inds)
    {
        std::cout << ind << ", ";
    }
    std::cout << "\n";
    // [3, 7]
    return 0;
}