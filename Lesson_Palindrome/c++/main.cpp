#include <iostream>
#include <unordered_map>
#include <string>



std::string find_palindrome(const std::string& word)
{
    // create counter
    std::unordered_map<char,int> counter;
    for(auto c : word)
    {
        if(counter.find(c)==counter.end())
        {
            counter[c] = 1;
        }
        else
        {
            counter[c]+= 1;
        }
    }

    // odd counter
    int num_odd = 0;
    std::string res = "";

    // go over dictionary while building results
    for(auto it : counter)
    {
        char c = it.first;
        int count = it.second;
        
        // check number of odd characters, if we get more than one we can't have a palindrome
        if(count % 2 != 0)
        {
            num_odd += 1;
        }
        if(num_odd>1)
        {
            return "";
        }

        // build result
        if(count % 2 == 0)
        {
            // put char on ends
            while(count > 0)
            {
                res = c + res + c;
                count -= 2;
            }
        }
        else
        {
            // put char on ends
            while(count > 1)
            {
                res = c + res + c;
                count -= 2;
            }
            // put char in  middle
            int s = int(res.size()/2);
            std::string p1 = res.substr(0,s);
            std::string p2 = res.substr(s,s);
            res = p1 + c + p2;
        }

    }
    return res;
}

int main()
{
    std::cout << find_palindrome("foxfo") << "\n";
    return 0;
}