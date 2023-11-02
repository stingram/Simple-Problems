#include <vector>
#include <string>
#include <iostream>



void find_combs_util(const std::string& s, std::vector<std::string>& result)
{
    if(s.empty())
    {
        result.push_back("");
        return;
    }
    else
    {
        std::vector<std::string> combs_without_first;
        find_combs_util(s.substr(1),combs_without_first);

        for(auto const & t: combs_without_first)
        {
            result.push_back(t);
            result.push_back(s[0]+t);
        }
        return;
    }
}

std::vector<std::string> finds_comb(const std::string& s)
{
    std::vector<std::string> result;
    find_combs_util(s, result);
    return result;
}

int main()
{
    std::string s = "abc";
    std::vector<std::string> r = finds_comb(s);
    for(auto const &comb : r)
    {
        std::cout << "\'" << comb << "\'" << ",";
    }
    std::cout << "\n";
    std::cout << "len: " << r.size() << "\n";
    return 0;
}