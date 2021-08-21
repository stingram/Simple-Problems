#include <iostream>
#include <string>
#include <unordered_map>



int convert(std::string num)
{
    std::unordered_map<char,int> vals =   {{'I',1},
                                          {'V',5},
                                          {'X',10},
                                          {'L',50},
                                          {'C',100},
                                          {'D',500},
                                          {'M',1000}};
    int res  = 0;
    int i = num.size() - 1;
    while(i>=0)
    {
        if(i!=0 && (vals[num[i]] > vals[num[i-1]]))
        {
            res += (vals[num[i]] - vals[num[i-1]]);
            i -= 2;
        }
        else
        {
            res += vals[num[i]];
            i--;
        }
    }
    return res;
}

int main()
{
    std::string n = "MCMIV";
    std::cout << convert(n) << '\n';
    // 1904
    return 0;
}