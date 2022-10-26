#include <iostream>
#include <vector>
#include <algorithm>
#include <sstream>


class Solution{
    private:
    int _N = 0;
    std::vector<std::string> strobo_helper(std::vector<int> nums, int N)
    {
        if(N==0)
        {
            return {""};
        }
        if(N==1)
        {
            return {"1", "8", "0"};
        }
        std::vector<std::string> res = strobo_helper(nums, N-2);
        std::vector<std::string> res_augmented;
        for(std::string r: res)
        {
            res_augmented.push_back("1"+r+"1");
            //res_augmented.push_back(std::string("8")+r+std::string("8"));
            //res_augmented.push_back(std::string("6")+r+std::string("9"));
            //res_augmented.push_back(std::string("9")+r+std::string("6"));
            // We can only do this at certain times. such when N > 2
            if(N != _N){
                res_augmented.push_back("0"+r+"0");
            }
        }
        return res_augmented;
    }

    static int f(std::string s)
    {
        return std::stoi(s);
    }

    public:
    std::vector<int> gen_strobogrammatic(int N)
    {
        _N = N;
        std::vector<std::string> res = strobo_helper({}, N);
        std::vector<int> output;
        output.resize(res.size());
        std::transform(res.begin(), res.end(), output.begin(), f);
        return output;
    }
};

void print_single_element(int a) {
    std::cout << a << "\n";
}

int main()
{
    int N = 4;
    std::vector<int> nums = Solution().gen_strobogrammatic(N);
    std::for_each(nums.begin(), nums.end(), &print_single_element);

    return 0;
}