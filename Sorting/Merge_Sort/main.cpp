#include <cstdlib>
#include <string>
#include <algorithm>
#include <iterator>
#include <numeric>

bool is_e(char c){
    return static_cast<unsigned char>(c) == 'e';
}

int main()
{
    std::string s1 = "Test";
    bool test = std::all_of(s1.begin(), s1.end(), is_e);

    

    return 0;
}