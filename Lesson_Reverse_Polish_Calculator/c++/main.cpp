#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <stack>
#include <limits>







float operation(char op, float a, float b)
{
    if(op == '+')
        return a+b;
    if(op=='-')
        return a-b;
    if(op=='*')
        return a*b;
    if(op=='/')
        return a/b;
    return INT_MIN;
}

int is_operand(char c)
{
    if(c >= '0' && c <='9')
        return 1;
    return -1;
}

int is_operator(char c)
{
    if(c == '+' || c== '-' || c == '*' || c == '/')
    {
        return 1;
    }
    return -1;
}


float make_float(char c)
{
    int value;
    
    //cast as int
    value = c;

    // need to substract value of '0' to get integer equivalent
    return float(c-'0');
}

float calc(std::string s1)
{
    std::string::iterator it;
    std::stack<float> nums;
    float a,b;
    for(it=s1.begin();it<s1.end();it++)
    {
        if(is_operator(*it) == 1)
        {
            b = nums.top();
            nums.pop();
            a = nums.top();
            nums.pop();
            nums.push(operation(*it,a,b));
        }
        else if(is_operand(*it) == 1){
            nums.push(make_float(*it));
        }
    }
    return nums.top();


}


int main()
{
    std::string post = "53+62/*35*+"; // 39
    std::cout << "The result is: "<<calc(post); 
    return 0;
}