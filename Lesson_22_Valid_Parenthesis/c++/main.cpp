#include <cstdlib>
#include <vector>
#include <stack>
#include <iostream>
#include <string>






class Solution
{
    std::stack<char> my_stack;
    public:
        bool is_valid(const std::string input)
        {
            for(auto c : input)
            {
                if(c == '(')
                {
                    my_stack.push(c);
                }
                else if(c == ')')
                {
                    if(my_stack.size() == 0)
                        return false;
                    if(my_stack.top() != '(')
                        return false;
                    else
                        my_stack.pop();
                }

                if(c == '[')
                {
                    my_stack.push(c);
                }
                else if(c == ']')
                {
                    if(my_stack.size() == 0)
                        return false;
                    if(my_stack.top() != '[')
                        return false;
                    else
                        my_stack.pop();
                }

                if(c == '{')
                {
                    my_stack.push(c);
                }
                else if(c == '}')
                {
                    if(my_stack.size() == 0)
                        return false;
                    if(my_stack.top() != '{')
                        return false;
                    else
                        my_stack.pop();
                }
            }
            if(my_stack.size() > 0)
                return false;
            return true;
        }
};


int main()
{
    std::string test= "{{[](()))}}";
    bool result = Solution().is_valid(test);
    std::cout << "Result: " << std::boolalpha << result << "\n"; 
    return 0;
}