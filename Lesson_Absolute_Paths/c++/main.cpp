#include <string>
#include <iostream>
#include <vector>
#include <sstream>
#include <stack>

std::vector<std::string> split(const std::string& s, char delimiter)
{
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}


std::string clean_path(std::string& raw_path)
{
    std::vector<std::string> split_up = split(raw_path, '/');
    std::vector<std::string> my_stack;
    for(std::string dir: split_up)
    {
        if(dir == ".")
        {
            continue;
        }
        else if(dir == "..")
        {
            my_stack.pop_back();
        }
        else
        {
            my_stack.push_back(dir);
        }
    }

    // build with "/"
    std::string res;
    for(std::string dir: my_stack)
    {
        res.append(dir);
        res.append("/");
    }

    return res;
}


int main()
{
    std::string path = "/users/tech/docs/.././desk/../";
    std::cout << clean_path(path) << "\n";
    // /users/tech/
    return 0;
}