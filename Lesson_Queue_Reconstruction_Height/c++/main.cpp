#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>




typedef std::vector<std::pair<int,int>> vec;


bool compare_people(const std::pair<int,int>& a, const std::pair<int,int>& b)
{
    // Compare height
    if(a.first<b.first)
        return false;
    else if (a.first>b.first)
        return true;

    // Compare k value
    if(a.second<b.second)
        return true;
    else if (a.second>b.second)
        return false;
}

vec reconstruct_queue(std::vector<std::pair<int,int>>& people)
{
    std::stable_sort(people.begin(),people.end(),&compare_people);
    vec result;
    auto pos = people.begin();
    int offset;
    for(std::pair<int,int> person: people)
    {
        offset = person.second;
        if (offset >= result.size())
            result.push_back(person);
        else
            result.insert(result.begin() + offset, person);
    }


    return result;
}


void print_vec(const vec& people)
{
    std::cout << "[";
    for(auto person: people)
    {
        std::cout << "[";
        std::cout << person.first << ", " << person.second;
        std::cout << "],";
    }
    std::cout << "]";
}


int main()
{
// first number in each pair is a person's height
// second number is the number of people the person sees in front of them (taller people do not see shorter people)
vec people = {{7, 0}, {4, 4}, {7, 1}, {5, 0}, {6, 1}, {5, 2}};
vec result = reconstruct_queue(people);


print_vec(result);
// [[5,0], [7, 0], [5, 2], [6, 1], [4, 4], [7, 1]]
return 0;
}