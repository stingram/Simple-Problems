#include <iostream>
#include <vector>
#include <unordered_map>
#include <numeric>

int multitask(std::vector<int>& tasks, int cooldown)
{
    std::unordered_map<int,int> tdict;
    int current = 0;
    for(int task: tasks)
    {
        if(tdict.find(task) != tdict.end())
        {
            if((current - tdict[task]) <= cooldown)
            {
                current = cooldown + tdict[task] + 1;
            }
        }
        tdict[task] = current;
        current += 1;
    }
    return current;
}

int main()
{
    std::vector<int> tasks = {1,1,2,1};
    std::cout << multitask(tasks, 2) << "\n";

    return 0;
}