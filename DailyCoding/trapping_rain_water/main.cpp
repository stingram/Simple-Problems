#include <vector>
#include <algorithm>
#include <iostream>


int trapped_water(const std::vector<int>& inputs)
{
    std::vector<int> L(inputs.size(),0);
    std::vector<int> R(inputs.size(),0);

    // Since L we can start at position 1 instead of 0
    for(int i = 1;i<inputs.size();i++)
    {
        L[i] = std::max(inputs[i-1],L[i-1]);
    }

    int amount = 0;

    // Since R we can start at position size - 2 instead of size -1, also we don't need to check position 0
    for(int i = inputs.size()-2;i>=1;i--)
    {
        R[i] = std::max(inputs[i+1],R[i+1]);
        // since we have left and right we can compute how much we can fill here
        amount += std::max(std::min(R[i],L[i])-inputs[i],0);
    }

    return amount;
}

int main()
{
    std::vector<int> inputs = {0,1,0,2,1,0,1,3,2,1,2,1};
    std::cout << "Water that can be trapped: " << trapped_water(inputs) << ".\n";
    inputs = {1,5,3,5,10,3,2,1,10};
    std::cout << "Water that can be trapped: " << trapped_water(inputs) << ".\n";
}