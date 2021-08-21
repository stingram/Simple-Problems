#include <vector>
#include <iostream>
#include <string>
#include <unordered_map>

int h_index(const std::vector<int>& papers)
{
    std::vector<int> freqs(papers.size()+1,0);
    for (int p : papers)
    {
        freqs[p] += 1;
    }


    int i = papers.size();
    int total = 0;

    while(i>= 0)
    {
        total += freqs[i];
        if(total>=i)
            return i;
        i -= 1;
    }

    return 0;


}

int main(){
    std::cout << h_index({5, 3, 3, 1, 0}) << "\n";
    // 3

    std::cout << h_index({5, 3, 3, 1, 4, 4, 4}) << "\n";
    // 4

    std::cout << h_index({0, 3, 3, 1, 4, 4, 4}) << "\n";
    // 3
    return 0;
}   