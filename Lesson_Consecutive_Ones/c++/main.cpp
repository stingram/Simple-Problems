#include <iostream>







int ones(int num)
{
    int count = 0;
    while(num)
    {
        num = num & (num << 1);
        count++;
    }
    return count;

}

int main()
{
    std::cout << ones(242) << "\n";
    // 4
    return 0;
}