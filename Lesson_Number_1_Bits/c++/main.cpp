#include <numeric>
#include <iostream>




int num_1bits(int num, const int method=0)
{
    if(method == 0){
        int count = 0;

        while(num>0){
            if((num & 1) == 1)
                count++;
            num = num >> 1;
        }
        return count;
    }
    else
    {
        int count = 0;
        int q = num;
        int r = 0;
        while(q>0)
        {
            r = q % 2;
            if(r ==1)
                count++; 
            q = int(q/2);
        }
        return count;
    }
}

int main()
{
    std::cout << num_1bits(23) << "\n";
    std::cout << num_1bits(23,1) << "\n";
    return 0;
}