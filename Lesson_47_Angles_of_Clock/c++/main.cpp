#include <iostream>
#include <string>
#include <cstdlib>
#include <numeric>
#include <algorithm>

class Solution
{
    public:
    float angles(int H, int M)
    {
        float angle_H = (360/(12.0*60))*(H*60.0+M);
        float angle_M = 360/60.0*M;
        float angle = std::abs(angle_H-angle_M);
        return std::min(angle, 360-angle);
    }
};

int main()
{
    int H = 11;
    int M = 59;
    std::cout << Solution().angles(H,M) << std::endl;
    return 0;
}

// class Solution(object):
    
//     def angles_v2(self, H: int, M: int) -> float:
//         angle_H = (360/(12.0*60))*(H*60+M)
//         angle_M = 360/60.0*M
//         alpha = abs(angle_H-angle_M) 
//         return min(alpha, 360-alpha)
    

// # Assuming 12H clock
// H=11
// M=59

// print(Solution().angles(H,M))

// print(Solution().angles_v2(H,M))