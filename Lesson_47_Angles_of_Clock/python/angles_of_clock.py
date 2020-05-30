class Solution(object):
    
    def angles_v2(self, H: int, M: int) -> float:
        angle_H = (360/(12.0*60))*(H*60+M)
        angle_M = 360/60.0*M
        alpha = abs(angle_H-angle_M) 
        return min(alpha, 360-alpha)
    

# Assuming 12H clock
H=11
M=59

print(Solution().angles(H,M))

print(Solution().angles_v2(H,M))