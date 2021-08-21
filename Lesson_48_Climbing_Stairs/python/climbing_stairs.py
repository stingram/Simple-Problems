class Solution(object):
    def climb_stairs(self, n):
        
        if n == 0 or n == 1:
            return 1
        first = 1
        second = 1
        for i in range(2,n+1):
            third = first + second
            first = second
            second = third
        return third
    
    
n = 8
print(Solution().climb_stairs(n))
