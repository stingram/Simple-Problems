
class Solution(object):
    def __init__(self):
        self.vals = {'I':1,'V': 5, 'X':10,'L':50,'C':100,'D':500,'M':1000}
    
    
    def romanToInt(self, n):
        
        i = len(n)-1
        res = 0
        while i >= 0:
            # check pair
            if(i!=0 and (self.vals[n[i-1]] < self.vals[n[i]])):
                # subtract
                res += (self.vals[n[i]] - self.vals[n[i-1]])
                i -= 2
            else:
                res += self.vals[n[i]]
                i -= 1
        return res

n = 'MCMIV'
print(Solution().romanToInt(n))
# 1904