
import sys


def jump_to_end(nums):
    C=[]
    N=[]
    C.append((nums[0],0,0))
    min_hops = float('inf')
    n = len(nums)
    while C:
        while C:
            curr = C.pop()
            print(f"curr[0] = {curr[0]}")
            for i in range(curr[0]):
                print(i)
                dist = i + curr[1]
                if dist == n-1:
                    if dist < min_hops:
                        min_hops = dist
                elif dist < n:
                    N.append((nums[dist], dist, curr[2]+1))
                elif dist > n:
                    break
        C = N
        N = []

    return min_hops

    
def jump_to_end_v2(nums):
    hops = [float('inf')] * len(nums)
    hops[0] = 0
    for i, n in enumerate(nums):
        for j in range(1,n+1):
            if i+j < len(hops):
                hops[i+j] = min(hops[i+j], hops[i]+ 1)
            else:
                break
    return hops[-1]


print(jump_to_end_v2([3, 2, 5, 1, 1, 9, 3, 4]))
# 2