import heapq
import random

# TIME - O(n*lg(n))
# SPACE - O(n*k*lg(n)) since it take O(n) time to construct
def find_kth_largest(nums, k):
    return sorted(nums)[len(nums)-k]

# TIME - O(k*lg(n)) - construction takes linear time based on how many elements need to be put in
# SPACE - O(n)
def find_kth_largest2(nums, k):
    return heapq.nlargest(k,sum)[-1]

# TIME  - O(n)
# SPACE - O(lg(n))
def find_kth_largest3(nums, k):
    def select(list, l, r, index):
        if l == r:
            return list[l]
        # select pivot
        pivot_index = random.randint(l,r)
        
        # move pivot to beginning of the of list
        list[l], list[pivot_index] = list[pivot_index], list[l]
        
        # partition block
        i = l
        # start i on left, start j 1 position in front of it
        for j in range(l+1, r+1):
            # compare j to left
            if list[j] < list[l]:
                # if j is less than left, then increment i
                i += 1
                # now we swap i and j
                list[i], list[j] = list[j], list[i]
        # move pivot to create location
        list[i], list[l] = list[l], list[i]
        # recursively partition one side
        if index == i:
            return list[i]
        elif index < i:
            return select(list, l, i-1, index)
        else:
            return select(list, i+1, r, index)
    return select(nums, 0, len(nums) -1, len(nums) -k)
        
print(find_kth_largest3([3,5,2,4,6,8], 3))
# 5                
            