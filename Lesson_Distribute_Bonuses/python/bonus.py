

   

def bonus(nums):
    b = [1]*len(nums)
    for i in range(len(nums)):
        # get performance of neighbors
        if i == 0:
            if nums[i] > nums[i+1]:
                b[i] = b[i+1] + 1
        elif i == len(nums) - 1:
            if nums[i] > nums[i-1]:
                b[i] = b[i-1] + 1
        else:
            if nums[i] > nums[i+1] and nums[i] > nums[i-1]:
                b[i] = max(b[i-1]+1,b[i+1]+1)
            elif nums[i] > nums[i+1] and nums[i] <= nums[i-1]:
                b[i] = b[i+1]+1
            elif nums[i] > nums[i-1] and nums[i] <= nums[i+1]:
                b[i] = b[i-1]+1 
                
    for i in range(len(nums)-1,-1,-1):
        # get performance of neighbors
        if i == 0:
            if nums[i] > nums[i+1]:
                b[i] = b[i+1] + 1
        elif i == len(nums) - 1:
            if nums[i] > nums[i-1]:
                b[i] = b[i-1] + 1
        else:
            if nums[i] > nums[i+1] and nums[i] > nums[i-1]:
                b[i] = max(b[i-1]+1,b[i+1]+1)
            elif nums[i] > nums[i+1] and nums[i] <= nums[i-1]:
                b[i] = b[i+1]+1
            elif nums[i] > nums[i-1] and nums[i] <= nums[i+1]:
                b[i] = b[i-1]+1     
    
    return b


def bonus_v2(nums):
    b = [1]*len(nums)
    
    for i in range(1,len(nums)):
        if nums[i-1] < nums[i]:
            b[i] = b[i-1]+1
            
    for i in range(len(nums)-2,-1,-1):
        if nums[i+1] < nums[i]:
            b[i] = max(b[i], b[i+1]+1)
            
    return b


print(bonus([1, 2, 3, 4, 3, 1]))
# [1, 2, 3, 4, 2, 1]

print(bonus([4,3,2,1]))
# [4, 3, 2, 1]

print(bonus_v2([1, 2, 3, 4, 3, 1]))
# [1, 2, 3, 4, 2, 1]

print(bonus_v2([4,3,2,1]))
# [4, 3, 2, 1]