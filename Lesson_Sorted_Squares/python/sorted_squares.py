def sort_squares(nums):
    
    
    # get index of first non-negative value
    for i, num in enumerate(nums):
        if num > -1:
            ind1 = i-1
            break
        
    # merge sort
    i = ind1
    j = ind1+1
    
    new_arr = [0]*len(nums)
    
    k = 0
    while(i>=0 and j<len(nums)):
        p = nums[j]*nums[j]
        n = nums[i]*nums[i]
        if p > n:
            new_arr[k] = n
            i-=1
        else: 
            new_arr[k] = p
            j+=1
        k+=1
        
    # if we still have i left
    while(i>=0):
        new_arr[k] = nums[i]*nums[i]
        i-=1
        k+=1
        
    # if we still have j left
    while(j<len(nums)):
        new_arr[k] = nums[j]*nums[j]
        j+=1
        k+=1
        
    return new_arr


# def sort_squares_in_place(nums):
#     # set up negative indices
#     neg_i = -1
#     i = 0
    
#     result = []
#     for n in nums:
#         # get to first non-negative value
#         if n >=0:
#             # set negative index
#             if neg_i == -1:
#                 neg_i = i
                
            
                
                
#         # update i
#         i += 1


nums = [-5,-3,-2,-1,0,1,2,3]

print(nums)
print(sort_squares(nums))
