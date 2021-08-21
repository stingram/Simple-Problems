def sort_nums(nums):
    low = 0
    high = len(nums)-1
    index = 0
    while index < high:
        if(nums[index] == 1):
            nums[index], nums[low] = nums[low], nums[index]
            low += 1
            index += 1
            
        if(nums[index] == 2):
            index += 1
            
        if(nums[index] == 3):            
            nums[index], nums[high] = nums[high], nums[index]
            high -= 1
            
    return nums


print(sort_nums([3, 3, 2, 1, 3, 2, 1]))
# [1, 1, 2, 2, 3, 3, 3]