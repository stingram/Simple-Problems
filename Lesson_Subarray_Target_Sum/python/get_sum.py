

def find_subarray(nums, target):
    sums_dict={}
    acc = 0
    res = []
    for i,num in enumerate(nums):
        acc += num
        
        # get current sum and add to dictionary
        sums_dict[acc] = i
        
        # check if acc - target exist in dictionary
        test = acc - target
        if test in sums_dict:
            start_ind = sums_dict[test]+1
            end_ind = i+1
            return nums[start_ind:end_ind]
            
    
    return res






print(find_subarray([1, 3, 2, 5, 7, 2], 14))