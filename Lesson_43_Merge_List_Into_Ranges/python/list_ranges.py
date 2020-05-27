from typing import List

class Solution:
    def list_ranges(self, nums: List[int]) -> List[str]:
        res = []
        new_range = True
        for i in range(len(nums)-1):
            # if starting new range
            if new_range:
                string_range=str(nums[i])
                new_range = False
            # inside a range
            # else
            if nums[i] == nums[i+1] or nums[i] == nums[i+1] - 1:
                # May need to add dash
                if string_range[-1] != '-':
                    string_range += '-'
                else:
                    continue
            else:
                # close range
                if string_range[-1] == '-':
                    string_range += str(nums[i])
                new_range = True
                res.append(string_range)
        
        # Handle end
        last = str(nums[len(nums)-1])
        if new_range:
            res.append(last)
        else:
            res.append(string_range+last)
        
        return res
    
    def list_ranges_v2(self, nums: List[int]) -> List[str]:
        ranges =[]
        if not nums:
            return []
        low = nums[0]
        high = nums[0]
        for n in nums:
            if high + 1 < n:
                ranges.append(str(low) + '-' + str(high))
                low = high = n
            else:
                high = n
        ranges.append(str(low) + '-' + str(high))
        return ranges
    
    

nums = [0,1,2,5,7,8,9,9,10,11,15]
print(Solution().list_ranges(nums))

nums = [0,1,2,5,7,8,9,9,10,11,12]
print(Solution().list_ranges(nums))

nums = [0]
print(Solution().list_ranges(nums))

nums = [0,1]
print(Solution().list_ranges(nums))

nums = [0,2,4,6,8]
print(Solution().list_ranges(nums))