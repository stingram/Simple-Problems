from typing import List
class Solution:
    def three_sum_v1(self, nums:List[int]) -> List[List[int]]:
        res = set()
        nums.sort()
        for i in range(len(nums) - 2):
            for j in range(i+1, len(nums) -1):
                for k in range(j+1, len(nums)):
                    if nums[i] + nums[j] + nums[k] == 0:
                        res.add([nums[i], nums[j], nums[k]])
        return list(res)
    
    def three_sum_v2(self, nums:List[int]) -> List[List[int]]:
        res = []
        # REQUIRES SORTED
        nums.sort()
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i-1]:
                continue

            # two sum
            j = i + 1
            k = len(nums) -1
            while j < k:
                sum = nums[i] + nums[j] + nums[k]
                if sum == 0:
                    res.append([nums[i], nums[j], nums[k]])
                
                    # to skip over duplicates
                    while j < k and nums[j] == nums[j+1]:
                        j += 1
                    while j < k and nums[k] == nums[k-1]:
                        k -= 1
                    # now advance pointers
                    j += 1
                    k -= 1
                
                # didn't get sum we needed
                elif sum < 0:
                    j += 1
                elif sum > 0:
                    k -= 1
            # end two sum   
        
                
            