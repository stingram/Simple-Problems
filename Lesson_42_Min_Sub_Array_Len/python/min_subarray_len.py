class Solution:
    def min_subarray_len(self, arr, s):
        min_count  = len(arr) + 1
        for i in range(len(arr)):
            L, R = i,i
            done = False
            while not done:
                subsum = sum(arr[L:R])
                if subsum >= s and (R - L) < min_count:
                    min_count = R - L
                new_L = max(0,L-1)
                new_R = min(len(arr),R+1)
                if new_L == L and new_R == R:
                    done = True
                L = new_L
                R = new_R
        if min_count == len(arr) + 1:
            min_count = 0
        return min_count
    
    
    def min_subarray_len_v2(self, nums, s):
        res = float('inf')
        sum = 0
        left = 0
        right = 0
        while right < len(nums):
            sum += nums[right]
            while sum >= s:
                # updating the results
                res = min(res, right - left + 1)
                # increment left pointer
                sum -= nums[left]
                left += 1
            # increment right pointer since sum is smaller than s
            right += 1
        if res == float('inf'):
            return 0
        return res
    
    
arr = [2,3,1,2,4,3]
print(Solution().min_subarray_len(arr, 7))