from typing import List

class Solution:
    def getRange(self, arr: List, target: int):
        first = self.binarySearchIterative(arr, 0, len(arr) - 1, target, True)
        last = self.binarySearchIterative(arr, 0, len(arr) - 1, target, False)
        return [first, last]
    
    def binarySearchIterative(self, arr: list,
                              low: int, high: int, target: int,
                              findFirst: bool):
        while True:
            # check boundary indices we are searching in for the target
            # if we got here we couldn't find it because low started at 0
            # and high started at len(arr) - 1
            if high < low:
                return -1
            # set midpoint between boundary indices
            mid = low + (high - low) // 2
            
            # find first occurance of the target value
            if findFirst:
                
                # if we find that mid index gets us target value, AND EITHER 
                # of the two conditions are true
                # 1. mid is 0, so that's the first index, so we can't go and lower
                # 2. We check value one to the left of mid and if our target value
                #    is greater than the value at index mid -1,
                # THEN, we have found the first index of the occurence of our target
                if(mid == 0 or target > arr[mid - 1]) and arr[mid] == target:
                    return mid
                
                # if our target is larger than value at index mid, then our new low index
                # search index would be mid + 1
                if target > arr[mid]:
                    low = mid + 1
                # otherwise our target is smaller than or equal to value at index mid, so
                # our high index for finding first index of our target value needs to be
                # mid - 1
                else:
                    high = mid - 1
                
            # looking for index of last occurance of target    
            else:
                # likewise we can return mid here if (we find our target) AND
                # (we are at the end of our array OR if the value at index mid + 1 
                # is greater than our target value)
                if (mid == len(arr) -1 or target < arr[mid+1]) and arr[mid] == target:
                    return mid
                # if our target is smaller than value at index mid, then we need 
                # set high index to be mid - 1
                elif target < arr[mid]:
                    high = mid - 1
                # else, our target is greater than value at index mid, we need to
                # set low index boundary to mid + 1
                else:
                    low = mid + 1
                        
arr = [1,3,3,5,6,8,9,9,9,15]
x = 9
print(Solution().getRange(arr, x))