

def partition(nums, L, R):
    i = L - 1
    p = nums[R]
    for j in range(L,R):
        if nums[j] < p:
            i += 1
            nums[i], nums[j] = nums[j], nums[i]
    nums[i+1], nums[R] = nums[R], nums[i+1]
    return i+1 


def partition_list(nums, k, L, R):
    while L<R:
        p = partition(nums,L,R)
        if p == k:
            break
        elif p > k:
            # sort left side
            R=p-1
        else:
            #sort right side
            L=p+1


def partition_v2(nums, k):
    high = len(nums) - 1
    low = 0
    i =0
    while i < high:
        n =nums[i]
        if n > k:
            nums[high], nums[i] = nums[i], nums[high]
            high -= 1
        if n < k:
            nums[low], nums[i] = nums[i], nums[low]
            low += 1
            i += 1
        if n==k:
            i+= 1

nums = [8,9,2,4,1,0]
k=3
partition_list(nums,k,0,len(nums)-1)
print(nums)

nums = [8,9,10,4,1,5,3,7,6,2]
k=3
partition_list(nums,k,0,len(nums)-1)
print(nums)


nums = [8,9,2,4,1,0]
k=3
partition_v2(nums,k)
print(nums)
