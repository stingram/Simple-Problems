# Given an array of integers, determine whether it contains a Pythagorean triplet.
# Recall that a Pythogorean triplet (a, b, c) is defined by the equation a2+ b2= c2.

# Easier to start with the end since it's the biggest and then setting left
# at beginning and right one index to the right of current end
def find_pythagorean_triplet(nums):
    squared_nums = [num**2 for num in nums]
    squared_nums.sort()
    N = len(nums)
    for i in range(N-1,1,-1):
        left = 0
        right = i - 1
        while left < right:
            a = squared_nums[left]
            b = squared_nums[right]
            c = squared_nums[i]
            if a+b == c:
                return True
            elif a+b<c:
                left += 1
            else:
                right -= 1
    return False

nums=[3,5,6,7,8,9,4]
print(f"{find_pythagorean_triplet(nums)}")