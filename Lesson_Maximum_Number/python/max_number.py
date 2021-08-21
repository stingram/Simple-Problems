from functools import cmp_to_key


def max_number(nums):
    
    sorted_nums = sorted(nums, key=cmp_to_key(
        lambda a,b :
            1 if str(a) + str(b) < str(b) + str(a)
            else -1
    ))
    
    res = "".join(str(n) for n in sorted_nums)
    return res



print(max_number([17, 7, 2, 45, 72]))
# 77245217


def max_number_v2(nums):
    
    sorted_nums = sorted(nums, key=cmp_to_key(
        lambda a,b : 
            return str(a)+ str(b) < str(b) + str(a)))
    
    res = "".join(str(n) for n in sorted_nums)
    return res

print(max_number_v2([17, 7, 2, 45, 72]))
# 77245217