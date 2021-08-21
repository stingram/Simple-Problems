



def longest_run(num):
    count = 0
    max_count = 0
    q = num
    while 1:
        q, mod = divmod(q,2)
        if mod == 1:
            count+=1
        if mod == 0:
            count=0
        if count > max_count:
            max_count = count
        if q == 0:
            break
        
        
        
    return max_count


def v2(num):
    count = 0
    while num:
        num = num & (num << 1)
        count+=1
    return count



print(longest_run(242))
# 4

print(v2(242))
# 4