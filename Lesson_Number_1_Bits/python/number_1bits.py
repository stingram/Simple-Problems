
 # Time O(lg(n))
 #Space O(1)


def number_1bits(num: int):
    if num == 0:
        return 0

    q = num
    count = 0
    while(1):
        # divide number by 2, get quotient and remainder
        q, r = divmod(q, 2)

        if r == 1:
            count += 1

        if q == 0:
            return count


def number_1bits_v2(num: int):
    count = 0
    while(num>0):
        if (num & 1) == 1:
            count += 1
        num = num >> 1
    return count

print(number_1bits(23))

print(number_1bits_v2(23))


