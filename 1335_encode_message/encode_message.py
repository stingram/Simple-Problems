# Given the mapping a = 1, b = 2, ... z = 26,
# and an encoded message, count the number of
# ways it can be decoded.

# For example, the message '111' would give 3,
# since it could be decoded as 'aaa', 'ka', and 'ak'.

# You can assume that the messages are decodable.
# For example, '001' is not allowed.

def _helper(memo,curr):
    # base case
    if curr == "":
        return 1

    # if we've already solved
    if curr in memo:
        return memo[curr]

    # get initial answer from subproblem
    ans = _helper(memo,curr[1:])
    
    # if we can select two digits we need to
    # add to our answer
    if len(curr) >= 2 and int(curr[:2]) >= 10 and int(curr[:2]) <= 26:
        ans += _helper(memo,curr[2:])

    # save answer
    memo[curr] = ans
    return ans

def num_ways_to_encode(message: str) -> int:
    memo = {}
    curr = message
    return _helper(memo,curr)

# Time: O(n)
# Space: O(n)

test = '111'
print(f'Num ways for {test} is {num_ways_to_encode(test)}')
test = '1111'
print(f'Num ways for {test} is {num_ways_to_encode(test)}')
test = '111111'
print(f'Num ways for {test} is {num_ways_to_encode(test)}')
test = '11111111111'
print(f'Num ways for {test} is {num_ways_to_encode(test)}')
