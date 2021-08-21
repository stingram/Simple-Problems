

def add(a, b):
    return a+b
    
def subtract(a,b):
    return a-b

def multiply(a,b):
    return a*b

def divide(a,b):
    return a/b

def calc(sequence):
    acc = 0
    nums = []
    valid_ops = {'+': add,
                 '-': subtract,
                 '*': multiply,
                 '/': divide }
    for s in sequence:
        if s in valid_ops:
            b = int(nums.pop())
            a = int(nums.pop())
            nums.append(valid_ops[s](a,b))
        else:
            nums.append(s)  
    return nums.pop()

s = [1, 2, 3, '+', 2, '*', '-']
print(calc(s))