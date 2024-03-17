# Given an arithmetic expression in Reverse Polish Notation, write a program to evaluate it.

# The expression is given as a list of numbers and operands.
# For example: [5, 3, '+'] should return 5 + 3 = 8.

# For example, [15, 7, 1, 1, '+', '-', '/', 3, '*', 2, 1, 1, '+', '+', '-']
# should return 5, since it is equivalent to ((15 / (7 - (1 + 1))) * 3) - (2 + (1 + 1)) = 5.

# [15, 7, 1, 1, '+', '-', '/', 3, '*', 4]

# two stacks
# one for numbers the other for operands

# when we get an operator we push onto stack
# when we get a number x we pop operator and previous result
# if we have one. If we have no previous result we have to pop next
# element which has to be a number. We apply operator and push onto 
# result stack
from typing import List, Union

def add(a,b):
    return a+b
def subtract(a,b):
    return a-b
def multiply(a,b):
    return a*b
def divide(a,b):
    return float(a)/float(b)

operator_mapper = {'+': add,'-':subtract,'/':divide,'*':multiply}

def compute_val(a,b,operator) -> float:
    return operator_mapper[operator](a,b)

def reverse_polish_notation(expression: List[Union[str,int]])-> float:
    val_stack = []
    operators = set(['+','-','/','*'])
    for item in expression:
        if item in operators:
            b = val_stack.pop()
            a = val_stack.pop()
            val_stack.append(compute_val(a,b,item))
        else:
            val_stack.append(item)
    return val_stack[-1]

test = [5,3,'+']
print(f"Result is: {reverse_polish_notation(test)}")

test = [15, 7, 1, 1, '+', '-', '/', 3, '*', 2, 1, 1, '+', '+', '-']
#  ((15 / (7 - (1 + 1))) * 3) - (2 + (1 + 1)) = 5
print(f"Result is: {reverse_polish_notation(test)}")