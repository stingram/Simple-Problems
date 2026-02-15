from typing import List
import numpy as np

def reverse_list(l: List[int]) -> List[int]:
    reversed = l[::-1]
    return reversed

def evens_times_two(l: List[int]) -> List[int]:
    return [num*2 for num in l if num % 2 == 0]

# takes a list of strings and returns new list containing the lengths of
# those strings starting with a vowel
def length_of_words_starting_with_vowel(l: List[str]) -> List[int]:
     vowels = 'AEIOUaeiou'
     
     # get vowel strings
     vowel_strings = list(filter(lambda s: s[0] in vowels,l))
     
     # get lengths
     lengths = list(map(lambda s: len(s), vowel_strings))
     
     return lengths
 
# combine dicts, if key in both, use value from second
def combine_dicts(dict1, dict2):
    merged = {**dict1, **dict2} # last dictionary included will be the value for the key
    return merged

# implement a stack
class Stack:
    def __init__(self):
        self.items = []
    
    def add(self, item) -> None:
        self.items.append(item)
    
    def remove(self):
        if not self.is_empty():
            return self.items.pop()
        else:
            raise IndexError("remove from empty stack")
    
    def top(self):
        if not self.is_empty():
            return self.items[-1]
        else:
            raise IndexError("top from empty stack")
    
    def is_empty(self) -> bool:
        return len(self.items) == 0 
    
# do matrix multiplication
def matrix_multiplication(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    assert x.dtype == w.dtype
    assert x.shape[-1] == w.shape[-2]
    return np.matmul(x, w)

# do dot product of two vectors
def dot_product(x: np.ndarray, y: np.ndarray) -> np.float32:
    assert x.dtype == y.dtype
    assert x.ndim == 2 and y.ndim == 2
    assert x.shape == y.shape
    return np.dot(x,y)

# write function that replaces all negative numbers in array with 0 
# and all positive numbers with 1
def neg_to_zero_pos_to_one(x: np.ndarray) -> np.ndarray:
    return np.where(x < 0, 0, 1)

# find indices of 