class MaxStack(object):
    def __init__(self):
        
        # Have two lists to keep max value synched with the stack
        # When there is a push, both lists get a value
        # When there is a pop, both lists get popped
        self.stack = []
        self.maxes = []
        
    def push(self, val):
        self.stack.append(val)
        # Check thast self.maxes has something in it and then check it's last value
        if self.maxes and self.maxes[-1] > val:
            self.maxes.append(self.maxes[-1])
        else:
            self.maxes.append(val)
        
    def pop(self):
        # Call pop on the lists to keep them synchronized
        self.stack.pop()
        self.maxes.pop()
        
    def max(self):
        # Return last element of maxes
        return self.maxes[-1]
        
        
s = MaxStack()
s.push(1)
s.push(2)
s.push(3)
s.push(2)
print('max', s.max())
print(s.pop())
print('max', s.max())
print(s.pop())
print('max', s.max())
print(s.pop())
print('max', s.max())
print(s.pop())