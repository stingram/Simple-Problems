class Queue(object):
    def __init__(self):
        self.s1 = []
        self.s2 = []
        
        
    def enqueue(self, val):
        # always enqueue onto stack 1
        self.s1.append(val)
        
    def dequeue(self):
        
        # try to read from stack 2 first, these will be in correct order
        if self.s2:
            return self.s2.pop()
        
        # if stack 2 was empty, then we remove any elements from stack 1
        # and put into stack 2, then those elements will be in correct
        # order
        if self.s1:
            while self.s1:
                self.s2.append(self.s1.pop())
            # once they've all been moved, pop top element
            return self.s2.pop()
        return None
    
q = Queue()
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
q.enqueue(4)

print(q.dequeue())
print(q.dequeue())
print(q.dequeue())
print(q.dequeue())