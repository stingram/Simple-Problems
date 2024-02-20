# This problem was asked by Netflix.

# Implement a queue using a set of fixed-length arrays.
# The queue should support enqueue, dequeue, and get_size operations.

ARRAY_SIZE = 2

class Queue:
    def __init__(self) -> None:
        self._array_size = ARRAY_SIZE
        self._size = 0 # global size
        self._queue = []
        # where next element will be pulled from
        self._front = -1
        # where next element will be added 
        self._back = 0
    def enqueue(self, num):
        if self._back >= self._array_size or self._size == 0:
            # allocate new array
            self._queue.append([num]*self._array_size)
            self._back = 1
        else:
            self._queue[-1][self._back]= num
            self._back += 1
        self._size += 1

        if self._front == -1:
            self._front = 0

    def dequeue(self):
        if self._size <= 0:
            raise Exception("No elements in queue")
        val = self._queue[0][self._front]
        # check if item the end of an array
        if self._front == self._array_size - 1:
            del self._queue[0]
            self._size -= 1
            if self._size == 0:
                self._front = -1
            else:
                self._front = 0
        else:
            self._size -= 1
            self._front += 1
        return val
    
    def get_size(self):
        return self._size
    
    def print(self):
        print(f"front: {self._front}, back: {self._back}, {self._queue}")

        
q = Queue()
q.enqueue(1)
print(f"Size: {q.get_size()}")
q.enqueue(2)
print(f"Size: {q.get_size()}")
q.enqueue(3)
print(f"Size: {q.get_size()}")

# debug
q.print()

print(f"val: {q.dequeue()}")
print(f"Size: {q.get_size()}")

q.print()

print(f"val: {q.dequeue()}")
print(f"Size: {q.get_size()}")

q.print()

print(f"val: {q.dequeue()}")
print(f"Size: {q.get_size()}")

q.print()