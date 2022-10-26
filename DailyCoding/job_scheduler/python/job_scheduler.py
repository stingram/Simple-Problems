# Implement a job scheduler which takes in a function f 
# and an integer n, and calls f after n milliseconds.

# REFERENCE: 
# https://sathishbabu96n.medium.com/daily-coding-problem-problem-10-da50b93bfc67

# TIME COMPLEXITY - O(log(n))
# SPACE COMPLEXITY - O(n)

import heapq
import threading

from typing import Callable
from time import sleep, time

# BAD WaY - BLOCKING
def job_scheduler(f: Callable[[None],None], n: int):
    # easy
    sleep(n/1000)
    f()

# BETTER WAY
class Scheduler:
    def __init__(self):
        self.functions = []
        # create min heap for functions
        heapq.heapify(self.functions)
        # create a thread that uses polling
        thread = threading.Thread(target=self._poll)
        # start the thread
        thread.start()
        
    # polling to check if it's time to run a function
    def _poll(self):
        while True:
            # get current time
            now = time() * 1000
            if len(self.functions) > 0:
                # get first element, guaranteed to be smallest due value
                due, func, args, kwargs = self.functions[0]
                # if it is past due time to run function, run it
                if now > due:
                    if args is not None and kwargs is not None:
                        func()
                    else:
                        func(*args,**kwargs)
                    # remove function from heap (remove top element)
                    heapq.heappop(self.functions)
                # We are choosing to sleep for 10ms, so we could be off by that much
                sleep(0.01)
                
    def schedule(self, func, n, *args, **kwargs):
        # just push info into heap
        heapq.heappush(self.functions, (n+time()*1000, func, args, kwargs))

def my_func():
    print(f"Derp.")
job_scheduler(my_func, 500)

scheduler = Scheduler()
scheduler.schedule(my_func,500, None, None)