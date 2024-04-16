# Given a list of meetings, represented as tuples
# with a start and an end time, determine the
# minimum number of rooms required to schedule
# all the meetings. Start/End times range from 
# 0 to 10000 inclusive

from typing import List
from collections import defaultdict

def num_rooms_needed(meetings: List[List[int]]) -> int:
    curr = 0
    res = 0

    events = defaultdict(int)

    # save deltas into events dictionary
    for (start,end) in meetings:
        events[start] += 1
        events[end] -= 1

    # sorted deltas
    sorted_times = sorted(events.keys())
    
    # loop over times
    for time in sorted_times:
        curr += events[time] # update current room count
        if curr > res:
            res = curr
          
    return res
    
    
meetings = [[9, 10], [1, 3], [5, 7], [4, 6], [7, 9]]
print(f"{num_rooms_needed(meetings)}")    
    