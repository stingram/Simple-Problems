# In the given problem, we have n rooms where meetings are scheduled to take place.
# Our goal is to identify which room will have hosted the most number of meetings
# by the end of all meetings. Each meeting is represented as a pair of integers
# indicating its start and end times. A key point to note is that these meetings'
# start times are unique, meaning no two meetings will start at exactly the same 
# time. 
# The meetings follow certain allocation rules to rooms:
# Each meeting is scheduled in the lowest-numbered room available.
# If all rooms are busy, the meeting will be delayed until a room is free. The meeting will retain its original duration even when delayed.
# Rooms are allocated to meetings based on their original start times, with earlier meetings getting priority.

# Meetings are represented by half-closed intervals [start, end),
# which include the start time but not the end time. The objective
# is to find the room number that has the most meetings by the end.
# And if there's a tie, we are interested in the room with the lowest number.

from typing import List
import heapq

class Solution:
    def find_busiest_room(self, N: int, meetings: List[List[int]]):
        # Create idle heap
        # Use index as key
        idle = list(range(N))
        heapq.heapify(idle)
        
        # Create empty busy heap
        # key will be when the room is free, the second value will be the index
        busy = []
        
        # define counts array
        counts = [0]*N
        
        # loop over meeetings
        for start, end in meetings:
            # first, check if any rooms need to be removed from busy and made idle
            while busy and busy[0][0] <= start:
                _, i = heapq.heappop(busy)
                heapq.heappush(idle, i)
        
            # now we check if any rooms are free
            if idle:
                room_index = heapq.heappop(idle)
                # push the end time for this meeting into busy heap
                heapq.heappush(busy,(end,room_index))
            else:
                # we have no idle rooms so we need to update 
                # a busy room that will end the soonest
                b_end, room_index = heapq.heappop(busy)
                heapq.heappush(busy,(b_end+(end-start),room_index))
                
            counts[room_index] += 1
        most_booked_room = 0
        for i, count in enumerate(counts):
            if counts[most_booked_room] < count:
                most_booked_room = i
        return most_booked_room
    
    
    
meetings = [[0, 10], [0, 5], [10, 15]]
N = 2
print(f"Busiest room is {Solution().find_busiest_room(N,meetings)}")
