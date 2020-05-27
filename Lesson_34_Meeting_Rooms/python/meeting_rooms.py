from typing import List

class Solution(object):
    def meeting_rooms(self, constraints: List[List[int]])-> int:
        
        times_dict = {}
        
        # collect times of overlap
        for c_list in constraints:
            for time in range(c_list[0],c_list[1]+1):
                if time not in times_dict:
                    times_dict[time] = 1
                else:
                    times_dict[time] +=1
                    
        # find maximum value in dictionary
        max_rooms = 0
        for k,v in times_dict.items():
            
            if v > max_rooms:
                max_rooms = v
                
        return max_rooms
    
    
    def meeting_rooms_v2(self, intervals: List[List[int]])-> int:
        start = []
        end = []
        for i in intervals:
            start.append(i[0])
            end.append(i[1])
        start.sort()
        end.sort()
        
        s = 0
        e = 0
        num_rooms = 0
        available = 0
        
        while s < len(start):
            if start[s] < end[e]:
                if available:
                    available -= 1
                else:
                    num_rooms += 1
                s += 1
            else:
                available += 1
                e += 1
                
        return num_rooms
        
        
        
        
    
input = [[0,30],[5,10],[15,20]]
print(Solution().meeting_rooms(input))

input = [[7,10],[2,4]]
print(Solution().meeting_rooms(input))

input = [[7,10],[7,11],[7,12],[7,13],[7,14]]
print(Solution().meeting_rooms(input))