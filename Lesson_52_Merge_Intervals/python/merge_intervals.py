from typing import List

class Solution:
    def merge_intervals(self, intervals: List[List[int]]) -> List[List[int]]:
        res = []
        S = []
        E = []
        for l in intervals:
            S.append(l[0])
            E.append(l[1])
        S = sorted(S)
        E = sorted(E)
        
        i = 0
        unmerged = set([i for i in range(len(S))])
        while(i < len(S) - 1):
            s=i+1
            temp_l = []
            print("I: {}".format(i), flush=True)
            print("S: {}".format(s), flush=True)
            while(E[i]>=S[s]):
                print("HERE", flush=True)

                temp_l.append(i)
                temp_l.append(s)
                unmerged.remove(i)
                unmerged.remove(s)
                s += 1

            if temp_l != []:
                res.append(temp_l)
            i = s
        
        ret = []
        for l in res:
            ret.append([intervals[l[0]][0],intervals[l[-1]][1]])
        print(unmerged)
        for l in unmerged:
            ret.append(intervals[l])
        return ret

    def take_first(self, elem):
        return elem[0]

    def merge_intervals_v2(self, intervals: List[List[int]]) -> List[List[int]]:

        # Sort input lists by start times
        intervals.sort(key=self.take_first)
        
        # Results 
        res = []
        
        # for every interval
        for interval in intervals:
            
            # if we have no interval at OR 
            # if the End time in the last item of result is less
            # than the start time of this current interval
            if not res or res[-1][1] < interval[0]:
                
                # add this interval to results since there isn't 
                # an overlap
                res.append(interval)
            else:
                
                # make the end time of the last item in result be the
                # maximum of either the end time of the last item in result
                # OR the end time of this interval we are are looking at
                res[-1][1] = max(res[-1][1], interval[1])
        
        # return results
        return res
    
in_list = [[1,3],[2,6],[8,10],[15,18]]
print(Solution().merge_intervals_v2(in_list))

in_list = [[1,4],[4,5]]
print(Solution().merge_intervals_v2(in_list))