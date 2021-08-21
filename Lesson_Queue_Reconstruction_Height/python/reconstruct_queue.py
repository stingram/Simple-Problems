# we place people in the queue first based on height, so for example, all the 7 ft are put in
# then all the six ft people are put in, then the 5ft people. We can do it this way because short
# people are invisible to tall people

class Solution(object):
    def construct_queue(self, people):
        # we can use custome key for sort function to sort
        # Note we can have multiple sequential constraints in our lambda function for sorting
        # So the first term, -x[0] means put people in descending order based on height,
        # now that all same-height people are grouped, further sort each group in ascending order by the # of people in front 
        people=sorted(people,key= lambda x: (-x[0], x[1]))  # nlg(n)
        print(people)
        
        # now we can scan this newly sorted array and use python's insert function to place each person based on k value
        # we can do insert because we're always going to be
        ans = []
        for p in people: # O(n)
            ans.insert(p[1], p) # Here, p is inserted to the list at the p[1] index. All the elements after p are shifted to the right.
        return ans
    
    
# Time: O(n*lg(n))
# Space: O(n)





# first number in each pair is a person's height
# second number is the number of people the person sees in front of them (taller people do not see shorter people)
people = [[7, 0], [4, 4], [7, 1], [5, 0], [6, 1], [5, 2]]
print(Solution().construct_queue(people))
# [[5,0], [7, 0], [5, 2], [6, 1], [4, 4], [7, 1]]