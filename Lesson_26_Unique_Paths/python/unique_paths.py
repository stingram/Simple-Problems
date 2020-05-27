class Solution:
    def unique_paths(self, m: int, n: int) -> int:
        matrix = []
        
        # build zero matrix
        for i in range(m):
            matrix.append([0]*n)
            
        # set top row to one
        matrix[0][:] = [1]*n
         
         # set left column to one
        for i in range(m):
            matrix[i][0] = 1
            
        # update matrix
        for i in range(1,m):
            for j in range(1,n):
                matrix[i][j] = matrix[i-1][j]+ matrix[i][j-1]
                
        # return
        return matrix[m-1][n-1]
    
print(Solution().unique_paths(7,3))