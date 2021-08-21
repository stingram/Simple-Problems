

def maze_paths(maze):
    paths = [[0]*len(row) for row in maze]
    paths[0][0] = 1
    
    for i, row in enumerate(maze):
        for j, col in enumerate(maze[i]):
        
            if i == 0 and j == 0:
                continue    
            
            if maze[i][j] == 0:
                # top row
                if i == 0:
                    paths[i][j] = paths[i][j-1]
                    
                # left column
                elif j == 0:
                    paths[i][j] = paths[i-1][j]
                
                else:
                    paths[i][j] = paths[i][j-1] + paths[i-1][j]
                
    return paths[-1][-1]


print(maze_paths([[0, 1, 0],
                  [0, 0, 1],
                  [0, 0, 0]]))