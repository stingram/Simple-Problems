def is_safe(r,c,board):
    if r not in board.rows and c not in board.cols and (r+c) not in board.diag_asc and (r-c) not in board.diag_desc:
        return True
    return False 


def queens_helper(N,row, board):
    
    # base case, all queens have been placed
    # return true
    if row >= N:
        return True

    # try placing queen in all columns for given row
    for i in range(N):
        if is_safe(row,i, board):
            
            # place queens
            board.rows[row] = True
            board.cols[i] = True
            board.diag_asc[row+i] = True 
            board.diag_desc[row-i] = True
            board.positions[row] = i        
    
            # recurse on rest of rows
            if queens_helper(N,row+1,board):
                return True
            
            
            # if placeing queen at (row, i) doesn't lead
            # to a solution, undo our placement
            board.rows.pop(row, None)
            board.cols.pop(i, None)
            board.diag_asc.pop(row+i, None) 
            board.diag_desc.pop(row-i, None)
            board.positions[row] = 0 
        
        
    # the queen could not be placed, so we return false
    return False

class Board:
    def __init__(self, N):
        self.rows = {}
        self.cols = {}
        self.diag_asc = {}
        self.diag_desc = {}
        self.positions = [0]*N
        

def n_queens(N):
    board = Board(N)
    queens_helper(N, 0, board)
    return board.positions

    
    
N = 4 
print(n_queens(N))
