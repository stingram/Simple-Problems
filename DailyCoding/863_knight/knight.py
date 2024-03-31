def build_board(start_x, start_y, board_size, k):
    board = [[[0 for _ in range(k+1)] for _ in range(board_size)] for _ in range(board_size)]
    board[start_x][start_y][0] = 1.0
    return board

def update_board(board,board_size,i):
    for r in range(board_size):
        for c in range(board_size):
            if board[r][c][i-1] != 0:
                make_move(board,board_size,r,c,i)
    return 

def make_move(board,board_size,row,col,i):
    delta_x = [1, 1, 2, 2, -1, -1, -2, -2,]
    delta_y = [2,-2, 1,-1,  2, -2,  1, -1]
    
    for dx,dy in zip(delta_x, delta_y):
        new_r = row+dx
        new_c = col+dy
        if 0 <= new_r < board_size and 0 <= new_c < board_size:
            board[new_r][new_c][i] += (1./8)*board[row][col][i-1]
    return

def knight_on_board_prob(start_x, start_y, board_size, k):
    board = build_board(start_x, start_y, board_size, k)
    for i in range(1,k+1):
        update_board(board,board_size,i)
        
    prob = 0
    for row in range(board_size):
        for col in range(board_size):
            prob += board[row][col][k]
    return prob



board_size = 3
k = 1
start_x = 1
start_y = 1
print(f"Prob is {knight_on_board_prob(start_x,start_y,board_size,k)} after {k} moves.")
